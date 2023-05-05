import cv2
import numpy as np
import torch
from enum import Enum, auto
from typing import Dict, Tuple
from torchvision import transforms
from ..helper import dilate_black_regions
from .corner_provider import CornerProvider
from ..helper import distance, binarize, apply_mask
from abc import ABC, abstractmethod


class IdealizeColorsMode(Enum):
    MASKING = auto()
    ASSIGN_EXTREME = auto()


class ImageProcessor(ABC):
    def __init__(self):
        self._next_processor = None

    def set_next(self, processor: "ImageProcessor") -> "ImageProcessor":
        self._next_processor = processor
        return processor

    def process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed_image = self._process(image_layers)
        if self._next_processor:
            return self._next_processor.process(processed_image)
        return processed_image

    @abstractmethod
    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass


class IdentityProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return image_layers


class PerspectiveTransformer(ImageProcessor):
    def __init__(self, use_gui: bool = True):
        super().__init__()
        self.corner_provider = CornerProvider("Corner Selection Preview", use_gui)

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.corner_provider.update(image_layers["whiteboard"])
        corners = self.corner_provider.get_corners()
        image_layers["whiteboard"] = self.quadrilateral_to_rectangle(
            image_layers["whiteboard"], corners
        )
        return image_layers

    def quadrilateral_to_rectangle(
        self, image: np.ndarray, corners: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        """Warps a quadrilateral region in the input image into a rectangular shape using a perspective transformation.

        Args:
            image (np.ndarray): A numpy array representing the input image.
            corners (Dict[str, Tuple[int, int]]): A dictionary containing 4 corner points defining the quadrilateral
                                                region in the input image. The keys should be 'upper_left', 'upper_right',
                                                'lower_right', 'lower_left', and the values are tuples containing the x and
                                                y coordinates (integers) of each corner point.

        Returns:
            np.ndarray: A numpy array representing the output image with the quadrilateral region warped into a rectangular
                        shape.
        """
        width_upper = distance(corners["upper_left"], corners["upper_right"])
        width_lower = distance(corners["lower_left"], corners["lower_right"])
        max_width = int(max(width_upper, width_lower))
        height_left = distance(corners["upper_left"], corners["lower_left"])
        height_right = distance(corners["upper_right"], corners["lower_right"])
        max_height = int(max(height_left, height_right))
        target_corners = np.array(
            [(0, 0), (max_width, 0), (max_width, max_height), (0, max_height)],
            dtype=np.float32,
        )
        quad_to_rect_transform = cv2.getPerspectiveTransform(  # type: ignore
            np.float32(list(corners.values())), target_corners
        )
        out = cv2.warpPerspective(image, quad_to_rect_transform, (max_width, max_height))  # type: ignore
        return out


class ColorIdealizer(ImageProcessor):
    def __init__(self, color_mode: IdealizeColorsMode):
        super().__init__()
        self.IdealizeColorsMode = color_mode

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image_layers["whiteboard"] = self.idealize_colors(
            image_layers["whiteboard"], self.IdealizeColorsMode
        )
        return image_layers

    def idealize_colors(
        self, image: np.ndarray, mode: IdealizeColorsMode
    ) -> np.ndarray:
        if mode == self.IdealizeColorsMode.MASKING:
            return self.idealize_colors_masking(image)
        if mode == self.IdealizeColorsMode.ASSIGN_EXTREME:
            return self.idealize_colors_assign_extreme(image)
        else:
            return image

    def idealize_colors_masking(self, image: np.ndarray) -> np.ndarray:
        mask = binarize(image)
        masked_image = apply_mask(image, mask)
        return masked_image

    def idealize_colors_assign_extreme(self, image: np.ndarray) -> np.ndarray:
        threshold = 128
        max_val = 255
        # Split the image into B, G, and R channels
        b, g, r = cv2.split(image)  # type: ignore
        # Apply the threshold to each channel
        _, b = cv2.threshold(b, threshold, max_val, cv2.THRESH_BINARY)  # type: ignore
        _, g = cv2.threshold(g, threshold, max_val, cv2.THRESH_BINARY)  # type: ignore
        _, r = cv2.threshold(r, threshold, max_val, cv2.THRESH_BINARY)  # type: ignore
        # Merge the thresholded channels back into a single image
        recolored_image = cv2.merge((b, g, r))  # type: ignore
        return recolored_image


class ForegroundRemover(ImageProcessor):
    def __init__(self):
        super().__init__()
        self.torch_model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_mobilenet_v3_large",
            weights="DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT",
        )
        self.torch_model.eval()

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image_layers["foreground_mask"] = self.remove(image_layers["whiteboard"])
        return image_layers

    def remove(self, img: np.ndarray) -> np.ndarray:
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            self.torch_model.to("cuda")

        with torch.no_grad():
            output = self.torch_model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        prediction_in_numpy = output_predictions.byte().cpu().numpy()

        mask = cv2.inRange(prediction_in_numpy, 0, 0)
        mask = dilate_black_regions(mask, iterations=11)

        return mask


class Inpainter(ImageProcessor):
    def __init__(self):
        super().__init__()
        self._last_image = None

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image_layers["whiteboard"] = self.inpaint_missing(
            image_layers["whiteboard"], image_layers["foreground_mask"]
        )
        return image_layers

    def inpaint_missing(
        self, image: np.ndarray, missing_mask: np.ndarray
    ) -> np.ndarray:
        """Inpaints the missing regions in the input image using the provided binary mask,
        and the last image given to this function.

        Args:
            image (np.ndarray): A numpy array representing the input image.
            missing_mask (np.ndarray): A numpy array representing the binary mask indicating missing regions
                                        (0 for missing regions, non-zero for existing regions).

        Raises:
            ValueError: If the input image and missing_mask have different height and width.

        Returns:
            np.ndarray: A numpy array representing the inpainted image with missing regions filled.
        """
        # If last_image is not set or input image is different shape from last_image
        if self._last_image is None or image.shape != self._last_image.shape:
            # Initialize last_image
            self._last_image = np.ones(image.shape, dtype=np.uint8) * 255
        if image.shape[:2] != missing_mask.shape[:2]:
            raise ValueError(
                "The input image and missing_mask must have the same height and width."
            )
        # Ensure the mask is a binary mask (0 or 255)
        binary_mask = (missing_mask == 0).astype(np.uint8) * 255
        # Apply the mask to the last_image using bitwise operations
        masked_last_image = cv2.bitwise_and(  # type: ignore
            self._last_image, self._last_image, mask=binary_mask
        )
        # Invert the binary_mask to apply it to the input image
        inverted_binary_mask = cv2.bitwise_not(binary_mask)  # type: ignore
        masked_input = cv2.bitwise_and(image, image, mask=inverted_binary_mask)  # type: ignore
        # Combine the masked images to create the inpainted result
        inpainted_image = cv2.add(masked_input, masked_last_image)  # type: ignore
        self._last_image = inpainted_image
        return inpainted_image
