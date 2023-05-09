import cv2
import torch
import numpy as np
from pathlib import Path
from ..Africa import Slave
from enum import Enum, auto
from torchvision import transforms
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from .corner_provider import CornerProvider
from ..helper import distance, binarize, apply_mask, AvgBgr
from ..helper import RunningStats, dilate_black_regions, fullness, write_path_with_date_and_time


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

        actor_ref = Slave.start()
        #foreground_mask = actor_ref.ask(self.remove(image_layers["whiteboard"])

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
        mask = dilate_black_regions(mask, iterations=100)

        return mask


class ColorAdjuster(ImageProcessor):
    def __init__(self, avg_bgr: AvgBgr, saturate_input: float, bright_input: int):
        super().__init__()
        self.saturate_input = saturate_input
        self.bright_input = bright_input
        self.avg_color = avg_bgr

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image_layers["whiteboard"] = self.color_adjust(image_layers["whiteboard"])
        return image_layers

    def color_adjust(self, image: cv2.Mat) -> cv2.Mat:
        """
        Apply white balancing to an input image using a pre-calculated average of B, G, R channels.
        Also Applying saturation, brightness, and normalization.

        :param image: Input image as a numpy array.
        :type image: numpy.ndarray
        :return: Color adjusted image as a numpy array.
        :rtype: numpy.ndarray
        """

        # Applying white balancing
        result = self.avg_color.white_balance(image)

        # Up saturation & brightness
        saturation_boost = self.saturate_input
        brightness = self.bright_input

        if self.saturate_input != 1 or self.bright_input != 0:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            if self.saturate_input != 1:
                result[:, :, 1] = cv2.convertScaleAbs(result[:, :, 1], alpha=float(saturation_boost))
            if self.bright_input != 0:
                result[:, :, 2] = cv2.add(result[:, :, 2], int(brightness))
            result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

        # Normalize image
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result


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


class WipeSaver(ImageProcessor):
    def __init__(self, path: str):
        super().__init__()
        self._path = Path(path)
        self._n_filter = MeanAdaptiveSignificantChangeFilter(2, 2)
        self._σ_filter = σAdaptiveSignificantChangeFilter(0, 0.25)
        self._peak_filter = DelayedPeakFilter()

    def _process(self, image_layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        whiteboard = self._n_filter.filter(image_layers["whiteboard"])
        if whiteboard is not None:
            whiteboard = self._σ_filter.filter(whiteboard)
        if whiteboard is not None:
            whiteboard = self._peak_filter.filter(whiteboard)
        if whiteboard is not None:
            cv2.imshow("I SAVED THIS", whiteboard)  # type: ignore
            cv2.imwrite(write_path_with_date_and_time("wipe", self._path), whiteboard)
        return image_layers


class σAdaptiveSignificantChangeFilter:  # noqa: N801
    def __init__(self, σ_climbing_threshold: float, σ_descending_threshold: float):
        self._stats = RunningStats()
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._σ_climbing_threshold = σ_climbing_threshold
        self._σ_descending_threshold = σ_descending_threshold

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        self._stats.update(abs_Δ_fullness)
        self._significant_change_filter._climbing_Δ_threshold = \
            self._σ_climbing_threshold * self._stats.get_standard_deviation()
        self._significant_change_filter._descending_Δ_threshold = \
            self._σ_descending_threshold * self._stats.get_standard_deviation()
        self._last_image = image
        return self._significant_change_filter.filter(image)


class MeanAdaptiveSignificantChangeFilter:
    def __init__(self, mean_climbing_threshold: float, mean_descending_threshold: float):
        self._stats = RunningStats()
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._mean_climbing_threshold = mean_climbing_threshold
        self._mean_descending_threshold = mean_descending_threshold

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        self._stats.update(abs_Δ_fullness)
        self._significant_change_filter._climbing_Δ_threshold = \
            self._mean_climbing_threshold * self._stats.get_mean()
        self._significant_change_filter._descending_Δ_threshold = \
            self._mean_descending_threshold * self._stats.get_mean()
        self._last_image = image
        return self._significant_change_filter.filter(image)


class EmaAdaptiveSignificantChangeFilter:
    def __init__(self, window: float, ema_climbing_threshold: float, ema_descending_threshold: float):
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._ema_Δ_fullness = 0
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._α = 2 / (window + 1)
        self._climbing_sensitivity = ema_climbing_threshold
        self._descending_sensitivity = ema_descending_threshold

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        self._ema_Δ_fullness = (1 - self._α) * self._ema_Δ_fullness + self._α * abs_Δ_fullness
        self._significant_change_filter._climbing_Δ_threshold = \
            self._climbing_sensitivity * self._ema_Δ_fullness
        self._significant_change_filter._descending_Δ_threshold = \
            self._descending_sensitivity * self._ema_Δ_fullness
        self._last_image = image
        return self._significant_change_filter.filter(image)


class SignificantChangeFilter:
    """
    A filter that returns an image only if there is a significant change in its fullness.

    The filter considers an image change as significant if the change in fullness surpasses a
    predefined threshold. It maintains the last significant image and returns the current image
    if the change is significant; otherwise, it returns None.

    Attributes:
        _climbing_Δ_threshold (float): Threshold for increasing fullness.
        _descending_Δ_threshold (float): Threshold for decreasing fullness.
        _last_significant_image (np.ndarray): The last significant image.
    """

    def __init__(self, climbing_Δ_threshold: float, descending_Δ_threshold: float) -> None:  # noqa:N803
        """
        Initialize the SignificantChangeFilter instance.

        Args:
            climbing_Δ_threshold (float): Threshold for increasing fullness (0 <= value <= 1).
            descending_Δ_threshold (float): Threshold for decreasing fullness (0 <= value <= 1).
        """
        assert 0 <= climbing_Δ_threshold and climbing_Δ_threshold <= 1
        assert 0 <= descending_Δ_threshold and descending_Δ_threshold <= 1
        self._climbing_Δ_threshold = climbing_Δ_threshold
        self._descending_Δ_threshold = descending_Δ_threshold
        # Initialize last_image to a 10x10 white image
        self._last_significant_image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply the significant change filter to the given image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            Optional[np.ndarray]: The input image if the change is significant, otherwise None.
        """
        Δ_fullness = fullness(image) - fullness(self._last_significant_image)  # noqa: N806
        if Δ_fullness > 0:
            Δ_fullness_threshold = self._climbing_Δ_threshold  # noqa: N806
        elif Δ_fullness <= 0:
            Δ_fullness_threshold = self._descending_Δ_threshold  # noqa: N806
        if Δ_fullness_threshold < abs(Δ_fullness):
            self._last_significant_image = image
            return self._last_significant_image
        return None


class DelayedPeakFilter:
    """
    A filter that returns an image when it detects a peak in fullness after a delay.

    The filter detects a peak in fullness by switching between two modes: climbing and descending.
    When in climbing mode, the filter checks if the fullness of the current image is less than the
    fullness of the previous image. If so, it switches to descending mode and returns the peak image.
    In descending mode, it switches back to climbing mode when the fullness of the current image
    becomes greater than the previous image.

    Attributes:
        _last_image (np.ndarray): The last image received by the filter.
        _last_peak (np.ndarray): The last detected peak in fullness.
        _mode (Mode): The current mode of the filter, either CLIMBING or DESCENDING.
    """

    class Mode(Enum):
        CLIMBING = auto()
        DESCENDING = auto()

    def __init__(self):
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._mode = self.Mode.CLIMBING

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply the delayed peak filter to the given image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            Optional[np.ndarray]: The peak image when detected, otherwise None.
        """
        output = None
        if self._mode is self.Mode.CLIMBING:
            if fullness(self._last_image) <= fullness(image):
                self._mode = self.Mode.CLIMBING
            elif fullness(self._last_image) > fullness(image):
                self._mode = self.Mode.DESCENDING
                output = self._last_image
        elif self._mode is self.Mode.DESCENDING:
            if fullness(self._last_image) < fullness(image):
                self._mode = self.Mode.CLIMBING
            elif fullness(self._last_image) >= fullness(image):
                self._mode = self.Mode.DESCENDING
        self._last_image = image
        return output
