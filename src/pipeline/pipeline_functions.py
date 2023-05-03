import cv2
import torch
import numpy as np
from enum import Enum
from ..helper import distance
from typing import Dict, Tuple
from torchvision import transforms
from ..helper import dilate_black_regions

class IdealizeColorsMode(Enum):
    MASKING = 1
    ASSIGN_EXTREME = 2


def quadrilateral_to_rectangle(
    image: np.ndarray, corners: Dict[str, Tuple[int, int]]
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
    quad_to_rect_transform = cv2.getPerspectiveTransform(np.float32(list(corners.values())), target_corners)  # type: ignore
    out = cv2.warpPerspective(image, quad_to_rect_transform, (max_width, max_height))  # type: ignore
    return out


def remove_foreground(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    return np.ones((height, width, 1), dtype=np.uint8) * 255


def idealize_colors(image: np.ndarray, mode: IdealizeColorsMode) -> np.ndarray:
    if mode == IdealizeColorsMode.MASKING:
        return idealize_colors_masking(image)
    if mode == IdealizeColorsMode.ASSIGN_EXTREME:
        return idealize_colors_assign_extreme(image)
    else:
        return image


def idealize_colors_masking(image: np.ndarray) -> np.ndarray:
    mask = binarize(image)
    masked_image = apply_mask(image, mask)
    return masked_image


def idealize_colors_assign_extreme(image: np.ndarray) -> np.ndarray:
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


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_image = cv2.bitwise_and(image, image, mask=mask)  # type: ignore
    masked_image[mask == 0] = 255  # make the masked area white
    return masked_image


def binarize(image: np.ndarray) -> np.ndarray:
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore
    binary_image = cv2.adaptiveThreshold(  # type: ignore
        image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4  # type: ignore
    )
    binary_image = cv2.medianBlur(binary_image, 3)  # type: ignore
    binary_image = cv2.bitwise_not(binary_image)  # type: ignore
    return binary_image


def scale_saturation(image: np.ndarray, amount: float) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGRA2HSV)  # type: ignore
    # Increase the saturation by amount%
    hsv_image[..., 1] = hsv_image[..., 1] * amount
    output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGRA)  # type: ignore
    return output_image


def inpaint_missing(
        image: np.ndarray, missing_mask: np.ndarray, last_image: cv2.Mat
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
    if last_image is None or image.shape != last_image.shape:
        # Initialize last_image
        last_image = np.ones(image.shape, dtype=np.uint8) * 255
    if image.shape[:2] != missing_mask.shape[:2]:
        raise ValueError(
            "The input image and missing_mask must have the same height and width."
        )
    # Ensure the mask is a binary mask (0 or 255)
    binary_mask = (missing_mask == 0).astype(np.uint8) * 255
    # Apply the mask to the last_image using bitwise operations
    masked_last_image = cv2.bitwise_and(  # type: ignore
        last_image, last_image, mask=binary_mask
    )
    # Invert the binary_mask to apply it to the input image
    inverted_binary_mask = cv2.bitwise_not(binary_mask)  # type: ignore
    masked_input = cv2.bitwise_and(image, image, mask=inverted_binary_mask)  # type: ignore
    # Combine the masked images to create the inpainted result
    inpainted_image = cv2.add(masked_input, masked_last_image)  # type: ignore
    last_image = inpainted_image
    return inpainted_image


def segment(torch_model, img: np.ndarray) -> np.ndarray:
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
        torch_model.to("cuda")

    with torch.no_grad():
        output = torch_model(input_batch)["out"][0]
    output_predictions = output.argmax(0)

    prediction_in_numpy = output_predictions.byte().cpu().numpy()

    mask = cv2.inRange(prediction_in_numpy, 0, 0)
    mask = dilate_black_regions(mask, iterations=11)

    return mask
