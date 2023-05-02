import cv2
import numpy as np
from enum import Enum
from ..helper import distance
from typing import Dict, Tuple
from .segmenter import Segmentor
from .inpainter import Inpainter
from .corner_provider import CornerProvider


class Pipeline:

    def __init__(self):
        self.corner_provider = CornerProvider("Corner Selection Preview")
        self.inpainter = Inpainter()
        self.foreground_remover = Segmentor()

    def process(self, image: np.ndarray, avg_color: avg_bgr) -> np.ndarray:
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        foreground_mask = self.foreground_remover.segment(whiteboard)
        whiteboard = color_adjust(whiteboard, avg_color) # insert param
        whiteboard = idealize_colors(whiteboard, IdealizeColorsMode.MASKING)
        whiteboard = self.inpainter.inpaint_missing(whiteboard, foreground_mask)
        return whiteboard


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


class IdealizeColorsMode(Enum):
    MASKING = 1
    ASSIGN_EXTREME = 2

class avg_bgr:
    def __init__(self, avg_b: float, avg_g: float, avg_r: float):
        self.b = avg_b
        self.g = avg_g
        self.r = avg_r

    def white_balance(self, image: np.ndarray) -> np.ndarray:
        # Split channels
        blue, green, red = cv2.split(image)

        # Calculate scaling factors for each channel
        scale_b = self.g / self.b
        scale_r = self.g / self.r

        # Apply scaling factors to each channel
        blue = cv2.convertScaleAbs(blue, alpha=scale_b)
        red = cv2.convertScaleAbs(red, alpha=scale_r)

        # Merge channels
        result = cv2.merge((blue, green, red))
        return result


def color_adjust(image: np.ndarray, avg_color: avg_bgr, saturate_input: float, bright_input: int) -> np.ndarray:
    """
    Apply white balancing to an input image using a pre-calculated average of B, G, R channels.
    Also Applying saturation, brightness, and normalization.

    :param image: Input image as a numpy array.
    :type image: numpy.ndarray
    :return: Color adjusted image as a numpy array.
    :rtype: numpy.ndarray
    """

    # Applying white balancing
    result = avg_color.white_balance(image)

    # Up saturation & brightness
    saturation_boost = saturate_input
    brightness = bright_input

    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    result[:, :, 1] = cv2.convertScaleAbs(result[:, :, 1], alpha=saturation_boost)
    result[:, :, 2] = cv2.add(result[:, :, 2], brightness)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    # Normalize image
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    return result


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
