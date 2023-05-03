import cv2
import math
import numpy as np
from enum import Enum, auto

from ..helper import distance
from typing import Dict, Optional, Tuple
from .segmenter import Segmentor
from .inpainter import Inpainter
from .corner_provider import CornerProvider


class Pipeline:

    def __init__(self):
        self.corner_provider = CornerProvider("Corner Selection Preview")
        self.inpainter = Inpainter()
        self.foreground_remover = Segmentor()

    def process(self, image: np.ndarray) -> np.ndarray:
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        foreground_mask = self.foreground_remover.segment(whiteboard)
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


def scale_saturation(image: np.ndarray, amount: float) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGRA2HSV)  # type: ignore
    # Increase the saturation by amount%
    hsv_image[..., 1] = hsv_image[..., 1] * amount
    output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGRA)  # type: ignore
    return output_image


def fullness(whiteboard: np.ndarray) -> float:
    """Calculate the fullness of a whiteboard, that is, the ratio of the whiteboard that has been drawn on.

    Args:
        whiteboard (np.ndarray): The input whiteboard in BGR format.

    Returns:
        float: The fullness of the whiteboard as a ratio.
    """
    image_grey = cv2.cvtColor(whiteboard, cv2.COLOR_BGR2GRAY)  # type: ignore
    binary_image = cv2.adaptiveThreshold(  # type: ignore
        image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4  # type: ignore
    )
    binary_image = cv2.medianBlur(binary_image, 3)  # type: ignore
    binary_image = cv2.bitwise_not(binary_image)  # type: ignore
    return np.count_nonzero(binary_image == 255) / size(whiteboard)


def size(image: np.ndarray) -> int:
    height, width, _ = image.shape
    return height * width


def binarize(image: np.ndarray) -> np.ndarray:
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore
    binary_image = cv2.adaptiveThreshold(  # type: ignore
        image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4  # type: ignore
    )
    binary_image = cv2.medianBlur(binary_image, 3)  # type: ignore
    binary_image = cv2.bitwise_not(binary_image)  # type: ignore
    return binary_image

class AdaptiveSignificantPeakFilter:

    def __init__(self) -> None:  # noqa: N803
        self._significant_change_filter1 = MeanAdaptiveSignificantChangeFilter(2, 2)
        self._significant_change_filter2 = SignificantChangeFilter(0, 0.005)
        self._peak_filter = DelayedPeakFilter()

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        _image = self._significant_change_filter1.filter(image)
        if _image is not None:
            _image = self._significant_change_filter2.filter(_image)
        if _image is not None:
            return self._peak_filter.filter(_image)
        return None


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.count < 2:
            return float('nan')
        return self.M2 / (self.count - 1)

    def get_standard_deviation(self):
        return math.sqrt(self.get_variance())


class SignificantPeakFilter:
    """
    A filter that returns a peak image if there is a significant change in fullness.

    The filter combines the functionality of SignificantChangeFilter and DelayedPeakFilter.
    It first applies the SignificantChangeFilter to detect significant changes in fullness, and
    then applies the DelayedPeakFilter to detect peaks in the significant changes.

    Attributes:
        _significant_change_filter (SignificantChangeFilter): A filter to detect significant changes in fullness.
        _peak_filter (DelayedPeakFilter): A filter to detect peaks in fullness.
    """

    def __init__(self, climbing_Δ_threshold: float, descending_Δ_threshold: float) -> None:  # noqa: N803
        """
        Initialize the SignificantPeakFilter instance.

        Args:
            climbing_Δ_threshold (float): Threshold for detecting significant increases in fullness.
            descending_Δ_threshold (float): Threshold for detecting significant decreases in fullness.
        """
        self._significant_change_filter = SignificantChangeFilter(climbing_Δ_threshold, descending_Δ_threshold)
        self._peak_filter = DelayedPeakFilter()

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply the significant peak filter to the given image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            Optional[np.ndarray]: The peak image when detected, otherwise None.
        """
        _image = self._significant_change_filter.filter(image)
        if _image is None:
            return None
        return self._peak_filter.filter(_image)

class σAdaptiveSignificantChangeFilter:
    def __init__(self, climbing_sensitivity: float, descending_sensitivity: float):
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._stats = RunningStats() 
        self._climbing_sensitivity = climbing_sensitivity
        self._descending_sensitivity = descending_sensitivity

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        self._stats.update(abs_Δ_fullness)
        self._significant_change_filter._climbing_Δ_threshold = self._climbing_sensitivity * self._stats.get_standard_deviation()
        self._significant_change_filter._descending_Δ_threshold = self._descending_sensitivity* self._stats.get_standard_deviation()
        self._last_image = image
        return self._significant_change_filter.filter(image)


class MeanAdaptiveSignificantChangeFilter:
    def __init__(self, climbing_sensitivity: float, descending_sensitivity: float):
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._mean_Δ_fullness = 0
        self._count = 0
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._climbing_sensitivity = climbing_sensitivity
        self._descending_sensitivity = descending_sensitivity

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        self._count += 1
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        Δ = abs_Δ_fullness - self._mean_Δ_fullness
        self._mean_Δ_fullness += (Δ / self._count)
        self._significant_change_filter._climbing_Δ_threshold = self._climbing_sensitivity * self._mean_Δ_fullness
        self._significant_change_filter._descending_Δ_threshold = self._descending_sensitivity* self._mean_Δ_fullness
        self._last_image = image
        print(self._mean_Δ_fullness)
        return self._significant_change_filter.filter(image)


class EmaAdaptiveSignificantChangeFilter:
    def __init__(self, N: float, climbing_sensitivity: float, descending_sensitivity: float):
        self._significant_change_filter = SignificantChangeFilter(1, 1)
        self._ema_Δ_fullness = 0
        self._last_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        self._α = 2 / (N + 1)
        self._climbing_sensitivity = climbing_sensitivity
        self._descending_sensitivity = descending_sensitivity

    def filter(self, image: np.ndarray) -> Optional[np.ndarray]:
        abs_Δ_fullness = abs(fullness(image) - fullness(self._last_image))  # noqa: N806
        self._ema_Δ_fullness = (1 - self._α) * self._ema_Δ_fullness + self._α * abs_Δ_fullness
        self._significant_change_filter._climbing_Δ_threshold = self._climbing_sensitivity * self._ema_Δ_fullness
        self._significant_change_filter._descending_Δ_threshold = self._descending_sensitivity * self._ema_Δ_fullness
        self._last_image = image
        print(self._ema_Δ_fullness)
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
        if self._mode is self.Mode.CLIMBING:
            if fullness(self._last_image) <= fullness(image):
                self._mode = self.Mode.CLIMBING
            elif fullness(self._last_image) > fullness(image):
                self._mode = self.Mode.DESCENDING
                return self._last_image
        elif self._mode is self.Mode.DESCENDING:
            if fullness(self._last_image) < fullness(image):
                self._mode = self.Mode.CLIMBING
            elif fullness(self._last_image) >= fullness(image):
                self._mode = self.Mode.DESCENDING
        self._last_image = image
        return None
