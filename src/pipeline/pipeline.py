import cv2
import numpy as np
from enum import Enum
from ..helper import distance
from typing import Dict, Tuple
from .segmenter import Segmentor
from .inpainter import Inpainter
from .corner_provider import CornerProvider

class Pipeline:
    # def __init__(self):
    #     # self.corner_provider = CornerProvider("Corner Selection Preview")
    #     # self.inpainter = Inpainter()
    #     # self.foreground_remover = Segmentor()
        

    def set_next(self, step):
        self._next_step = step
    


    def process(self, image: np.ndarray) -> np.ndarray:
        skip_step = False
        
        skip_step_handler = SkipStepHandler(skip_step)
        corner_provider_handler = CornerProviderHandler()
        foreground_remover_handler = ForegroundRemoverHandler()
        idealize_colors_handler = IdealizeColorsHandler()
        inpainter_handler = InpainterHandler()
        final_handler = FinalHandler()

        skip_step_handler.set_successor(corner_provider_handler)
        corner_provider_handler.set_successor(foreground_remover_handler)
        foreground_remover_handler.set_successor(idealize_colors_handler)
        idealize_colors_handler.set_successor(inpainter_handler)
        inpainter_handler.set_successor(final_handler)

        result = final_handler.handle(image)
        return result


class ImageHandler:
    def __init__(self, successor=None):
        self._successor = successor

    def set_successor(self, successor):
        self._successor = successor

    def handle(self, image):
        raise NotImplementedError()
    
class SkipStepHandler(ImageHandler):
    def __init__(self, skip_condition: bool, successor: ImageHandler = None):
        super().__init__(successor)
        self._skip_condition = skip_condition
    
    def handle(self, image: np.ndarray) -> np.ndarray:
        if self._skip_condition:
            return image
        else:
            return self._successor.handle(image)
        
class CornerProviderHandler(ImageHandler):
    def __init__(self):
        self.corner_provider = CornerProvider(gui_window_name="test")
    def handle(self, image):
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        return self._successor.handle(whiteboard)

class ForegroundRemoverHandler(ImageHandler):
    def __init__(self):
        self.foreground_remover = Segmentor()
    def handle(self, image):
        foreground_mask = self.foreground_remover.segment(image)
        return self._successor.handle((image, foreground_mask))

class IdealizeColorsHandler(ImageHandler):
    def __init__(self, successor=None):
        super().__init__(successor)
    def handle(self, data):
        whiteboard, foreground_mask = data
        whiteboard = idealize_colors(whiteboard, IdealizeColorsMode.MASKING)
        return self._successor.handle((whiteboard, foreground_mask))

class InpainterHandler(ImageHandler):
    def __init__(self):
        self.inpainter = Inpainter()
    def handle(self, data):
        whiteboard, foreground_mask = data
        whiteboard = self.inpainter.inpaint_missing(whiteboard, foreground_mask)
        return whiteboard
    
class FinalHandler (ImageHandler):
    def __init__(self, successor=None):
        super().__init__(successor)
    def handle(self, image):
        return image

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
