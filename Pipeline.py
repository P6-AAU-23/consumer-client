from CornerProvider import CornerProvider
from typing import Dict, Tuple
from helper import distance
import numpy as np
import cv2


class Pipeline:
    def __init__(self):
        self.corner_provider = CornerProvider("Corner Selection Preview")

    def process(self, image):
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        whiteboard = remove_foreground(image) 
        whiteboard = idealize_colors(whiteboard)
        whiteboard = inpaint_missing(whiteboard)
        return whiteboard


def quadrilateral_to_rectangle(
    image: np.ndarray, corners: Dict[str, Tuple[int, int]]
) -> np.ndarray:
    """
    Warps a quadrilateral region in the input image into a rectangular shape using a perspective transformation.

    :param image: A numpy array representing the input image.
    :param corners: A dictionary containing 4 corner points defining the quadrilateral region in the input image.
                    The keys should be 'upper_left', 'upper_right', 'lower_right', 'lower_left', and the values are
                    tuples containing the x and y coordinates (integers) of each corner point.
    :return: A numpy array representing the output image with the quadrilateral region warped into a rectangular shape.
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
    return image


def idealize_colors(image: np.ndarray) -> np.ndarray:
    return image


def inpaint_missing(image: np.ndarray) -> np.ndarray:
    return image
