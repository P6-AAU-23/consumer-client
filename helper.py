from typing import Dict, Tuple
import numpy as np
import cv2

def distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    :param pt1: A tuple representing the first point (x1, y1).
    :param pt2: A tuple representing the second point (x2, y2).
    :return: The Euclidean distance between the two points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def quadrilateral_to_rectangle(image: np.ndarray, corners: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """
    Warps a quadrilateral region in the input image into a rectangular shape using a perspective transformation.

    :param image: A numpy array representing the input image.
    :param corners: A dictionary containing 4 corner points defining the quadrilateral region in the input image.
                    The keys should be 'upper_left', 'upper_right', 'lower_right', 'lower_left', and the values are
                    tuples containing the x and y coordinates (integers) of each corner point.
    :return: A numpy array representing the output image with the quadrilateral region warped into a rectangular shape.
    """
    width_upper = distance(corners['upper_left'], corners['upper_right'])
    width_lower = distance(corners['lower_left'], corners['lower_right'])
    max_width = int(max(width_upper, width_lower))
    height_left = distance(corners['upper_left'], corners['lower_left'])
    height_right = distance(corners['upper_right'], corners['lower_right'])
    max_height = int(max(height_left, height_right))
    target_corners = np.array([
            (0, 0),
            (max_width, 0),
            (max_width, max_height),
            (0, max_height)
        ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.float32(list(corners.values())), target_corners) # type: ignore
    out = cv2.warpPerspective(image, H, (max_width, max_height)) # type: ignore
    return out

