from typing import Tuple
import numpy as np
import os
import cv2


def distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points in 2D space.

    Args:
        pt1 (tuple): A tuple representing the first point (x1, y1).
        pt2 (tuple): A tuple representing the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def uniquify_file_name(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def dilate_black_regions(binary_mask: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    inverted_mask = cv2.bitwise_not(binary_mask)  # type: ignore
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # type: ignore
    dilated_inverted_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)  # type: ignore
    dilated_mask = cv2.bitwise_not(dilated_inverted_mask)  # type: ignore
    return dilated_mask
