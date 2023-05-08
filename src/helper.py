import os
import cv2
import math
import numpy as np
from typing import Tuple
from pathlib import Path
from datetime import datetime


def distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points in 2D space.

    Args:
        pt1 (tuple): A tuple representing the first point (x1, y1).
        pt2 (tuple): A tuple representing the second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def write_path_with_date_and_time(name: str, path: Path) -> str:
    """Generates a path with a unique name with the time and date for an image you want to save.

    Args:
        name: Name of the file you want save.
        path: The path to the location you want to save.

    Returns:
        str: String of the full path with unique name and date/time.
    """
    now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    full_name = name + now + ".jpg"
    full_path = path / full_name
    return str(uniquify_file_name(full_path))


def write_path_with_unique_name(name: str, path: Path) -> str:
    """Generates a path with a unique name for an image you want to save.

    Args:
        name: Name of the file you want save.
        path: The path to the location you want to save.

    Returns:
        str: String of path with unique name
    """
    full_name = name + ".jpg"
    full_path = path / full_name
    return str(uniquify_file_name(full_path))


def uniquify_file_name(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def dilate_black_regions(
    binary_mask: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1
) -> np.ndarray:
    inverted_mask = cv2.bitwise_not(binary_mask)  # type: ignore
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # type: ignore
    dilated_inverted_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)  # type: ignore
    dilated_mask = cv2.bitwise_not(dilated_inverted_mask)  # type: ignore
    return dilated_mask


def list_ports() -> list[str]:
    dev_port = 0
    working_ports = []
    available_ports = []

    i = 0
    while i <= 10:  # looking through 10 ports
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            i = 0  # if we find a port, the counter is reset to look through 10 more ports
            is_reading, img = camera.read()

            if is_reading:
                working_ports.append(str(dev_port))
            else:
                available_ports.append(str(dev_port))

        i += 1
        dev_port += 1
    return working_ports


def try_int_to_string(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def try_float_to_string(s: str) -> str | float:
    try:
        return float(s)
    except ValueError:
        return s


def binarize(image: np.ndarray) -> np.ndarray:
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # type: ignore
    binary_image = cv2.adaptiveThreshold(  # type: ignore
        image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4  # type: ignore
    )
    binary_image = cv2.medianBlur(binary_image, 3)  # type: ignore
    binary_image = cv2.bitwise_not(binary_image)  # type: ignore
    return binary_image


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_image = cv2.bitwise_and(image, image, mask=mask)  # type: ignore
    masked_image[mask == 0] = 255  # make the masked area white
    return masked_image


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> float:
        return self.mean

    def get_variance(self) -> float:
        if self.count < 2:
            return float('nan')
        return self.M2 / (self.count - 1)

    def get_standard_deviation(self) -> float:
        return math.sqrt(self.get_variance())


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