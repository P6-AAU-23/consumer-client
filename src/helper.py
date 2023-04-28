import os
import cv2
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


def dilate_black_regions(binary_mask: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    inverted_mask = cv2.bitwise_not(binary_mask)  # type: ignore
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # type: ignore
    dilated_inverted_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)  # type: ignore
    dilated_mask = cv2.bitwise_not(dilated_inverted_mask)  # type: ignore
    return dilated_mask


def list_ports() -> list[str]:
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    dev_port = 0
    working_ports = []
    available_ports = []

    i = 0
    while i <= 10: # looking through 10 ports
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            i = 0 # if we find a port, the counter is reset to look through 10 more ports
            is_reading, img = camera.read()
            
            if is_reading:
                working_ports.append(str(dev_port))
            else:
                available_ports.append(str(dev_port))

        i += 1
        dev_port +=1
    return working_ports

def try_int_to_string(self, s):
        try: 
            return int(s)
        except:
            return s