from typing import Tuple
import numpy as np
import os
import cv2
from datetime import datetime
from pathlib import Path


def distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    :param pt1: A tuple representing the first point (x1, y1).
    :param pt2: A tuple representing the second point (x2, y2).
    :return: The Euclidean distance between the two points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def write_path_with_date_and_time(name: str, path: Path):
    """
    Generates a path with a unique name with the time and date for an image you want to save.

    :param name: Name of the file you want save.
    :param path: The path to the location you want to save.
    :return: The full path with unique name and date/time
    """
    now = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    full_name = name + now + ".jpg"
    full_path = path / full_name
    return str(uniquify_file_name(full_path))

def write_path_with_unique_name(name: str, path: Path):
    """
    Generates a path with a unique name for an image you want to save.

    :param name: Name of the file you want save.
    :param path: The path to the location you want to save.
    :return: The full path with unique name
    """
    full_name = name + ".jpg"
    full_path = path / full_name
    return str(uniquify_file_name(full_path))


def uniquify_file_name(path):
    """
    Generates a unique name for a path with a number, for files with same name i.e. img(1).jpg

    :param path: full path with a name of the file.
    :return: The full path with unique name
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def dilate_black_regions(binary_mask, kernel_size=(3, 3), iterations=1):
    inverted_mask = cv2.bitwise_not(binary_mask)  # type: ignore
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # type: ignore
    dilated_inverted_mask = cv2.dilate(inverted_mask, kernel, iterations=iterations)  # type: ignore
    dilated_mask = cv2.bitwise_not(dilated_inverted_mask)  # type: ignore
    return dilated_mask
