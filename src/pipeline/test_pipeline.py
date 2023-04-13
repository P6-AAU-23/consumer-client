from .Pipeline import quadrilateral_to_rectangle
import numpy as np
import cv2


def test_warp_quadrilateral_to_rectangle_no_transformation() -> None:
    """
    Tests the case when the input quadrilateral region is already a rectangle with the same size as the input image.
    In this case, the output should be the same as the input.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (75, 75), (255, 255, 255), -1)  # type: ignore
    corners = {
        "upper_left": (0, 0),
        "upper_right": (100, 0),
        "lower_right": (100, 100),
        "lower_left": (0, 100),
    }
    warped_image = quadrilateral_to_rectangle(image, corners)
    print(image.shape)
    print(warped_image.shape)
    assert np.array_equal(image, warped_image)


def test_quadrilateral_to_rectangle_known_transformation() -> None:
    """
    Tests the case when the input quadrilateral region is a rectangle within the input image, and the output should be
    that rectangular region. The test creates a 100x100 image with a white 50x50 rectangle centered in it and applies
    the perspective warp to extract the 50x50 rectangle. The result is then compared to a 50x50 image of a white rectangle.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (74, 74), (255, 255, 255), -1)  # type: ignore
    corners = {
        "upper_left": (25, 25),
        "upper_right": (75, 25),
        "lower_right": (75, 75),
        "lower_left": (25, 75),
    }
    warped_image = quadrilateral_to_rectangle(image, corners)
    expected_image = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(expected_image, (0, 0), (49, 49), (255, 255, 255), -1)  # type: ignore
    assert np.array_equal(warped_image, expected_image)
