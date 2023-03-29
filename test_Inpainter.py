from Inpainter import Inpainter
import cv2
import numpy as np


def test_inpaint_missing_inpaints_missing():
    # Arrange
    input = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(input, (25, 25), (74, 74), (255, 0, 0), -1)  # type: ignore

    expected = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(expected, (25, 25), (74, 74), (0, 255, 0), -1)  # type: ignore

    mask = np.ones((100, 100, 1), dtype=np.uint8) * 255
    cv2.rectangle(mask, (25, 25), (74, 74), 0, -1)  # type: ignore

    inpainter = Inpainter()
    inpainter._last_image = expected

    # Act
    actual = inpainter.inpaint_missing(input, mask)

    # Assert
    assert np.array_equal(actual, expected)
