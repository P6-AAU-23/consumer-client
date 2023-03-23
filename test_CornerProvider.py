import pytest
import numpy as np
from CornerProvider import CornerProvider


@pytest.fixture
def cp_initialized():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cp = CornerProvider("test_window", use_gui=False)
    cp._initialize_corners(image)
    return cp


def test_initialize_corners(cp_initialized):
    corners = cp_initialized.get_corners()
    assert corners["upper_left"] == (0, 0)
    assert corners["upper_right"] == (199, 0)
    assert corners["lower_right"] == (199, 99)
    assert corners["lower_left"] == (0, 99)


def test_corners_are_on_image(cp_initialized):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert cp_initialized._corners_are_on_image(image) is True
    cp_initialized.corners["upper_left"] = (-1, -1)
    assert cp_initialized._corners_are_on_image(image) is False


def test_on_corner(cp_initialized):
    assert cp_initialized._on_corner(0, 0) == "upper_left"
    assert cp_initialized._on_corner(199, 0) == "upper_right"
    assert cp_initialized._on_corner(199, 99) == "lower_right"
    assert cp_initialized._on_corner(0, 99) == "lower_left"
    assert cp_initialized._on_corner(50, 50) == "None"


def test_update(cp_initialized):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    preview_image = cp_initialized.update(image)
    # Test that the corner points were drawn on the preview_image
    assert np.any(preview_image[0:10, 0:10])  # upper_left
    assert np.any(preview_image[0:10, 189:200])  # upper_right
    assert np.any(preview_image[89:100, 189:200])  # lower_right
    assert np.any(preview_image[89:100, 0:10])  # lower_left
