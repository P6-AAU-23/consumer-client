from typing import Any
from src.pipeline.pipeline_modules import DelayedPeakFilter, PerspectiveTransformer, SignificantChangeFilter
import numpy as np
import cv2
import pytest


def test_warp_quadrilateral_to_rectangle_no_transformation() -> None:
    """
    Tests the case when the input quadrilateral region is already a rectangle with the same size as the input image.
    In this case, the output should be the same as the input.
    """
    corner_provider = PerspectiveTransformer(use_gui=False)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (75, 75), (255, 255, 255), -1)  # type: ignore
    corners = {
        "upper_left": (0, 0),
        "upper_right": (100, 0),
        "lower_right": (100, 100),
        "lower_left": (0, 100),
    }
    warped_image = corner_provider.quadrilateral_to_rectangle(image, corners)
    print(image.shape)
    print(warped_image.shape)
    assert np.array_equal(image, warped_image)


def test_quadrilateral_to_rectangle_known_transformation() -> None:
    """
    Tests the case when the input quadrilateral region is a rectangle within the input image, and the output should be
    that rectangular region. The test creates a 100x100 image with a white 50x50 rectangle centered in it and applies
    the perspective warp to extract the 50x50 rectangle. The result is then compared to a 50x50 image of a white rectangle.
    """
    corner_provider = PerspectiveTransformer(use_gui=False)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (74, 74), (255, 255, 255), -1)  # type: ignore
    corners = {
        "upper_left": (25, 25),
        "upper_right": (75, 25),
        "lower_right": (75, 75),
        "lower_left": (25, 75),
    }
    warped_image = corner_provider.quadrilateral_to_rectangle(image, corners)
    expected_image = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(expected_image, (0, 0), (49, 49), (255, 255, 255), -1)  # type: ignore
    assert np.array_equal(warped_image, expected_image)


@pytest.fixture
def delayed_peak_filter_setup() -> Any:
    empty = cv2.imread("resources/IMG_2582.jpg")  # type: ignore
    full = cv2.imread("resources/IMG_2583.jpg")  # type: ignore
    filter = DelayedPeakFilter()
    return empty, full, filter


def test_delayed_peak_filter_peak(delayed_peak_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = delayed_peak_filter_setup
    # act
    filter.filter(empty)
    filter.filter(full)
    actual = filter.filter(empty)
    # assert
    assert np.array_equal(actual, full)


def test_delayed_peak_filter_vally(delayed_peak_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = delayed_peak_filter_setup
    # act
    filter.filter(full)
    filter.filter(empty)
    actual = filter.filter(full)
    # assert
    assert actual is None


def test_delayed_peak_filter_climbing_to_flat(delayed_peak_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = delayed_peak_filter_setup
    # act
    filter.filter(empty)
    filter.filter(full)
    actual = filter.filter(full)
    # assert
    assert actual is None


def test_delayed_peak_filter_descending_to_flat(delayed_peak_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = delayed_peak_filter_setup
    # act
    filter.filter(full)
    filter.filter(empty)
    actual = filter.filter(empty)
    # assert
    assert actual is None
    # pass


@pytest.fixture
def significant_change_filter_setup() -> Any:
    empty = cv2.imread("resources/IMG_2582.jpg")  # type: ignore
    full = cv2.imread("resources/IMG_2583.jpg")  # type: ignore
    filter = SignificantChangeFilter(0.03, 0.03)
    return empty, full, filter


def test_significant_change_filter_climbing(significant_change_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = significant_change_filter_setup
    # act
    filter.filter(empty)  # Apply the filter with the empty image first to update _last_significant_image
    actual = filter.filter(full)
    # assert
    assert np.array_equal(actual, full)


def test_significant_change_filter_descending(significant_change_filter_setup: Any) -> None:
    # arrange
    empty, full, filter = significant_change_filter_setup
    # act
    filter.filter(full)  # Apply the filter with the full image first to update _last_significant_image
    actual = filter.filter(empty)
    # assert
    assert np.array_equal(actual, empty)


def test_significant_change_filter_no_significant_change(significant_change_filter_setup: Any) -> None:
    # arrange
    empty, _, filter = significant_change_filter_setup
    # act
    filter.filter(empty)  # Apply the filter with the empty image first to update _last_significant_image
    actual = filter.filter(empty)
    # assert
    assert actual is None
