import numpy as np
import pytest
from .pipeline import ChangeSuppressor


def test_init_invalid_sensitivity_too_low() -> None:
    with pytest.raises(AssertionError):
        ChangeSuppressor(-0.1)


def test_init_invalid_sensitivity_too_high() -> None:
    with pytest.raises(AssertionError):
        ChangeSuppressor(1.1)


def test_change_over_threshold_same_dimension() -> None:
    # Arrange
    input = np.ones((10, 10, 3), dtype=np.uint8) * 255
    input[3:7, 3:7] = 0
    cs = ChangeSuppressor(0.01)
    # Act
    actual = cs.suppress(input)
    # Assert
    expected = input
    np.array_equal(actual, expected)


def test_change_under_threshold_same_dimension() -> None:
    # Arrange
    input = np.ones((10, 10, 3), dtype=np.uint8) * 255
    input[3:7, 3:7] = 0
    cs = ChangeSuppressor(0.9)
    # Act
    actual = cs.suppress(input)
    # Assert
    expected = cs._last_significant_image
    np.array_equal(actual, expected)
