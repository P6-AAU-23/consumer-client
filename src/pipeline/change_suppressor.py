import numpy as np
from .pipeline import binarize

class ChangeSuppressor :
    def __init__(self, sensitivity: float) -> None:
        assert 0 <= sensitivity and sensitivity <= 1
        self.sensitivity = sensitivity
        # Initialize last_image to a 10x10 white image
        self._last_significant_image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    def suppress(self, image: np.ndarray):
        ratio = self._size(self._last_significant_image) / self._size(image)
        relative_difference = abs(self._fullness(self._last_significant_image) - (ratio * self._fullness(image)))
        threshold = self.sensitivity * self._size(self._last_significant_image)
        if threshold < relative_difference:
            self._last_significant_image = image
        return self._last_significant_image

    @staticmethod
    def _fullness(image: np.ndarray) -> int:
        binary_image = binarize(image)
        return np.count_nonzero(binary_image == 255)

    @staticmethod
    def _size(image: np.ndarray) -> int:
        height, width, _ = image.shape
        return height * width
