import cv2
import numpy as np


class Inpainter:
    def __init__(self):
        self._last_image = None

    def inpaint_missing(
        self, image: np.ndarray, missing_mask: np.ndarray
    ) -> np.ndarray:
        """Inpaints the missing regions in the input image using the provided binary mask,
        and the last image given to this function.

        Args:
            image (np.ndarray): A numpy array representing the input image.
            missing_mask (np.ndarray): A numpy array representing the binary mask indicating missing regions
                                       (0 for missing regions, non-zero for existing regions).

        Raises:
            ValueError: If the input image and missing_mask have different height and width.

        Returns:
            np.ndarray: A numpy array representing the inpainted image with missing regions filled.
        """
        # If _last_image is not set or input image is different shape from _last_image
        if self._last_image is None or image.shape != self._last_image.shape:
            # Initialize _last_image
            self._last_image = np.ones(image.shape, dtype=np.uint8) * 255
        if image.shape[:2] != missing_mask.shape[:2]:
            raise ValueError(
                "The input image and missing_mask must have the same height and width."
            )
        # Ensure the mask is a binary mask (0 or 255)
        binary_mask = (missing_mask == 0).astype(np.uint8) * 255
        # Apply the mask to the _last_image using bitwise operations
        masked_last_image = cv2.bitwise_and(  # type: ignore
            self._last_image, self._last_image, mask=binary_mask
        )
        # Invert the binary_mask to apply it to the input image
        inverted_binary_mask = cv2.bitwise_not(binary_mask)  # type: ignore
        masked_input = cv2.bitwise_and(image, image, mask=inverted_binary_mask)  # type: ignore
        # Combine the masked images to create the inpainted result
        inpainted_image = cv2.add(masked_input, masked_last_image)  # type: ignore
        self._last_image = inpainted_image
        return inpainted_image
