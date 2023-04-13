from typing import Dict, Tuple, Any
from helper import distance
import cv2
import numpy as np


class CornerProvider:
    """
    This class provides an interface to manipulate corner points on a given image using OpenCV.
    """

    CORNER_POINT_SIZE = 10
    FRAME_COLOR = (0, 255, 0)
    FRAME_THICKNESS = 1
    UNINITIALIZED_CORNER = (-1, -1)

    def __init__(self, gui_window_name: str, use_gui: bool = True):
        """
        Initializes a CornerProvider instance with a named OpenCV window and a mouse callback function.

        :param gui_window_name: The name of the OpenCV window.
        :param use_gui: Whether to use the GUI functionality or not. Defaults to True.
        """
        self.corners: Dict[str, Tuple[int, int]] = {
            "upper_left": CornerProvider.UNINITIALIZED_CORNER,
            "upper_right": CornerProvider.UNINITIALIZED_CORNER,
            "lower_right": CornerProvider.UNINITIALIZED_CORNER,
            "lower_left": CornerProvider.UNINITIALIZED_CORNER,
        }
        self._move_this = "None"
        self.use_gui = use_gui
        if self.use_gui:
            self.gui_window_name = gui_window_name
            cv2.namedWindow(self.gui_window_name)  # type: ignore
            cv2.setMouseCallback(self.gui_window_name, self._move_corner)  # type: ignore

    def get_corners(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns a dictionary containing the coordinates of the corner points.

        :return: A dictionary with keys 'upper_left', 'upper_right', 'lower_right', 'lower_left'
                 and values that are tuples of the x and y coordinates of the corresponding corner point.
        """
        return self.corners

    def update(self, image: np.ndarray) -> np.ndarray:
        """
        Updates the image displayed in the GUI window, drawing corner points and lines on the image.
        Returns the modified image with corner points and lines drawn.

        :param image: The image to be updated.
        :return: The modified image (preview_image) with corner points and lines drawn.
        """
        if not self._corners_are_on_image(image) and self._move_this == "None":
            self._initialize_corners(image)
        preview_image = image.copy()
        self._draw_preview(preview_image)
        if self.use_gui:
            cv2.imshow(self.gui_window_name, preview_image)  # type: ignore
        return preview_image

    def _initialize_corners(self, image: np.ndarray):
        """Initializes corner points based on the image dimensions."""
        height, width, _ = image.shape
        self.corners["upper_left"] = (0, 0)
        self.corners["upper_right"] = (width - 1, 0)
        self.corners["lower_right"] = (width - 1, height - 1)
        self.corners["lower_left"] = (0, height - 1)

    def _corners_are_on_image(self, image: np.ndarray) -> bool:
        """Checks if all corner points are within the image boundaries."""
        height, width, _ = image.shape
        for x, y in self.corners.values():
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True

    def _draw_preview(self, preview_image: np.ndarray):
        """Draws corner points and lines on the image."""
        for point in self.corners.values():
            self._draw_corner(preview_image, point)
        corner_points = list(self.corners.values())
        for i in range(len(corner_points)):
            self._draw_line(
                preview_image,
                corner_points[i],
                corner_points[(i + 1) % len(corner_points)],
            )

    def _draw_corner(self, preview_image: np.ndarray, corner: Tuple[int, int]):
        """
        Draws a corner point on the image.

        :param preview_image: The image to draw the corner point on.
        :param corner: The corner point to draw.
        """
        cv2.circle(  # type: ignore
            preview_image,
            corner,
            CornerProvider.CORNER_POINT_SIZE,
            CornerProvider.FRAME_COLOR,
            -1,  # -1 fills the circle
        )

    def _draw_line(
        self,
        preview_image: np.ndarray,
        corner1: Tuple[int, int],
        corner2: Tuple[int, int],
    ):
        """
        Draws a line between two corner points on the image.

        :param preview_image: The image to draw the line on.
        :param corner1: The first corner point.
        :param corner2: The second corner point.
        """
        cv2.line(  # type: ignore
            preview_image,
            corner1,
            corner2,
            CornerProvider.FRAME_COLOR,
            CornerProvider.FRAME_THICKNESS,
        )

    def _move_corner(self, event: int, x: int, y: int, flags: int, param: Any):
        """
        Handles mouse events for corner points manipulation.

        :param event: The OpenCV mouse event.
        :param x: The x-coordinate of the mouse event.
        :param y: The y-coordinate of the mouse event.
        :param flags: The OpenCV event flags.
        :param param: The user-defined parameter passed by the OpenCV mouse callback.
        """
        if event == cv2.EVENT_MOUSEMOVE and self._move_this != "None":  # type: ignore
            self.corners[self._move_this] = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONDOWN:  # type: ignore
            self._move_this = self._on_corner(x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # type: ignore
            self._move_this = "None"

    def _on_corner(self, x: int, y: int) -> str:
        """
        Checks if the given point is on a corner point and returns the corner name if so.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.
        :return: The corner name if the point is on a corner, 'None' otherwise.
        """
        for corner, point in self.corners.items():
            if distance((x, y), point) <= CornerProvider.CORNER_POINT_SIZE:
                return corner
        return "None"

import pytest
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


# test for bug #21
def test_corner_outside_of_image_not_being_moved_resets_all_corners(cp_initialized):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cp_initialized.corners["upper_left"] = (-1, -1)
    cp_initialized.update(image)
    corners = cp_initialized.get_corners()
    assert corners["upper_left"] == (0, 0)
    assert corners["upper_right"] == (199, 0)
    assert corners["lower_right"] == (199, 99)
    assert corners["lower_left"] == (0, 99)


# test for bug #21
def test_corner_outside_of_image_being_moved_does_not_resets_all_corners(
    cp_initialized,
):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cp_initialized._move_this = "upper_left"
    cp_initialized.corners["upper_left"] = (-1, -1)
    cp_initialized.update(image)
    corners = cp_initialized.get_corners()
    assert corners["upper_left"] == (-1, -1)
    assert corners["upper_right"] == (199, 0)
    assert corners["lower_right"] == (199, 99)
    assert corners["lower_left"] == (0, 99)
