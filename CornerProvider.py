from typing import Dict, Tuple
from helper import distance
import cv2
import numpy as np

class CornerProvider():
    """
    This class provides an interface to manipulate corner points on a given image using OpenCV.
    """
    CORNER_POINT_SIZE = 10
    FRAME_COLOR = (0, 255, 0)
    FRAME_THICKNESS = 1
    UNINITIALIZED_CORNER = (-1, -1)

    def __init__(self, gui_window_name: str):
        """
        Initializes a CornerProvider instance with a named OpenCV window and a mouse callback function.

        :param gui_window_name: The name of the OpenCV window.
        """
        self.corners: Dict[str, Tuple[int, int]] = {
            'upper_left': CornerProvider.UNINITIALIZED_CORNER,
            'upper_right': CornerProvider.UNINITIALIZED_CORNER,
            'lower_right': CornerProvider.UNINITIALIZED_CORNER,
            'lower_left': CornerProvider.UNINITIALIZED_CORNER,
        }
        self._move_this = 'None'
        self.gui_window_name = gui_window_name
        cv2.namedWindow(self.gui_window_name)  # type: ignore
        cv2.setMouseCallback(self.gui_window_name, self._move_corner)  # type: ignore

    def get_corners(self) -> np.ndarray:
        """Returns the corner points as a numpy array."""
        return np.array(list(self.corners.values()))

    def update(self, image: np.ndarray):
        """
        Updates the image displayed in the GUI window, drawing corner points and lines on the image.

        :param image: The image to be updated.
        """
        if not self._corners_are_on_image(image):
            self._initialize_corners(image)
        preview_image = image.copy()
        self._draw_preview(preview_image)
        cv2.imshow(self.gui_window_name, preview_image)  # type: ignore

    def _initialize_corners(self, image: np.ndarray):
        """Initializes corner points based on the image dimensions."""
        height, width, _ = image.shape
        self.corners['upper_left'] = (0, 0)
        self.corners['upper_right'] = (width - 1, 0)
        self.corners['lower_right'] = (width - 1, height - 1)
        self.corners['lower_left'] = (0, height - 1)

    def _corners_are_on_image(self, image: np.ndarray) -> bool:
        """Checks if all corner points are within the image boundaries."""
        height, width, _ = image.shape
        for (x, y) in self.corners.values():
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True

    def _draw_preview(self, preview_image: np.ndarray):
        """Draws corner points and lines on the image."""
        for point in self.corners.values():
            self._draw_corner(preview_image, point)
        corner_points = list(self.corners.values())
        for i in range(len(corner_points)):
            self._draw_line(preview_image, corner_points[i], corner_points[(i + 1) % len(corner_points)])

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
            -1 # -1 fills the circle
        )

    def _draw_line(self, preview_image: np.ndarray, corner1: Tuple[int, int], corner2: Tuple[int, int]):
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
            CornerProvider.FRAME_THICKNESS
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
        if event == cv2.EVENT_MOUSEMOVE and self._move_this != 'None':  # type: ignore
            self.corners[self._move_this] = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONDOWN:  # type: ignore
            self._move_this = self._on_corner(x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # type: ignore
            self._move_this = 'None'

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
        return 'None'

