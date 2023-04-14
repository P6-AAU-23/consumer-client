from typing import Dict, Optional, Tuple, Any
from ..helper import distance
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
        self.corners: Dict[str, Tuple[int, int]] = {
            "upper_left": CornerProvider.UNINITIALIZED_CORNER,
            "upper_right": CornerProvider.UNINITIALIZED_CORNER,
            "lower_right": CornerProvider.UNINITIALIZED_CORNER,
            "lower_left": CornerProvider.UNINITIALIZED_CORNER,
        }
        self._move_this: Optional[str] = None
        self.use_gui = use_gui
        if self.use_gui:
            self.gui_window_name = gui_window_name
            cv2.namedWindow(self.gui_window_name)  # type: ignore
            cv2.setMouseCallback(self.gui_window_name, self._move_corner)  # type: ignore

    def get_corners(self) -> Dict[str, Tuple[int, int]]:
        """Returns a dictionary containing the coordinates of the corner points.

        Returns:
            dict: A dictionary with keys 'upper_left', 'upper_right', 'lower_right', 'lower_left'
                  and values that are tuples of the x and y coordinates of the corresponding corner point.
        """
        return self.corners

    def update(self, image: np.ndarray) -> np.ndarray:
        """Updates the image displayed in the GUI window, drawing corner points and lines on the image.

        Args:
            image (ImageType): The image to be updated.

        Returns:
            ImageType: The modified image (preview_image) with corner points and lines drawn.
        """
        if not self._corners_are_on_image(image) and self._move_this is None:
            self._initialize_corners(image)
        preview_image = image.copy()
        self._draw_preview(preview_image)
        if self.use_gui:
            cv2.imshow(self.gui_window_name, preview_image)  # type: ignore
        return preview_image

    def _initialize_corners(self, image: np.ndarray) -> None:
        height, width, _ = image.shape
        self.corners["upper_left"] = (0, 0)
        self.corners["upper_right"] = (width - 1, 0)
        self.corners["lower_right"] = (width - 1, height - 1)
        self.corners["lower_left"] = (0, height - 1)

    def _corners_are_on_image(self, image: np.ndarray) -> bool:
        height, width, _ = image.shape
        for x, y in self.corners.values():
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True

    def _draw_preview(self, preview_image: np.ndarray) -> None:
        for point in self.corners.values():
            self._draw_corner(preview_image, point)
        corner_points = list(self.corners.values())
        for i in range(len(corner_points)):
            self._draw_line(
                preview_image,
                corner_points[i],
                corner_points[(i + 1) % len(corner_points)],
            )

    def _draw_corner(self, preview_image: np.ndarray, corner: Tuple[int, int]) -> None:
        """Draws a corner point on the image.

        Args:
            preview_image (ImageType): The image to draw the corner point on.
            corner (Tuple[int, int]): The corner point to draw.
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
    ) -> None:
        """Draws a line between two corner points on the image.

        Args:
            preview_image (ImageType): The image to draw the line on.
            corner1 (Tuple[int, int]): The first corner point.
            corner2 (Tuple[int, int]): The second corner point.
        """
        cv2.line(  # type: ignore
            preview_image,
            corner1,
            corner2,
            CornerProvider.FRAME_COLOR,
            CornerProvider.FRAME_THICKNESS,
        )

    def _move_corner(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handles mouse events for corner points manipulation.

        Args:
            event (int): The OpenCV mouse event.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): The OpenCV event flags.
            param (Any): The user-defined parameter passed by the OpenCV mouse callback.
        """
        if event == cv2.EVENT_MOUSEMOVE and self._move_this is not None:  # type: ignore
            self.corners[self._move_this] = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONDOWN:  # type: ignore
            self._move_this = self._on_corner(x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # type: ignore
            self._move_this = None

    def _on_corner(self, x: int, y: int) -> Optional[str]:
        """Checks if the given point is on a corner point and returns the corner name if so.

        Args:
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.

        Returns:
            Optional[str]: The corner name if the point is on a corner, None otherwise.
        """
        for corner, point in self.corners.items():
            if distance((x, y), point) <= CornerProvider.CORNER_POINT_SIZE:
                return corner
        return None
