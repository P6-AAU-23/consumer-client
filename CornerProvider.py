from typing import Dict, Tuple
import pykka
import cv2
import numpy as np

from helper import distance

class CornerProvider(
        # pykka.ThreadingActor
        ):
    CORNER_POINT_SIZE = 10
    FRAME_COLOR = (0, 255, 0)
    FRAME_THICKNESS = 1
    UNINITIALIZED_CORNER = (-1, -1)

    def __init__(self, gui_window_name: str):
        # super().__init__()
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

    def get_corners(self):
        return np.array(list(self.corners.values()))

    def update(self, image):
        if not self._corners_are_on_image(image):
            self._initialize_corners(image)
        preview_image = image.copy()
        self._draw_preview(preview_image)
        cv2.imshow(self.gui_window_name, preview_image) # type: ignore

    def stop(self):
        cv2.destroyAllWindows() # type: ignore
        # super().top()

    def _initialize_corners(self, image):
            height, width, _ = image.shape
            self.corners['upper_left'] = (0, 0)
            self.corners['upper_right'] = (width - 1, 0)
            self.corners['lower_right'] = (width - 1, height - 1)
            self.corners['lower_left'] = (0, height - 1)

    def _corners_are_on_image(self, image):
        height, width, _ = image.shape
        for (x, y) in self.corners.values():
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        return True

    def _draw_preview(self, preview_image):
        for point in self.corners.values():
            self._draw_corner(preview_image, point)
        corner_points = list(self.corners.values())
        for i in range(len(corner_points)):
            self._draw_line(preview_image, corner_points[i], corner_points[(i + 1) % len(corner_points)])

    def _draw_corner(self, preview_image, corner):
        cv2.circle( # type: ignore
                preview_image, 
                corner, 
                CornerProvider.CORNER_POINT_SIZE, 
                CornerProvider.FRAME_COLOR, 
                -1 # -1 fills the circle
            ) 

    def _draw_line(self, preview_image, corner1, corner2):
        cv2.line( # type: ignore
                 preview_image,
                 corner1,
                 corner2,
                 CornerProvider.FRAME_COLOR,
                 CornerProvider.FRAME_THICKNESS
            )

    def _move_corner(self, event, x ,y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and self._move_this != 'None': # type: ignore
            self.corners[self._move_this] = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONDOWN: # type: ignore
            self._move_this = self._on_corner(x, y)
        elif event == cv2.EVENT_LBUTTONUP: # type: ignore
            self._move_this = 'None'

    def _on_corner(self, x, y):
        for corner, point in self.corners.items():
            if distance((x,y), point) <= CornerProvider.CORNER_POINT_SIZE:
                return corner
        return 'None'

