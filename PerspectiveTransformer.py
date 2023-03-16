import numpy as np
import cv2
import math

class PerspectiveTransformer:
    CORNER_POINT_SIZE = 10
    FRAME_COLOR = (0, 255, 0)
    FRAME_THICKNESS = 1

    def __init__(self):
        self.left_upper_corner = None
        self.right_upper_corner = None
        self.right_lower_corner = None
        self.left_lower_corner = None
        self.move_this = None
        cv2.namedWindow('Corner Selection Preview') # type: ignore
        cv2.setMouseCallback('Corner Selection Preview', self._move_corner) # type: ignore

    def transform_perspective(self, image):
        preview_image = image.copy()
        self._set_corners(image)
        self._draw_preview(preview_image)
        cv2.imshow('Corner Selection Preview', preview_image) # type: ignore
        width_upper = distance(self.left_upper_corner, self.right_upper_corner)
        width_lower = distance(self.left_lower_corner, self.right_upper_corner)
        max_width = max(width_upper, width_lower)
        height_left = distance(self.left_upper_corner, self.right_lower_corner)
        height_right = distance(self.right_upper_corner, self.right_lower_corner)
        max_height = max(height_left, height_right)
        target_corners = np.array([
                (0, 0),
                (max_width - 1, 0),
                (max_width - 1, max_height - 1),
                (0, max_height - 1)
            ], np.float32)
        source_corners = np.array([
                self.left_upper_corner, 
                self.right_upper_corner, 
                self.right_lower_corner, 
                self.left_lower_corner
            ], np.float32)
        return warpImage(image, source_corners, target_corners, max_height, max_width)

    def _set_corners(self, image):
        any_corner_is_none = \
            self.left_upper_corner == None \
            or self.right_upper_corner == None \
            or self.right_lower_corner == None \
            or self.left_lower_corner == None
        if any_corner_is_none:
            height, width, _ = image.shape
            self.left_upper_corner = (0, 0)
            self.right_upper_corner = (width - 1, 0)
            self.right_lower_corner = (width - 1, height - 1)
            self.left_lower_corner = (0, height - 1)

    def _move_corner(self, event, x ,y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and self.move_this == 'LEFT_UPPER': # type: ignore
            self.left_upper_corner = (int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE and self.move_this == 'RIGHT_UPPER': # type: ignore
            self.right_upper_corner = (int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE and self.move_this == 'RIGHT_LOWER': # type: ignore
            self.right_lower_corner = (int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE and self.move_this == 'LEFT_LOWER': # type: ignore
            self.left_lower_corner = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONDOWN and self._on_corner(x, y) == self.left_upper_corner: # type: ignore
            self.move_this = 'LEFT_UPPER'
        elif event == cv2.EVENT_LBUTTONDOWN and self._on_corner(x, y) == self.right_upper_corner: # type: ignore
            self.move_this = 'RIGHT_UPPER'
        elif event == cv2.EVENT_LBUTTONDOWN and self._on_corner(x, y) == self.right_lower_corner: # type: ignore
            self.move_this = 'RIGHT_LOWER'
        elif event == cv2.EVENT_LBUTTONDOWN and self._on_corner(x, y) == self.left_lower_corner: # type: ignore
            self.move_this = 'LEFT_LOWER'
        elif event == cv2.EVENT_LBUTTONUP: # type: ignore
            self.move_this = None

    def _on_corner(self, x, y):
        if distance((x, y), self.left_upper_corner) <= PerspectiveTransformer.CORNER_POINT_SIZE:
            return self.left_upper_corner
        elif distance((x, y), self.right_upper_corner) <= PerspectiveTransformer.CORNER_POINT_SIZE:
            return self.right_upper_corner
        elif distance((x, y), self.right_lower_corner) <= PerspectiveTransformer.CORNER_POINT_SIZE:
            return self.right_lower_corner
        elif distance((x, y), self.left_lower_corner) <= PerspectiveTransformer.CORNER_POINT_SIZE:
            return self.left_lower_corner

    def _draw_preview(self, preview_image):
        self._draw_corner(preview_image, self.left_upper_corner)
        self._draw_corner(preview_image, self.right_upper_corner)
        self._draw_corner(preview_image, self.right_lower_corner)
        self._draw_corner(preview_image, self.left_lower_corner)
        self._draw_line(preview_image, self.left_upper_corner, self.right_upper_corner)
        self._draw_line(preview_image, self.right_upper_corner, self.right_lower_corner)
        self._draw_line(preview_image, self.right_lower_corner, self.left_lower_corner)
        self._draw_line(preview_image, self.left_lower_corner, self.left_upper_corner)

    def _draw_corner(self, preview_image, corner):
        cv2.circle( # type: ignore
                preview_image, 
                corner, 
                PerspectiveTransformer.CORNER_POINT_SIZE, 
                PerspectiveTransformer.FRAME_COLOR, 
                -1) # -1 fills the circle

    def _draw_line(self, preview_image, corner1, corner2):
        cv2.line( # type: ignore
                 preview_image,
                 corner1,
                 corner2,
                 PerspectiveTransformer.FRAME_COLOR,
                 PerspectiveTransformer.FRAME_THICKNESS)

def distance(p ,q):
        return math.sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2))

# straigh up just copied from here: 
# https://stackoverflow.com/questions/2992264/extracting-a-quadrilateral-image-to-a-rectangle
def warpImage(image, corners, target, height, width):
    H = cv2.getPerspectiveTransform(corners, target) # type: ignore
    out = cv2.warpPerspective(image, H, (int(height), int(width))) # type: ignore
    return out
