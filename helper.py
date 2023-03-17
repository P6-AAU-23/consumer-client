import numpy as np
import cv2

def distance(pt1: tuple, pt2: tuple) -> float:
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def quadrilateral_to_rectangle(image, corners):
    width_upper = distance(corners[0], corners[1])
    width_lower = distance(corners[2], corners[3])
    max_width = int(max(width_upper, width_lower))
    height_left = distance(corners[0], corners[3])
    height_right = distance(corners[1], corners[2])
    max_height = int(max(height_left, height_right))
    target_corners = np.array([
            (0, 0),
            (max_width - 1, 0),
            (max_width - 1, max_height - 1),
            (0, max_height - 1)
        ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.float32(corners), target_corners) # type: ignore
    out = cv2.warpPerspective(image, H, (max_width, max_height)) # type: ignore 
    return out
