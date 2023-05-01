import time
import cv2
import numpy as np
from ..src.helper import distance
from pipeline.corner_provider import CornerProvider


# Create a CornerProvider object
cp = CornerProvider("test")

# Create a test image
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Measure the time it takes to execute each method
start_time = time.perf_counter()
corners = cp.get_corners()
end_time = time.perf_counter()
print(f"get_corners(): {end_time - start_time:.6f} seconds")

start_time = time.perf_counter()
preview_image = cp.update(img)
end_time = time.perf_counter()
print(f"update(): {end_time - start_time:.6f} seconds")

# Destroy the GUI window
cv2.destroyAllWindows()
