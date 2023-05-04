import timeit
import cv2
import os

SCALE = 1
GREEN = "\033[1;32;48m"
RED = "\033[1;31;48m"
END = "\033[1;37;0m"

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = pipeline.process(image)
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline.pipeline import Pipeline
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
pipeline = Pipeline()
                """,
            number=SCALE,
        )
        
        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"----------------------------------{height}p----------------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")
