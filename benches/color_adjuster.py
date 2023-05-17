
import os
import cv2
import timeit

SCALE = 1000

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    for i in range(3):
        sum_time = timeit.timeit(
            """
color_adjuster.process({"whiteboard": image})
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (
    ColorAdjuster,
    )
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
avg_bgr = AvgBgr(125, 125, 133)
color_adjuster = ColorAdjuster(avg_bgr, 1.5, 50)
                """,
            number=SCALE,
        )

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"{project_root}/resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"----------------------------------{height}p----------------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")
        print(
            "-----------------------------------------------------------------------------"
        )
