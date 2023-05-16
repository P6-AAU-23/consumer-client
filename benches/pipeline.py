import os
import cv2
import timeit

SCALE = 1000

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = start.process({"whiteboard": image})["whiteboard"]
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (
    IdentityProcessor,
    PerspectiveTransformer,
    ForegroundRemover,
    ColorAdjuster,
    ColorIdealizer,
    Inpainter,
    WipeSaver,
    IdealizeColorsMode
    )
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
avg_bgr = AvgBgr(125, 125, 133)
start = IdentityProcessor()
perspective_transformer = PerspectiveTransformer()
foreground_remover_handler = ForegroundRemover()
color_adjuster_handler = ColorAdjuster(avg_bgr, 1.5, 50)
idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)
inpainter_handler = Inpainter()
head = start
head = head.set_next(perspective_transformer)
head = head.set_next(foreground_remover_handler)
head = head.set_next(color_adjuster_handler)
head = head.set_next(idealize_colors_handler)
head = head.set_next(inpainter_handler)
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
