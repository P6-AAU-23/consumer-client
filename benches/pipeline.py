import os
import cv2
import timeit
from pathlib import Path

SCALE = 1
GREEN = "\033[1;32;48m"
RED = "\033[1;31;48m"
END = "\033[1;37;0m"

if __name__ == "__main__":
    project_root = Path(os.getcwd())
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
wipe_saver = WipeSaver("dev_0")
head = start
head = head.set_next(perspective_transformer)
head = head.set_next(foreground_remover_handler)
head = head.set_next(color_adjuster_handler)
head = head.set_next(idealize_colors_handler)
head = head.set_next(inpainter_handler)
head = head.set_next(wipe_saver)
                """,
            number=SCALE,
        )

        sum_time1 = timeit.timeit(
            """
corner_provider.update(image)
corners = corner_provider.get_corners()
whiteboard = pers_trans.quadrilateral_to_rectangle(image, corners)
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline.corner_provider import CornerProvider
from src.pipeline.pipeline_modules import PerspectiveTransformer
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
corner_provider = CornerProvider("test")
pers_trans = PerspectiveTransformer()
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

        step_1_fps = SCALE / sum_time1
        step_1_time = sum_time1 / SCALE
        print(f"mhm testing:{step_1_fps}, {step_1_time}")
