import os
import sys
import cv2
import timeit


path_root = os.getcwd()
sys.path.append(str(path_root))

SCALE = 1

if __name__ == "__main__":
    project_root = os.getcwd()
    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = perspective_transformer.process({"whiteboard": image})["whiteboard"]
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (PerspectiveTransformer)
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
perspective_transformer = PerspectiveTransformer()
                """,
            number=SCALE,
        )
        print()
        print()

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------quadrilateral_to_rectangle {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")

    for i in range(3):
        sum_time = timeit.timeit(
            """
foreground_mask = foreground_remover_handler.process(layers)["foreground_mask"]
cv2.imshow("benchmark", foreground_mask)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (PerspectiveTransformer, ForegroundRemover)
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
perspective_transformer = PerspectiveTransformer()
foreground_remover_handler = ForegroundRemover()
layers = perspective_transformer.process({{"whiteboard": image}})
                """,
            number=SCALE,
        )
        print()
        print()

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------segmentation {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")

    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = color_adjuster_handler.process(layers)["whiteboard"]
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (PerspectiveTransformer, ForegroundRemover, ColorAdjuster)
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
avg_bgr = AvgBgr(125, 125, 133)
perspective_transformer = PerspectiveTransformer()
foreground_remover_handler = ForegroundRemover()
color_adjuster_handler = ColorAdjuster(avg_bgr, 1.5, 50)

layers = perspective_transformer.process({{"whiteboard": image}})
layers = foreground_remover_handler.process(layers)
                """,
            number=SCALE,
        )
        print()
        print()

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------color_adjust {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")

    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = idealize_colors_handler.process(layers)["whiteboard"]
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (
    PerspectiveTransformer, ForegroundRemover, ColorAdjuster, ColorIdealizer, IdealizeColorsMode
    )
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
avg_bgr = AvgBgr(125, 125, 133)
perspective_transformer = PerspectiveTransformer()
foreground_remover_handler = ForegroundRemover()
color_adjuster_handler = ColorAdjuster(avg_bgr, 1.5, 50)
idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)

layers = perspective_transformer.process({{"whiteboard": image}})
layers = foreground_remover_handler.process(layers)
layers = color_adjuster_handler.process(layers)
                """,
            number=SCALE,
        )
        print()
        print()

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------idealize_colors {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")

    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = inpainter_handler.process(layers)["whiteboard"]
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (
    PerspectiveTransformer, ForegroundRemover, ColorAdjuster, ColorIdealizer, Inpainter, IdealizeColorsMode
    )
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
avg_bgr = AvgBgr(125, 125, 133)
perspective_transformer = PerspectiveTransformer()
foreground_remover_handler = ForegroundRemover()
color_adjuster_handler = ColorAdjuster(avg_bgr, 1.5, 50)
idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)
inpainter_handler = Inpainter()

layers = perspective_transformer.process({{"whiteboard": image}})
layers = foreground_remover_handler.process(layers)
layers = color_adjuster_handler.process(layers)
layers = idealize_colors_handler.process(layers)
                """,
            number=SCALE,
        )
        print()
        print()

        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------inpaint_missing {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")
