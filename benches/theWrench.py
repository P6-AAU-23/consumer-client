import timeit
import cv2
import os
import sys


from pathlib import Path
path_root = os.getcwd()
sys.path.append(str(path_root))
from src.pipeline.pipeline import Pipeline
from src.pipeline.pipeline import Pipeline

SCALE = 1

#pipeline = Pipeline()
    # timeit.repeat(stmt='pass', setup='pass', timer=<default timer>, repeat=5, number=1000000, globals=None)

if __name__ == "__main__":
    project_root = os.getcwd()
    for i in range(3):
        sum_time = timeit.timeit(
            """
whiteboard = pipeline.corner_provider.update(image)
corners = pipeline.corner_provider.get_corners()
whiteboard = quadrilateral_to_rectangle(image, corners)
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline.pipeline import Pipeline, quadrilateral_to_rectangle
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
pipeline = Pipeline()
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
foreground_mask = pipeline.foreground_remover.segment(whiteboard)
cv2.imshow("benchmark", foreground_mask)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline.pipeline import Pipeline, quadrilateral_to_rectangle
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
pipeline = Pipeline()
whiteboard = pipeline.corner_provider.update(image)
corners = pipeline.corner_provider.get_corners()
whiteboard = quadrilateral_to_rectangle(image, corners)
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
whiteboard = pl.idealize_colors(whiteboard, pl.IdealizeColorsMode.MASKING)
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline import pipeline as pl
from src.pipeline.pipeline import Pipeline
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
pipeline = Pipeline()
whiteboard = pipeline.corner_provider.update(image)
corners = pipeline.corner_provider.get_corners()
whiteboard = pl.quadrilateral_to_rectangle(image, corners)
foreground_mask = pipeline.foreground_remover.segment(whiteboard)
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
whiteboard = pipeline.inpainter.inpaint_missing(whiteboard, foreground_mask)
cv2.imshow("benchmark", whiteboard)
cv2.waitKey(1)
                """,
            setup=f"""
import sys
sys.path.append("{project_root}")
from src.pipeline import pipeline as pl
from src.pipeline.pipeline import Pipeline
import cv2
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
pipeline = Pipeline()
whiteboard = pipeline.corner_provider.update(image)
corners = pipeline.corner_provider.get_corners()
whiteboard = pl.quadrilateral_to_rectangle(image, corners)
foreground_mask = pipeline.foreground_remover.segment(whiteboard)
whiteboard = pl.idealize_colors(whiteboard, pl.IdealizeColorsMode.MASKING)
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

#To-do make work
def benchmark(name, test, setup, SCALE):
    for i in range(3):
        sum_time = timeit.timeit(test, setup=setup, number=SCALE)
        
        actual_fps = SCALE / sum_time
        average_time = sum_time / SCALE
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"-------------------------{name} {height}p-------------------------"
        )
        print(f"FPS: {actual_fps}.")
        print(f"time: {average_time}.")


# def mockImages():
#     image = []
    
#     for i in range(0,3):
#        image.append(cv2.imread(f"resources/benchmark{i}.jpg"))
#     return image


# def otherBench():
#     for img in mockImages():
#         whiteboard = pipeline.process(img)

#         cv2.imshow("benchmark", whiteboard)
#         cv2.waitKey(1)
#     return whiteboard

# otherBench()

# timeit.repeat(otherBench, setup="pass", repeat=1, number=SCALE)