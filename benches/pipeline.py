import timeit
import cv2
import os

FPS_GOAL = 30
SCALE = 1000
GREEN = "\033[1;32;48m"
RED = "\033[1;31;48m"
END = "\033[1;37;0m"

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    for i in range(3):
        time = timeit.timeit(
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
        actual_fps = (SCALE) / time
        ratio = actual_fps / FPS_GOAL
        height, _, _ = cv2.imread(f"resources/benchmark{i}.jpg").shape  # type: ignore
        print(
            f"----------------------------------{height}p----------------------------------"
        )
        print(f"Goal FPS was {FPS_GOAL}.")
        print(f"Actual FPS was {actual_fps}.")
        if 1 < ratio:
            print(
                GREEN + f"The Pipeline is {ratio} faster than it needs to be :)" + END
            )
        else:
            print(RED + f"The Pipeline is {ratio} slower than it needs to be :(" + END)
