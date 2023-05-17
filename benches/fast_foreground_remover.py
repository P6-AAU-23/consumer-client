import os
import cv2
import timeit
from pathlib import Path

SCALE = 1000

if __name__ == "__main__":
    dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.abspath(dir / "..")
    for i in range(3):
        sum_time = timeit.timeit(
            """
foreground_remover.process({"whiteboard": image})
                """,
            setup=f"""
import sys
import cv2
sys.path.append("{project_root}")
from src.pipeline.pipeline import (
    FastForegroundRemover,
    Inpainter,
    )
from src.helper import AvgBgr
gc.enable()
image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
foreground_remover = FastForegroundRemover()
inpainter = Inpainter()
foreground_remover.set_next(inpainter)
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
