import cv2
import numpy as np
from pathlib import Path
from .helper import write_path_with_unique_name


class CurrentWhiteboard:
    __latest_whiteboard = None
    __path = None

    def __init__(self, path: Path):
        self.__path = path

    def get_whiteboard(self) -> np.ndarray:
        return self.__latest_whiteboard

    def set_whiteboard(self, img: np.ndarray) -> None:
        self.__latest_whiteboard = img

    def save_whiteboard(self, name: str) -> None:
        cv2.imwrite(
            write_path_with_unique_name(name, self.__path), self.__latest_whiteboard
        )

    def get_path(self) -> Path:
        return self.__path
