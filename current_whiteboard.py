import cv2
from helper import writePathWithUniqueName
from pathlib import Path


class CurrentWhiteboard:
    __latest_whiteboard = None
    __path = None

    def __init__(self, path: Path):
        self.__path = path

    def get_whiteboard(self) -> cv2.Mat:
        return self.__latest_whiteboard

    def set_whiteboard(self, img: cv2.Mat) -> None:
        self.__latest_whiteboard = img

    def save_whiteboard(self, name: str) -> None:
        cv2.imwrite(writePathWithUniqueName(name, self.__path), self.__latest_whiteboard)

    def get_path(self) -> Path:
        return self.__path
