import cv2
from helper import writePathWithUniqueName
from pathlib import Path

class CurrentWhiteboard:
    
    __latestWhiteboard = None
    __path = None

    def __init__(self, path: Path):
        self.__path = path

    def getWhiteboard(self):
        return self.__latestWhiteboard

    def setWhiteboard(self, img: cv2.Mat):
        self.__latestWhiteboard = img

    def saveWhiteboard(self, name: str):
        cv2.imwrite(writePathWithUniqueName(name, self.__path), self.__latestWhiteboard)

    def getPath(self) -> Path:
        return self.__path
        
    