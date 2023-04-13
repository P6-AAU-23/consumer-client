import cv2
from helper import uniquifyFileName
from pathlib import Path

class CurrentWhiteboard:
    
    __latestWhiteboard = None

    def getWhiteboard(self):
        #if self.__latestWhiteboard != None:
        return self.__latestWhiteboard

    def setWhiteboard(self, img: cv2.Mat):
        self.__latestWhiteboard = img

    def saveWhiteboard(self, path: Path, name: str):
        pathWithName = path / name

        finalPath = uniquifyFileName(pathWithName)

        cv2.imwrite(str(finalPath), self.__latestWhiteboard)
    