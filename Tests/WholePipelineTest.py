import cv2 as cv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
from MaskingPipeline.CaptureActivity import CaptureActivity
from Tests.testFunctions import GetPath


def main():
    arg = sys.argv[1:]
    WholePipelineTest(arg[0])


def WholePipelineTest(imgName):
    path = GetPath()
    imgName += ".jpg"
    fullPath = path / imgName
    img = cv.imread(str(fullPath))

    CA = CaptureActivity(img)
    CA.CaptureActivityAct(img)


if __name__ == "__main__":
    main()
