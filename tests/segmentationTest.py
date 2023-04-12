import cv2 as cv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
from MaskingPipeline.Pipeline.Segmentation import Segmentator  # noqa: E402
from Tests.testFunctions import GetPath  # noqa: E402


def main():
    arg = sys.argv[1:]
    segmentationTest(arg[0])


def segmentationTest(imgName):
    path = GetPath()
    imgName += ".jpg"
    fullPath = path / imgName
    img = cv.imread(str(fullPath))

    Seg = Segmentator()
    im = Seg.SegmentAct(img)

    cv.imshow("SegmentationTest", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
