import cv2 as cv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)
from MaskingPipeline.Pipeline.SegmentationRemoval import RemoveSegmentAct
from Tests.testFunctions import GetPath


def RemoveSegmentFromBinaryTest():
    path = GetPath()
    segPath = path / "segTest.jpg"
    binPath = path / "binTest.jpg"

    seg = cv.imread(str(segPath))
    bin = cv.imread(str(binPath))

    segGrey = cv.cvtColor(seg, cv.COLOR_BGR2GRAY)
    binGrey = cv.cvtColor(bin, cv.COLOR_BGR2GRAY)

    segBin = cv.adaptiveThreshold(
        segGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1501, 4
    )
    binBin = cv.adaptiveThreshold(
        binGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4
    )

    im = RemoveSegmentAct(binBin, segBin)
    cv.imshow("SegmentationRemover", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    RemoveSegmentFromBinaryTest()
