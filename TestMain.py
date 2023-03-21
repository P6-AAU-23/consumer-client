import cv2 as cv
import os
from pathlib import Path
from MaskingPipeline.CaptureActivity import CaptureActivity
from MaskingPipeline.Pipeline.SegmentationRemoval import RemoveSegmentAct

def GetPath():
    ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
    relativePath = Path('Tests/Images')
    return ROOT_DIR / relativePath 




def RemoveSegmentFromBinaryTest():
    path = GetPath()
    segPath = path / 'segTest.png'
    binPath = path / 'binTest.png'

    seg = cv.imread(str(segPath))
    bin = cv.imread(str(binPath))

    segGrey = cv.cvtColor(seg, cv.COLOR_BGR2GRAY)
    binGrey = cv.cvtColor(bin, cv.COLOR_BGR2GRAY)

    segBin = cv.adaptiveThreshold(segGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1501, 4)
    binBin = cv.adaptiveThreshold(binGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)

    segbin = RemoveSegmentAct(binBin, segBin)

    cv.imshow('test', segbin)
    cv.waitKey(0)

def WholePipelineTest():
    fullPath = GetPath() / 'whiteboard1.png'
    image = cv.imread(str(fullPath))

    CA = CaptureActivity(image)
    CA.CaptureActivity(image)

WholePipelineTest()
RemoveSegmentFromBinaryTest()