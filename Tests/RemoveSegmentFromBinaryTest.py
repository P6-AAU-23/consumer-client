import cv2 as cv
from MaskingPipeline.Pipeline.SegmentationRemoval import RemoveSegmentAct
from Tests.testFunctions import GetPath

def RemoveSegmentFromBinaryTest():
    path = GetPath()
    segPath = path / 'segTest.png'
    binPath = path / 'binTest.png'

    seg = cv.imread(str(segPath))
    bin = cv.imread(str(binPath))

    segGrey = cv.cvtColor(seg, cv.COLOR_BGR2GRAY)
    binGrey = cv.cvtColor(bin, cv.COLOR_BGR2GRAY)

    segBin = cv.adaptiveThreshold(segGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 1501, 4)
    cv.imshow('In:Segmented', segBin)
    binBin = cv.adaptiveThreshold(binGrey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
    cv.imshow('In:Binarized', binBin)

    segbin = RemoveSegmentAct(binBin, segBin)
    cv.imshow('Out:Segmentation removed', segbin)

    cv.waitKey(0)


