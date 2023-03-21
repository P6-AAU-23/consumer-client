import cv2 as cv
from MaskingPipeline.CaptureActivity import CaptureActivity
from Tests.testFunctions import GetPath

def WholePipelineTest():
    fullPath = GetPath() / 'whiteboard1.png'
    image = cv.imread(str(fullPath))

    CA = CaptureActivity(image)
    CA.CaptureActivity(image)