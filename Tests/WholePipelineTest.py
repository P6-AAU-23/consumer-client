import cv2 as cv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from MaskingPipeline.CaptureActivity import CaptureActivity
from Tests.testFunctions import GetPath

def WholePipelineTest():
    fullPath = GetPath() / 'whiteboard1.png'
    image = cv.imread(str(fullPath))

    CA = CaptureActivity(image)
    CA.CaptureActivity(image)

if __name__ == '__main__':   
    WholePipelineTest()