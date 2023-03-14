import cv2 as cv
import os

from MaskingPipeline.CaptureActivity import CaptureActivity

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))+'\\'
relativePath = 'Tests\Images\whiteboard1.png'
fullPath = os.path.join(ROOT_DIR, relativePath)
image = cv.imread(fullPath)
cv.waitKey(0)

CA = CaptureActivity()
CA.CaptureActivity(image)