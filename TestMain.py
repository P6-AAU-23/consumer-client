import cv2 as cv
import os
from pathlib import Path
from MaskingPipeline.CaptureActivity import CaptureActivity

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
relativePath = Path('Tests/Images')
fullPath = ROOT_DIR / relativePath / 'whiteboard1.png'

print(fullPath)

image = cv.imread(str(fullPath))

CA = CaptureActivity(image)
CA.CaptureActivity(image)