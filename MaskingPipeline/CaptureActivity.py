import sys
sys.path.append('../../server')
import cv2 as cv
import os

from Pipeline import Binarization
from Pipeline import Changes
from Pipeline import Colourization
from Pipeline import Segmentation
from Pipeline import UpdateWhiteboard

def CaptureActivity(img):
    segImg = Segmentation.SegmentAct(img)
    binImg = Binarization.BinarizeAct(img)
    TrackedChangesMask = Changes.ChangesAct(binImg, segImg)
    ColouredChanges = Colourization.ColouringAct(TrackedChangesMask)
    
    UpdateWhiteboard.UpdateWhiteboardAct(ColouredChanges)

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')+'/')
relativePath = 'Tests\Images\whiteboard1.png'
fullPath = os.path.join(ROOT_DIR, relativePath)
print(fullPath)

image = cv.imread(fullPath)
CaptureActivity(image)