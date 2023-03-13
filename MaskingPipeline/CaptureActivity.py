import sys
sys.path.append('../../server')
import cv2 as cv
import os
import numpy as np

from Pipeline import Binarization
from Pipeline import Changes
from Pipeline import Colourization
from Pipeline import Segmentation

class CaptureActivity:
        
    VirtualWhiteboard = None

    def CaptureActivity(self, img):
        self.VirtualWhiteboard = np.full(img.shape, 255, dtype=np.uint8)
        segImg = Segmentation.SegmentAct(img)
        binImg = Binarization.BinarizeAct(img)
        TrackedChangesMask = Changes.ChangesAct(binImg, segImg)
        ColouredChanges = Colourization.ColouringAct(TrackedChangesMask)
        
        self.__UpdateWhiteboardAct(ColouredChanges)
    
    
    def __UpdateWhiteboardAct(self, colouredChanges):

        self.virtualWhiteboard = colouredChanges
        
    
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')+'/')
relativePath = 'Tests\Images\whiteboard1.png'
fullPath = os.path.join(ROOT_DIR, relativePath)
image = cv.imread(fullPath)

CA = CaptureActivity()
CA.CaptureActivity(image)
