import sys
import cv2 as cv
import numpy as np
import time

from MaskingPipeline.Pipeline import Binarization
from MaskingPipeline.Pipeline import Changes
from MaskingPipeline.Pipeline import Colourization
from MaskingPipeline.Pipeline import Segmentation
from MaskingPipeline.Pipeline import UpdateWhiteboard

class CaptureActivity:
        
    VirtualWhiteboard = None

    def __init__(self, img):
        self.VirtualWhiteboard = np.full(img.shape, 255, dtype=np.uint8)

    def CaptureActivity(self, img):
        timestamp = time.time()

        origImg = img
        
        #if self.VirtualWhiteboard == None:
        #    self.VirtualWhiteboard = np.full(img.shape, 255, dtype=np.uint8)
        greyScaled = cv.cvtColor(origImg, cv.COLOR_BGR2GRAY)
        
        segImg = Segmentation.SegmentAct(img)
        binImg = Binarization.BinarizeAct(greyScaled)
        TrackedChangesMask = Changes.ChangesAct(binImg, segImg)
        ColouredChanges = Colourization.ColouringAct(TrackedChangesMask, origImg)
        self.VirtualWhiteboard = UpdateWhiteboard.UpdateWhiteboardAct(ColouredChanges, self.VirtualWhiteboard)

        print((time.time()-timestamp)*1000)
