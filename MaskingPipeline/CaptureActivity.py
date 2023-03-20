import sys
import cv2 as cv
import numpy as np
import time

from MaskingPipeline.Pipeline import Binarization, Changes, Colourization, Segmentation, UpdateWhiteboard, SegmentationRemoval

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
        yes = SegmentationRemoval.RemoveSegmentAct(binImg, segImg)
        TrackedChangesMask = Changes.ChangesAct(binImg, segImg)
        ColouredChanges = Colourization.ColouringAct(TrackedChangesMask, origImg)
        self.VirtualWhiteboard = UpdateWhiteboard.UpdateWhiteboardAct(ColouredChanges, self.VirtualWhiteboard)

        print((time.time()-timestamp)*1000)
