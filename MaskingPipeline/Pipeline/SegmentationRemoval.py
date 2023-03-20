import cv2 as cv
import numpy as np

def RemoveSegmentAct(binImg, segImg):
    
    sizeOfSegImg = segImg.shape

    for colPx in range(sizeOfSegImg[0]) :
        for rowPx in range(sizeOfSegImg[1]):
            if segImg[colPx, rowPx] == 0:
                binImg[colPx, rowPx] == 0

    
    cv.imshow('hng', binImg)
    cv.waitKey(0)