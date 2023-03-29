from Tests.testFunctions import CheckIfImageIsPassed
import cv2 as cv

def ColouringAct(Mask, OrigImg):

    colouredChanges = cv.bitwise_and(OrigImg, OrigImg, mask=Mask)

    return colouredChanges