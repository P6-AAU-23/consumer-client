from Tests.testFunctions import CheckIfImageIsPassed
import cv2 as cv

def ColouringAct(Mask, OrigImg):

    colouredChanges = cv.bitwise_and(Mask, OrigImg)

    return colouredChanges