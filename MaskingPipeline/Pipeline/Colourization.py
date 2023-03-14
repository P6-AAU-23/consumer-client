from Tests.testFunctions import CheckIfImageIsPassed
import cv2 as cv

def ColouringAct(Mask, OrigImg):
    CheckIfImageIsPassed(Mask, 'Mask in ColouringAct')
    CheckIfImageIsPassed(OrigImg, 'OrigImg in ColouringAct')

    colouredChanges = cv.bitwise_and(Mask, OrigImg)

    return colouredChanges