import cv2 as cv
from Tests.testFunctions import CheckIfImageIsPassed


def ChangesAct(binImg, segImg):
    CheckIfImageIsPassed(binImg, "binImg in ChangeAct")
    CheckIfImageIsPassed(segImg, "segImg in ChangeAct")
    TrackedChangesMask = cv.bitwise_and(binImg, segImg)
    return TrackedChangesMask
