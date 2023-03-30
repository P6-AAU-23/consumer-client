from Tests.testFunctions import CheckIfImageIsPassed
import cv2 as cv


def UpdateWhiteboardAct(colouredChanges, virtualWhiteboard):
    virtualWhiteboardMasked = cv.bitwise_or(
        virtualWhiteboard, virtualWhiteboard, Mask=colouredChanges
    )

    updatedWhiteboard = cv.bitwise_and(colouredChanges, virtualWhiteboardMasked)

    return updatedWhiteboard
