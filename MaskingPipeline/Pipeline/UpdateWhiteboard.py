from Tests.testFunctions import CheckIfImageIsPassed
import cv2 as cv

def UpdateWhiteboardAct(colouredChanges, virtualWhiteboard):

    CheckIfImageIsPassed(colouredChanges, 'colouredChanges in UpdateWhiteBoardAct')
    CheckIfImageIsPassed(virtualWhiteboard, 'virtualWhiteboard in UpdateWhiteBoardAct')

    updatedWhiteboard = cv.bitwise_and(colouredChanges, virtualWhiteboard)
    
    return colouredChanges