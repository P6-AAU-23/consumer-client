from Tests import testFunctions
import cv2 as cv

def BinarizeAct(img):

    binarizedImg = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4)

    binarizedBurredImg = cv.medianBlur(binarizedImg, 3)

    binarizedFinalImg = cv.bitwise_not(binarizedBurredImg)

    return binarizedFinalImg