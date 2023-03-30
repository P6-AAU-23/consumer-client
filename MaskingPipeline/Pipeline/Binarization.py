from Tests import testFunctions
import cv2 as cv
import time


# TODO: remove this
def BinarizeAct(img):
    timeStamp = time.time()

    greyScaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    binarizedImg = cv.adaptiveThreshold(
        greyScaled, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4
    )

    binarizedBurredImg = cv.medianBlur(binarizedImg, 3)

    binarizedFinalImg = cv.bitwise_not(binarizedBurredImg)

    print("Binarisation:" + str(time.time() - timeStamp))

    return binarizedFinalImg
