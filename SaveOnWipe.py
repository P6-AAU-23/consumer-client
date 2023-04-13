import cv2 as cv
import time
from CurrentWhiteboard import CurrentWhiteboard

class ChangeSavor:

    lastFrame = None

    #def __init__(self, initialImg):
    #    self.lastFrame = initialImg

        

    def func(self):
        while True:
            counter = time.time()

            if counter % 30 == 0:
                if self.lastFrame == None:
                    lastFrame = CurrentWhiteboard.getWhiteboard()
                    

                curFrame = CurrentWhiteboard.getWhiteboard()

                diff = cv.absdiff(self.lastFrame, curFrame)
                num = diff.countNonZero()
                print(num)