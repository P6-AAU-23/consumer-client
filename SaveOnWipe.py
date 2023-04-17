import cv2
import threading
import time
from datetime import datetime
from helper import writePathWithDateAndTime

class ChangeSavor:

    def __init__(self, currentWhiteboard):
        self.currentWhiteboard = currentWhiteboard
        self.lastWhiteboard = self.currentWhiteboard.getWhiteboard()
        

    def func(self, closing_event) -> None:
            time.sleep(5)
            while not closing_event.is_set():
                curWb = self.currentWhiteboard.getWhiteboard()

                if self.lastWhiteboard is None:
                    self.lastWhiteboard = curWb
                else:    
                    
                    if self.is_different_size(curWb):
                        cv2.imwrite(writePathWithDateAndTime("snapshot", self.currentWhiteboard.getPath()), self.lastWhiteboard)
                        self.lastWhiteboard = curWb

                    else:

                        diff = cv2.absdiff(self.lastWhiteboard, curWb)
                        grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        thresh_diff = cv2.threshold(grey_diff, 15, 255, cv2.THRESH_BINARY)[1]

                        # Calculate the difference between the 2 images
                        total_pixels = self.lastWhiteboard.shape[0] * self.lastWhiteboard.shape[1] * 1.0
                        diff_on_pixels = cv2.countNonZero(thresh_diff) * 1.0
                        difference_measure = diff_on_pixels / total_pixels
                        self.lastWhiteboard = curWb
                        print(difference_measure)

                time.sleep(5)
            
    def is_different_size(self, curWhiteboard):

         if curWhiteboard.shape[0] != self.lastWhiteboard.shape[0] or curWhiteboard.shape[1] != self.lastWhiteboard.shape[1]:
              return True
         else:
              return False