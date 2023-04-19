import cv2
import numpy
from helper import writePathWithDateAndTime

class ChangeSavor:

    def __init__(self, currentWhiteboard):
        self.currentWhiteboard = currentWhiteboard
        self.lastWhiteboard = self.currentWhiteboard.getWhiteboard()
        self.fullWipeImg = None
        self.sleep_time = 5
        self.different_rate = 0.035


    def event_func(self, closing_event, whiteboard_updated):
        whiteboard_updated.wait()
        self.lastWhiteboard = self.currentWhiteboard.getWhiteboard()
        while not closing_event.is_set():
            whiteboard_updated.wait()
            currentWhiteboard = self.currentWhiteboard.getWhiteboard()

            if self.is_different_size(currentWhiteboard):
                cv2.imwrite(writePathWithDateAndTime("snapshot", self.currentWhiteboard.getPath()), self.lastWhiteboard)
            else:
                if self.calculate_different_rate(currentWhiteboard) > self.different_rate:
                    if self.is_removed_rather_than_added(currentWhiteboard):
                            self.fullWipeImg = self.lastWhiteboard
                            while self.is_removed_rather_than_added(currentWhiteboard):
                                whiteboard_updated.wait()
                                self.lastWhiteboard = currentWhiteboard
                                currentWhiteboard = self.currentWhiteboard.getWhiteboard()
                            cv2.imwrite(writePathWithDateAndTime("snapshot-", self.currentWhiteboard.getPath()), self.fullWipeImg)
                            self.fullWipeImg = None
            self.lastWhiteboard = currentWhiteboard


    def is_different_size(self, curWhiteboard):
        if curWhiteboard.shape[0] != self.lastWhiteboard.shape[0] or curWhiteboard.shape[1] != self.lastWhiteboard.shape[1]:
            return True
        else:
            return False
         

    def calculate_different_rate(self, currentWhiteboard):
        diff = cv2.absdiff(self.lastWhiteboard, currentWhiteboard)
        grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thresh_diff = cv2.threshold(grey_diff, 15, 255, cv2.THRESH_BINARY)[1]

        # Calculate the difference between the 2 images
        total_pixels = self.lastWhiteboard.shape[0] * self.lastWhiteboard.shape[1] * 1.0
        diff_on_pixels = cv2.countNonZero(thresh_diff) * 1.0
        difference_measure = diff_on_pixels / total_pixels
        return difference_measure

    def is_removed_rather_than_added(self, curWhiteboard):
        lastWhiteboard = self.lastWhiteboard
        num_white_last = numpy.sum(lastWhiteboard == 255)
        num_white_cur = numpy.sum(curWhiteboard == 255)

        if num_white_cur > num_white_last:
            return True
        else:
            return False
