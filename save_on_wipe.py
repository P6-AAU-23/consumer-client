import cv2
import numpy
from threading import Event
from current_whiteboard import CurrentWhiteboard
from helper import writePathWithDateAndTime


class ChangeSavor:

    def __init__(self, current_whiteboard: CurrentWhiteboard):
        self.current_whiteboard = current_whiteboard
        self.last_whiteboard = self.current_whiteboard.getWhiteboard()
        self.full_wipe_img = None
        self.sleep_time = 5
        self.different_rate = 0.035

    def event_func(self, closing_event: Event, whiteboard_updated: Event) -> None:
        whiteboard_updated.wait()
        self.last_whiteboard = self.current_whiteboard.getWhiteboard()
        while not closing_event.is_set():
            whiteboard_updated.wait()
            current_whiteboard = self.current_whiteboard.getWhiteboard()

            if self.is_different_size(current_whiteboard):
                cv2.imwrite(writePathWithDateAndTime("snapshot", self.current_whiteboard.getPath()), self.last_whiteboard)
            else:
                if self.calculate_difference_rate(current_whiteboard) > self.different_rate:
                    if self.is_removed_rather_than_added(current_whiteboard):
                        self.full_wipe_img = self.last_whiteboard
                        while self.is_removed_rather_than_added(current_whiteboard):
                            whiteboard_updated.wait()
                            self.last_whiteboard = current_whiteboard
                            current_whiteboard = self.current_whiteboard.getWhiteboard()
                        cv2.imwrite(
                            writePathWithDateAndTime("snapshot-", self.current_whiteboard.getPath()),
                            self.full_wipe_img
                        )
                        self.full_wipe_img = None
            self.last_whiteboard = current_whiteboard

    def is_different_size(self, cur_whiteboard: cv2.Mat) -> bool:
        cur_width = cur_whiteboard.shape[0]
        cur_height = cur_whiteboard.shape[1]
        last_width = self.last_whiteboard.shape[0]
        last_height = self.last_whiteboard.shape[1]

        if cur_width != last_width or cur_height != last_height:
            return True
        else:
            return False

    def calculate_difference_rate(self, current_whiteboard: cv2.Mat) -> float:
        diff = cv2.absdiff(self.last_whiteboard, current_whiteboard)
        grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thresh_diff = cv2.threshold(grey_diff, 15, 255, cv2.THRESH_BINARY)[1]

        # Calculate the difference between the 2 images
        total_pixels = self.last_whiteboard.shape[0] * self.last_whiteboard.shape[1] * 1.0
        diff_on_pixels = cv2.countNonZero(thresh_diff) * 1.0
        difference_measure = diff_on_pixels / total_pixels
        return difference_measure

    def is_removed_rather_than_added(self, cur_whiteboard: cv2.Mat) -> bool:
        last_whiteboard = self.last_whiteboard
        num_white_last = numpy.sum(last_whiteboard == 255)
        num_white_cur = numpy.sum(cur_whiteboard == 255)

        if num_white_cur > num_white_last:
            return True
        else:
            return False
