import cv2
import numpy
from threading import Event
from .helper import write_path_with_date_and_time
from src.current_whiteboard import CurrentWhiteboard


class ChangeSavor:

    def __init__(self, current_whiteboard: CurrentWhiteboard):
        self.current_whiteboard = current_whiteboard
        self.sleep_time = 5
        self.different_rate = 0.004

    def event_func(self, closing_event: Event, whiteboard_updated: Event) -> None:
        whiteboard_updated.wait()
        last_whiteboard = self.current_whiteboard.get_whiteboard()
        while not closing_event.is_set():
            whiteboard_updated.wait()
            current_whiteboard = self.current_whiteboard.get_whiteboard()

            if self.is_different_size(current_whiteboard, last_whiteboard):
                cv2.imwrite(
                    write_path_with_date_and_time("new_corners", self.current_whiteboard.get_path()),
                    last_whiteboard
                )
            else:
                if self.is_removed_rather_than_added(current_whiteboard, last_whiteboard):
                    full_wipe_img = last_whiteboard
                    while self.is_removed_rather_than_added(current_whiteboard, last_whiteboard):
                        last_whiteboard = current_whiteboard
                        whiteboard_updated.wait()
                        current_whiteboard = self.current_whiteboard.get_whiteboard()
                    if self.calculate_difference_rate(current_whiteboard, full_wipe_img) > self.different_rate:
                        cv2.imwrite(
                            write_path_with_date_and_time("snapshot", self.current_whiteboard.get_path()),
                            full_wipe_img
                        )
                    full_wipe_img = None
            last_whiteboard = current_whiteboard


    def is_different_size(self, cur_whiteboard: cv2.Mat, last_whiteboard: cv2.Mat) -> bool:
        cur_width = cur_whiteboard.shape[0]
        cur_height = cur_whiteboard.shape[1]
        last_width = last_whiteboard.shape[0]
        last_height = last_whiteboard.shape[1]

        if cur_width != last_width or cur_height != last_height:
            return True
        else:
            return False


    def calculate_difference_rate(self, whiteboard1: cv2.Mat, whiteboard2: cv2.Mat) -> float:
        diff = cv2.absdiff(whiteboard2, whiteboard1)
        grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thresh_diff = cv2.threshold(grey_diff, 15, 255, cv2.THRESH_BINARY)[1]

        # Calculate the difference between the 2 images
        total_pixels = whiteboard1.shape[0] * whiteboard1.shape[1] * 1.0
        diff_on_pixels = cv2.countNonZero(thresh_diff) * 1.0
        difference_measure = diff_on_pixels / total_pixels
        return difference_measure


    def is_removed_rather_than_added(self, cur_whiteboard: cv2.Mat, last_whiteboard: cv2.Mat) -> bool:
        num_white_last = numpy.sum(last_whiteboard == 255)
        num_white_cur = numpy.sum(cur_whiteboard == 255)
        

        if num_white_cur > num_white_last:
            return True
        else:
            return False
