import cv2
from typing import Any
from pathlib import Path
from .pipeline.pipeline import Pipeline, SignificantChangeFilter
from .current_whiteboard import CurrentWhiteboard
from .bufferless_video_capture import BufferlessVideoCapture
from .helper import try_int_to_string


class Controller:
    def __init__(self, args: Any):
        self.args = args
        self.cap = BufferlessVideoCapture(try_int_to_string(args.video_capture_address))
        self.latest_whiteboard = CurrentWhiteboard(Path(args.saved_path))
        self.pipeline = Pipeline()
        self._significant_change_filter = SignificantChangeFilter(0, 0.005)

    def run(self) -> None:
        if not self.cap.is_opened():
            print("Can't open camera")
            exit()
        while True:
            ret, image = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            whiteboard = self.pipeline.process(image)
            self.latest_whiteboard.set_whiteboard(whiteboard)
            whiteboard = self._significant_change_filter.filter(whiteboard)
            if whiteboard is not None:
                self.latest_whiteboard.save_whiteboard

            cv2.imshow("preview", self.latest_whiteboard.get_whiteboard())  # type: ignore

            pressed_key = cv2.waitKey(1)

            if pressed_key == ord("q"):  # type: ignore
                break
            if pressed_key == ord("p"):  # type: ignore
                self.latest_whiteboard.save_whiteboard("whiteboard")

            is_cornerview_closed = cv2.getWindowProperty("Corner Selection Preview", cv2.WND_PROP_VISIBLE) < 1
            is_preview_closed = cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1
            if is_cornerview_closed or is_preview_closed:
                break

        self.latest_whiteboard.save_whiteboard("closing_whiteboard")
        cv2.destroyAllWindows()  # type: ignore
