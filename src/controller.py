import cv2
import threading
from typing import Any
from pathlib import Path
from .pipeline.pipeline import Pipeline
from .current_whiteboard import CurrentWhiteboard
from .bufferless_video_capture import BufferlessVideoCapture


class Controller:
    def __init__(self, args: Any):
        self.args = args
        self.cap = BufferlessVideoCapture(args.video_capture_address)
        self.latest_whiteboard = CurrentWhiteboard(Path(args.saved_path))
        self.pipeline = Pipeline(self.latest_whiteboard)

    def run(self) -> None:
        if not self.cap.is_opened():
            print("Can't open camera")
            exit()
        while True:
            self.whiteboard_updated.clear()
            ret, image = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            self.latest_whiteboard.set_whiteboard(self.pipeline.process(image))

            cv2.imshow("preview", whiteboard)  # type: ignore

            if cv2.waitKey(1) == ord("q"):  # type: ignore
                break
            if cv2.waitKey(1) == ord("p"):  # type: ignore
                self.latest_whiteboard.save_whiteboard("whiteboard")

            is_cornerview_closed = cv2.getWindowProperty("Corner Selection Preview", cv2.WND_PROP_VISIBLE) < 1
            is_preview_closed = cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1
            if is_cornerview_closed or is_preview_closed:
                break
        self.latest_whiteboard.save_whiteboard("closing_whiteboard")
        cv2.destroyAllWindows()  # type: ignore

