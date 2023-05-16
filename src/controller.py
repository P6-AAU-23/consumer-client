import cv2
from typing import Any
from pathlib import Path
from .helper import try_int_to_string, AvgBgr
from .pipeline.pipeline import pipeline_builder
from .current_whiteboard import CurrentWhiteboard
from .bufferless_video_capture import BufferlessVideoCapture


class Controller:
    def __init__(self, args: Any):
        self.args = args
        self.cap = BufferlessVideoCapture(try_int_to_string(args.video_capture_address))
        ret, frame = self.cap.read()
        if not ret:
            print("Can't open camera")
            exit()
        avg_b = cv2.mean(frame[:, :, 0])[0]
        avg_g = cv2.mean(frame[:, :, 1])[0]
        avg_r = cv2.mean(frame[:, :, 2])[0]
        avg_bgr = AvgBgr(avg_b, avg_g, avg_r)
        self.latest_whiteboard = CurrentWhiteboard(Path(args.saved_path))
        self.pipeline = pipeline_builder(args, avg_bgr)

    def run(self) -> None:
        while True:
            ret, image = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.latest_whiteboard.set_whiteboard(
                self.pipeline.process({"whiteboard": image})["whiteboard"]
            )

            cv2.imshow("preview", self.latest_whiteboard.get_whiteboard())  # type: ignore

            pressed_key = cv2.waitKey(1)

            if pressed_key == ord("q"):  # type: ignore
                break
            if pressed_key == ord("p"):  # type: ignore
                self.latest_whiteboard.save_whiteboard("whiteboard")

            is_cornerview_closed = (
                cv2.getWindowProperty("Corner Selection Preview", cv2.WND_PROP_VISIBLE) < 1
            )
            is_preview_closed = (
                cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1
            )
            if is_cornerview_closed or is_preview_closed:
                break

        self.latest_whiteboard.save_whiteboard("closing_whiteboard")
        cv2.destroyAllWindows()  # type: ignore
