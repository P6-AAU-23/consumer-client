import cv2
from typing import Any
from pathlib import Path
<<<<<<< HEAD
from .pipeline.pipeline import Pipeline, Avg_bgr
=======
from .pipeline.pipeline import pipeline_builder
>>>>>>> main
from .current_whiteboard import CurrentWhiteboard
from .bufferless_video_capture import BufferlessVideoCapture
from .helper import try_int_to_string, try_float_to_string


class Controller:
    def __init__(self, args: Any):
        self.args = args
        self.cap = BufferlessVideoCapture(try_int_to_string(args.video_capture_address))
        self.sat = try_float_to_string(args.saturation)
        self.bright = try_int_to_string(args.brightness)
        self.latest_whiteboard = CurrentWhiteboard(Path(args.saved_path))
        self.pipeline = pipeline_builder(args)

    def run(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            print("Can't open camera")
            exit()

        # Read and process the first frame to calculate the average B, G, R values
        avg_b = cv2.mean(frame[:,:,0])[0]
        avg_g = cv2.mean(frame[:,:,1])[0]
        avg_r = cv2.mean(frame[:,:,2])[0]
        avg_color = Avg_bgr(avg_b, avg_g, avg_r)

        while True:
            ret, image = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

<<<<<<< HEAD
            self.latest_whiteboard.set_whiteboard(self.pipeline.process(image, avg_color, self.sat, self.bright))
=======
            self.latest_whiteboard.set_whiteboard(
                self.pipeline.process({"whiteboard": image})["whiteboard"]
            )
>>>>>>> main

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
