import cv2
from typing import Any
from pathlib import Path
from .pipeline.pipeline import Pipeline
from .bufferless_video_capture import BufferlessVideoCapture


class Controller:
    def __init__(self, args: Any):
        self.args = args
        self.cap = BufferlessVideoCapture(args.video_capture_address)
        self.pipeline = Pipeline()

    def run(self):
        if not self.cap.is_opened():
            print("Can't open camera")
            exit()
        while True:
            ret, image = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            whiteboard = self.pipeline.process(image)
            cv2.imshow("preview", whiteboard)  # type: ignore
            if cv2.waitKey(1) == ord("q"):  # type: ignore
                path = Path(self.args.saved_path) / "whiteboard.jpg"
                path = uniquify_file_name(path)
                cv2.imwrite(str(path), whiteboard)
                break
        cv2.destroyAllWindows()  # type: ignore

