import cv2
import argparse
import os
import threading
from current_whiteboard import CurrentWhiteboard
from pathlib import Path
from src.helper import uniquify_file_name
from src.bufferless_video_capture import BufferlessVideoCapture
from src.pipeline.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--saved_path", nargs="?", default=os.getcwd())
    args = parser.parse_args()
    cap = BufferlessVideoCapture(args.video_capture_address)  # type: ignore
    latest_whiteboard = CurrentWhiteboard(Path(args.save_path))
    whiteboard_updated = threading.Event()
    pipeline = Pipeline(latest_whiteboard, whiteboard_updated)
    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # if not cap.isOpened():
    #     print("Can't open camera")
    #     exit()
    while True:
        whiteboard_updated.clear()
        image = cap.read()

        # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        latest_whiteboard.set_whiteboard(pipeline.process(image))
        whiteboard_updated.set()

        cv2.imshow("preview", latest_whiteboard.get_whiteboard())  # type: ignore

        if cv2.waitKey(1) == ord("q"):  # type: ignore
            break
        is_cornerview_closed = cv2.getWindowProperty("Corner Selection Preview", cv2.WND_PROP_VISIBLE) < 1
        is_preview_closed = cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1
        if is_cornerview_closed or is_preview_closed:
            break

    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # cap.release()
    latest_whiteboard.save_whiteboard("closing_whiteboard")
    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
