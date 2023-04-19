import cv2
import argparse
import os
import threading
from current_whiteboard import CurrentWhiteboard
from pathlib import Path
from BufferlessVideoCapture import BufferlessVideoCapture
from Pipeline import Pipeline


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    relative_path = Path("Tests/Results/")
    image_path = PROJECT_ROOT / relative_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--save_path", nargs="?", default=image_path)
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
        latest_whiteboard.setWhiteboard(pipeline.process(image))
        whiteboard_updated.set()

        cv2.imshow("preview", latest_whiteboard.getWhiteboard())  # type: ignore

        is_cornerview_closed = cv2.getWindowProperty("Corner Selection Preview", cv2.WND_PROP_VISIBLE) < 1
        is_preview_closed = cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1
        if cv2.waitKey(1) == ord("q"):  # type: ignore
            break
        if is_cornerview_closed or is_preview_closed:
            break

    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # cap.release()
    latest_whiteboard.saveWhiteboard("closing_whiteboard")
    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
