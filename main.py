import cv2
import argparse
import os
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
    pipeline = Pipeline()
    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # if not cap.isOpened():
    #     print("Can't open camera")
    #     exit()
    while True:
        image = cap.read()
        # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        whiteboard = pipeline.process(image)
        cv2.imshow("preview", whiteboard)  # type: ignore
        if cv2.waitKey(1) == ord("q"):  # type: ignore
            path = Path(args.saved_path) / "whiteboard.jpg"
            path = uniquify_file_name(path)
            cv2.imwrite(str(path), whiteboard)
            break

    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # cap.release()

    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
