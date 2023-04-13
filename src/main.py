import cv2
import argparse
import os
from pathlib import Path
from helper import uniquifyFileName
from BufferlessVideoCapture import BufferlessVideoCapture
from pipeline.Pipeline import Pipeline


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(PROJECT_ROOT)
    relative_path = Path("Tests/Results/")
    image_path = PROJECT_ROOT / relative_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--saved_path", nargs="?", default=image_path)
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
            path = uniquifyFileName(path)
            cv2.imwrite(str(path), whiteboard)
            break

    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # cap.release()

    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
