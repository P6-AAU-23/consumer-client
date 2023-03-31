import cv2
import argparse
import os
from pathlib import Path

from Tests.testFunctions import GetPath
from Pipeline import Pipeline


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(PROJECT_ROOT)
    relative_path = Path("Tests/Results/")
    image_path = PROJECT_ROOT / relative_path
    parser = argparse.ArgumentParser()
    parser.add_argument("video_capture_address")
    parser.add_argument("final_whiteboard_image_path", nargs="?", default=image_path)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video_capture_address)  # type: ignore

    pipeline = Pipeline()
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    while True:
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # type: ignore
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        whiteboard = pipeline.process(image)
        cv2.imshow("preview", whiteboard)  # type: ignore
        if cv2.waitKey(1) == ord("q"):  # type: ignore
            path = Path(args.final_whiteboard_image_path) / 'whiteboard.jpg'
            print(path)
            cv2.imwrite(str(path), whiteboard)
            break


    cap.release()
    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
