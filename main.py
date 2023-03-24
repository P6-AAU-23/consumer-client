import cv2
import argparse

from Pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_capture_address")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video_capture_address)  # type: ignore
    pipeline = Pipeline()
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    while True:
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # type: ignore
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        whiteboard = pipeline.process(image)
        cv2.imshow("preview", whiteboard)  # type: ignore
        if cv2.waitKey(1) == ord("q"):  # type: ignore
            break
    cap.release()
    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
