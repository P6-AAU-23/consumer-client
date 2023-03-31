import cv2
import argparse
from BufferlessVideoCapture import BufferlessVideoCapture

from Pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_capture_address")
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
            break
    # TODO: this should probably implemented for the BufferlessVideoCapture, and uncommented
    # cap.release()
    cv2.destroyAllWindows()  # type: ignore


if __name__ == "__main__":
    main()
