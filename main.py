import cv2
import argparse

from Pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_capture_address') 
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video_capture_address) # type: ignore
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    Pipeline.start(args, cap)

if __name__ == "__main__":
    main()
