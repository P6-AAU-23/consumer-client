import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_capture_address') 
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video_capture_address)
    if not cap.isOpened():
        print("Can't open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        whiteboard = idealize(frame)
        cv2.imshow('preview', whiteboard)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# TODO: implement pipeline here
def idealize(frame):
    return frame

if __name__ == "__main__":
    main()
