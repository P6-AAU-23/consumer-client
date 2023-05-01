import os
import argparse
from src.main import main


def parse_args() -> any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--saved_path", nargs="?", default=os.getcwd())
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
