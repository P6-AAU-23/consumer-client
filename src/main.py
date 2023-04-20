import argparse
import os
from src.controller import Controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--saved_path", nargs="?", default=os.getcwd())
    args = parser.parse_args()
    controller = Controller(args)
    controller.run()


if __name__ == "__main__":
    main()
