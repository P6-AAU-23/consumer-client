import os
import argparse
from src.main import main


def parse_args() -> any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_capture_address", nargs="?", default=0)
    parser.add_argument("--saved_path", nargs="?", default=os.getcwd())
    parser.add_argument("--saturation", nargs="?", default=1.0)
    parser.add_argument("--brightness", nargs="?", default=0)
    parser.add_argument("--disable_remove_foreground", action="store_true")
    parser.add_argument("--disable_color_adjuster", action="store_true")
    parser.add_argument("--disable_transform_perspective", action="store_true")
    parser.add_argument("--disable_idealize_colors", action="store_true")
    parser.add_argument("--save_on_wipe", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--slow", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
