import os
from src.main import main
from src.helper import list_ports
from gooey import Gooey, GooeyParser


@Gooey
def parse_args() -> any:
    cam_ports = list_ports()
    parser = GooeyParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--address",
        widget="TextField",
        default="rtmp://127.0.0.1:1935/live/",
        dest="video_capture_address",
        help="Write rtmp address"
    )
    group.add_argument(
        "--ports",
        choices=cam_ports,
        dest="video_capture_address",
        help="The port for your camera or webcam address"
    )

    parser.add_argument(
        "saved_path", nargs="?",
        widget="DirChooser",
        default=os.getcwd(),
        help="Choose the folder where you want to save the whiteboards"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
