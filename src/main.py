import os
import cv2
from .helper import list_ports
from gooey import Gooey, GooeyParser
from src.controller import Controller


def main() -> None:
    port_list = list_ports()
    args = parse_args(port_list)
    controller = Controller(args)
    controller.run()


@Gooey
def parse_args(cam_ports):
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
    
    parser.add_argument("saved_path", nargs="?", 
                        widget="DirChooser", 
                        default=os.getcwd(), 
                        help="Choose the folder where you want to save the whiteboards"
                        )
    
    return parser.parse_args()
