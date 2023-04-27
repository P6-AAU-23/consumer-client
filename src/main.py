import os
import cv2
from gooey import Gooey, GooeyParser
from src.controller import Controller

@Gooey
def main() -> None:
    parser = GooeyParser()
    cam_ports = list_ports()

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
                        default=cam_ports[0],
                        dest="video_capture_address",
                        help="The port for your camera or webcam address"
                        )
    
    parser.add_argument("saved_path", nargs="?", 
                        widget="DirChooser", 
                        default=os.getcwd(), 
                        help="Choose the folder where you want to save the whiteboards"
                        )

    args = parser.parse_args()
    controller = Controller(args)
    print(args)
    controller.run()


def list_ports() -> list[str]:
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    dev_port = 0
    working_ports = []
    available_ports = []
    print()
    print("I have been runnnnnnnned")
    print()

    i = 0
    while i <= 10: # looking through 10 ports
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            i = 0 # if we find a port, the counter is reset to look through 10 more ports
            is_reading, img = camera.read()
            
            if is_reading:
                working_ports.append(str(dev_port))
            else:
                available_ports.append(str(dev_port))

        i += 1
        dev_port +=1
    return working_ports
