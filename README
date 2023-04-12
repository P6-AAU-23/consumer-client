# Whiteboard Processing Pipeline

## Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [pip 22.0+](https://pip.pypa.io/en/stable/installation/)

## Installation

### Installation for webcam
1. Install prerequisites
2. `git clone https://github.com/P6-AAU-23/server`
3. `cd server && pip install -r requirements.txt`
4. `python3 main.py` (this should run the pipeline on your webcam)

### Installation for OBS streaming
1. Install prerequisites
2. Install [Docker](https://www.docker.com/)
3. Install [OBS](https://obsproject.com/)
4. `git clone https://github.com/P6-AAU-23/server`
5. `cd server && pip install -r requirements.txt`
6. `docker run -d -p 1935:1935 --net=host --name nginx-rtmp tiangolo/nginx-rtmp`
7. Open OBS, enter File > Settings > Stream
   1. Set service to Custom...
   2. Set server to `rtmp://localhost/live`
   3. Set stream key to anything you like
8. In OBS Start Streaming
9. `python3 main.py --video_capture_address rtmp://localhost/live/{stream key}` replacing `{stream key}` with the stream key you set in OBS (this should run the pipeline on your stream)

## Usage

## Configuration

### Streaming to the pipeline
If you want to stream to the pipeline, using OBS or something similar, you will have to set up a RTMP server.
We recommend a [nginx](http://nginx.org/en/) server with the [nginx-rtmp-module](https://github.com/arut/nginx-rtmp-module).
A docker [Docker](https://www.docker.com/) image for this already exists at https://hub.docker.com/r/tiangolo/nginx-rtmp/.
See the [OBS Quick Start](#OBS-Quick-Start) for a concrete guide.

### Full GPU utilization
The pip version of OpenCV is CPU only, therefore you need to remove this version and manually install OpenCV:

#### Windows
- `pip uninstall opencv-python`
- [Install OpenCV-Python](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)

#### Ubuntu
- `pip uninstall opencv-python`
- `sudo apt-get install python3-opencv`

