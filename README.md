# Whiteboard Processing Pipeline

## Prerequisites
- [Python 3.10+](https://www.python.org/downloads/)
- [pip 22.0+](https://pip.pypa.io/en/stable/installation/)

### Ubuntu (GUI wrapper for CLI)
```
sudo apt-get install -y \
   dpkg-dev 
   build-essential 
   python3-dev 
   freeglut3-dev 
   libgl1-mesa-dev 
   libglu1-mesa-dev
   libgstreamer-plugins-base1.0-dev 
   libgtk-3-dev 
   libjpeg-dev 
   libnotify-dev 
   libpng-dev 
   libsdl2-dev 
   libsm-dev 
   libtiff-dev 
   libwebkit2gtk-4.0-dev 
   libxtst-dev
&& \
   pip install attrdict3
```

## Installation

### CLI only
1. Install prerequisites
2. `git clone https://github.com/P6-AAU-23/server`
3. `cd server && pip install -r requirements.txt`

### GUI wrapper for CLI
1. Install prerequisites
2. `git clone https://github.com/P6-AAU-23/server`
3. `cd server && pip install -r requirements.txt && pip install -r gui_requirements.txt`

## Usage

### GUI wrapper for CLI
1. `python3 server_gui`

### Webcam
1. `python3 server_cli`

### OBS and tiangolo/nginx-rtmp
1. `docker run -d -p 1935:1935 --name nginx-rtmp tiangolo/nginx-rtmp`
2. Open OBS, enter File > Settings > Stream
   1. Set service to Custom...
   2. Set server to `rtmp://localhost/live`
   3. Set stream key to anything you like
3. In OBS Start Streaming
4. `python3 server_cli.py --video_capture_address rtmp://localhost/live/{stream key}` replacing `{stream key}` with the stream key you set in OBS (this should run the pipeline on your stream)

## Configuration

### Streaming to the pipeline
If you want to stream to the pipeline, using OBS or something similar, you will have to set up a RTMP server.
We recommend a [nginx](http://nginx.org/en/) server with the [nginx-rtmp-module](https://github.com/arut/nginx-rtmp-module).
A docker [Docker](https://www.docker.com/) image for this already exists at https://hub.docker.com/r/tiangolo/nginx-rtmp/.
See the [OBS Quick Start](#OBS-Quick-Start) for a concrete guide.

### Full GPU utilization
The pip version of OpenCV is CPU only, therefore you need to remove this version and manually install OpenCV:

#### Windows
1. `pip uninstall opencv-python`
2. [Install OpenCV-Python](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)

#### Ubuntu
1. `pip uninstall opencv-python`
2. `sudo apt-get install python3-opencv`

