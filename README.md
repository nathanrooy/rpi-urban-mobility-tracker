# Raspberry Pi Urban Mobility Tracker
The Raspberry Pi Urban Mobility Tracker is the simplest way to track and count pedestrians, cyclists, scooters, and vehicles. For more information, see the original blog post [<a target="_blank" href="https://nathanrooy.github.io/posts/2019-02-06/raspberry-pi-deep-learning-traffic-tracker/">here</a>]. Be warned, the code currently works but will require a substantial amount of work before it is mission ready. See the todo list for more information.

## Requirements
1) Raspberry Pi (<a target="_blank" href="https://www.raspberrypi.org/products/raspberry-pi-4-model-b"/>ideally v4-b</a>)
2) Raspberry Pi camera (<a target="_blank" href="https://www.raspberrypi.org/products/camera-module-v2/">ideally v2</a>)
3) Google Coral Accelerator (<a target="_blank" href="https://coral.ai/products/accelerator">Not required, but strongly encouraged</a>)

## Install

First install the required dependencies for cv2
```sh
sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-100
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
```
Next, create and initialize a virtual environment using `virtualenv` and `python 3.7`
```sh
sudo apt-get install virtualenv
virtualenv -p python3.7 venv
source venv/bin/activate
```

Now, install cv2
```sh
sudo apt-get install python-opencv
```
Finally, install the required python libraries
```sh
pip install filterpy
pip install imutils
pip install matplotlib
pip install numpy
pip install Pillow
pip install scipy
pip install scikit-image
```
To run models using TensorFlow Lite, you'll need to install the interpreter which can be found [<a target="_blank" href="https://www.tensorflow.org/lite/guide/python">here</a>].

## Usage
To run while using the Raspberry Pi camera data source run the following:
``` sh
python main.py -camera
```
To run the tracker on an image sequence, append the `-imageseq` flag followed by a path to the images. Included in this repo are the first 300 frames from the MOT (<a target="_blank" href="https://motchallenge.net/">Multiple Object Tracking Benchmark</a>) Challenge <a target="_blank" href="https://motchallenge.net/vis/PETS09-S2L1">PETS09-S2L1</a> video.
```sh
python main.py -imageseq data/images/PETS09-S2L1/
```
To view the bounding boxes and tracking ability of the system, append the `-display` flag to output images. Note that this will greatly slow down the fps and is only recommended for testing purposes.
```sh
python main.py -imageseq data/images/PETS09-S2L1/ -display
```
By default, only the first 10 frames will be processed. To increase or decrease this value, append the `-nframes` flag followed by an integer value.
```sh
python main.py -imageseq data/images/PETS09-S2L1/ -display -nframes 20
```
To run the tracker using a video file input, append the `-video` flag followed by a path to the video file. Included in this repo are two video clips of vehicle traffic.
```sh
python main.py -video data/videos/highway_01.mp4
```
In certain instances, you may want to override the default object detection threshold (default=0.5). To accompish this, append the `-threshold` flag followed by a float value in the range of [0,1]. A value closer to one will yield fewer detections with higher certainty while a value closer to zero will result in more detections with lower certainty. It's usually better to error on the side of lower certainty since these objects can always be filtered out during post processing.
```sh
python main.py -video data/videos/highway_01.mp4 -display -nframes 100 -threshold 0.4
```
To get the highest fps possible, append the `-tpu` flag to use the Coral USB Accelerator for inferencing.
```sh
python main.py -imageseq data/images/PETS09-S2L1/ -tpu
```

## Todo
- [x] Get Coral usb working
- [ ] Transfer learn new model
- [ ] Implement an efficient mobile version of Deep SORT [arxiv]
- [ ] Finish designing mounting hardware
- [ ] Create example data processing notebooks
