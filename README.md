# Raspberry Pi Urban Mobility Tracker
The Raspberry Pi Urban Mobility Tracker is the simplest way to track and count pedestrians, cyclists, scooters, and vehicles. For more information, see the original blog post [<a target="_blank" href="https://nathanrooy.github.io/posts/2019-02-06/raspberry-pi-deep-learning-traffic-tracker/">here</a>]. Be warned, the code currently works but will require a substantial amount of work before it is mission ready. See the todo list for more information.

## Requirements
1) Raspberry Pi (<a target="_blank" href="https://www.raspberrypi.org/products/raspberry-pi-4-model-b"/>ideally v4-b</a>)
2) Raspberry Pi camera (<a target="_blank" href="https://www.raspberrypi.org/products/camera-module-v2/">ideally v2</a>)
3) Google Coral Accelerator (Not required, but strongly encouraged)

## Install

First install the required dependencies for cv2
```
sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-100
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
```
Next, create a virtual environment using `virtualenv` and `python 3.7`
```
sudo apt-get install virtualenv
virtualenv -p python3.7 venv
```

Now, install cv2
```
sudo apt-get install python-opencv
```
Finally, install the required python libraries
```
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
``` 
python main.py -camera
```
To run the tracker on an image sequence, append the `-imageseq` flag followed by a path to the images. Included in this repo are the first 300 frames from the MOT (<a target="_blank" href="https://motchallenge.net/">Multiple Object Tracking Benchmark</a>) Challenge <a target="_blank" href="https://motchallenge.net/vis/PETS09-S2L1">PETS09-S2L1</a> video.
```
python main.py -imageseq data/images/PETS09-S2L1/
```
To view the bounding boxes and tracking ability of the system, append the `-display` flag to output images. Note that this will greatly slow down the fps and is only recommended for testing purposes.
```
python main.py -imageseq data/images/PETS09-S2L1/ -display
```
By default, only the first 10 frames will be processed. To increase or decrease this value, append the `-nframes` flag followed by an integer value.
```
python main.py -imageseq data/images/PETS09-S2L1/ -display -nframes 20
```


## Todo
- [ ] Get Coral usb working
- [ ] Transfer learn new model
- [ ] Implement an efficient mobile version of Deep SORT [arxiv]
- [ ] Finish designing mounting hardware
- [ ] Create example data processing notebooks
