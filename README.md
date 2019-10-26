# Raspberry Pi Urban Mobility Tracker
The Raspberry Pi Urban Mobility Tracker is the simplest way to track and count pedestrians, cyclists, scooters, and vehicles. For more information, see the original blog post [<a target="_blank" href="https://nathanrooy.github.io/posts/2019-02-06/raspberry-pi-deep-learning-traffic-tracker/">here</a>]. Be warned, the code currently works but will require a substantial amount of work before it is mission ready. See the todo list for more information.

## Install

First install the required dependencies for cv2
```
sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-100
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
```
Now, install cv2
```
sudo apt-get install python-opencv
```
Next, install the required python libraries
```
pip install numpy
pip install imutils
pip install matplotlib
pip install scipy
pip install Pillow
```


## Usage
To run while using the Raspberry Pi camera data source run the following:
``` 
python main.py --camera
```
Run the tracker using a video file as input:
```
python main.py --video
```


## Todo
- [ ] Get Coral usb working
- [ ] Transfer learn new model
- [ ] Implement an efficient mobile version of Deep SORT [<a target="_blank" href="https://arxiv.org/abs/1703.07402">arxiv</a>]
- [ ] Finish designing mounting hardware
- [ ] Create example data processing notebooks

