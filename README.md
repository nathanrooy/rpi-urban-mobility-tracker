# Raspberry Pi Urban Mobility Tracker
The Raspberry Pi Urban Mobility Tracker is the simplest way to track and count pedestrians, cyclists, scooters, and vehicles. For more information, see the original blog post [<a target="_blank" href="https://nathanrooy.github.io/posts/2019-02-06/raspberry-pi-deep-learning-traffic-tracker/">here</a>]. Be warned, the code currently works but will require a substantial amount of work before it is mission ready. See the todo list for more information.

## Install


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

