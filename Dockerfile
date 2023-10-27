# set base image
FROM arm64v8/debian:bullseye-slim

# install python 3.7 and create alias
RUN apt-get update -y
RUN apt install -y python3.9
RUN echo 'alias python="/usr/bin/python3.9"' >> ~/.bashrc

# install python dependencies
RUN apt-get install -y python3-dev python3-setuptools python3-pip
# RUN pip3 install --upgrade pip
# RUN pip3 install imutils==0.5.3
#RUN apt-get install -y python3-numpy
#RUN apt-get install -y python3-pillow
#RUN apt-get install -y python3-opencv
#RUN pip3 install -U opencv-python
# install scipy and associated dependencies
RUN apt-get install -y build-essential cmake git unzip pkg-config libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libcanberra-gtk* libgtk-3-dev libgstreamer1.0-dev gstreamer1.0-gtk3 libgstreamer-plugins-base1.0-dev gstreamer1.0-gl libxvidcore-dev libx264-dev python3-dev python3-numpy python3-pip libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev v4l-utils libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev gfortran libhdf5-dev libprotobuf-dev libgoogle-glog-dev libgflags-dev protobuf-compiler
RUN apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev
RUN apt-get install -y gfortran
RUN apt-get install -y libatlas-base-dev
#RUN apt-get install -y python3-scipy

# download tensorflow 2.4 wheel and install
WORKDIR /root
RUN apt-get install -y wget
RUN apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
RUN pip3 install keras_applications==1.0.8 --no-deps
RUN pip3 install keras_preprocessing==1.1.0 --no-deps
#RUN apt-get install -y python3-h5py
RUN apt-get install -y openmpi-bin 
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y libatlas-base-dev
# RUN apt-get install -y python3-libcamera2
# RUN pip3 install six==1.15.0
# RUN pip3 install wheel==0.36.2
# RUN pip3 install mock==4.0.3
#RUN wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
## RUN pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl --no-deps
# RUN pip3 install tensorflow
# RUN pip3 install absl-py==0.11.0
# RUN pip3 install --upgrade google-api-python-client
# RUN pip3 install wrapt==1.12.1
# RUN pip3 install typing-extensions==3.7.4.3
# RUN pip3 install opt_einsum==3.3.0
# RUN pip3 install gast==0.4.0
# RUN pip3 install astunparse==1.6.3
# RUN pip3 install termcolor==1.1.0
# RUN pip3 install flatbuffers==1.12

# install tflite runtime
# # RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
# RUN pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl 

# install some additional tools for debugging
RUN apt-get install -y feh
RUN apt-get install -y vim

# install git
RUN apt-get install -y git

RUN echo "deb http://archive.raspberrypi.org/debian bullseye main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E
RUN apt-get update 
RUN apt-get install -y python3-opencv python3-libcamera python3-kms++ python3-picamera2 libcamera-apps
# RUN apt install -y python3-pyqt5 python3-prctl libatlas-base-dev ffmpeg
# install deep sort
# RUN pip3 install git+https://github.com/mk-michal/deep_sort

ADD requirements.txt .
RUN pip3 install -r requirements.txt
# install umt
RUN pip3 install git+https://github.com/nathanrooy/rpi-urban-mobility-tracker --no-deps
RUN sed -i 's/f"net\//f"/g' /usr/local/lib/python3.9/dist-packages/deep_sort_tools/generate_detections.py
RUN sed -i 's/^import tflite_runtime.interpreter/import tensorflow.lite/' /usr/local/lib/python3.9/dist-packages/umt/umt_utils.py
RUN sed -i 's/^from imutils.video import VideoStream/from imutils.video.strmpilibcam import CamStream as VideoStream/' /usr/local/lib/python3.9/dist-packages/umt/umt_utils.py
RUN wget -O /usr/local/lib/python3.9/dist-packages/imutils/video/strmpilibcam.py https://raw.githubusercontent.com/pageauc/MoTrack-Picam2-Demo/master/strmpilibcam.py
RUN sed -i -r 's/hflip=False\):/hflip=False,src=None\):/' /usr/local/lib/python3.9/dist-packages/imutils/video/strmpilibcam.py 
RUN sed -i 's/np\.int/int/g' /usr/local/lib/python3.9/dist-packages/deep_sort_tools/generate_detections.py
RUN sed -i -r 's/np\.float\)/np.float32\)/g' /usr/local/lib/python3.9/dist-packages/deep_sort/detection.py
RUN sed -i -r 's/np\.float\)/np.float32\)/g' /usr/local/lib/python3.9/dist-packages/umt/umt_utils.py
