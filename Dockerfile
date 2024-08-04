# set base image
FROM arm32v7/debian:buster-20201209-slim

# install python 3.7 and create alias
RUN apt-get update -y
RUN apt install -y python3.7
RUN echo 'alias python="/usr/bin/python3.7"' >> ~/.bashrc

# install python dependencies
RUN apt-get install -y python3-dev python3-setuptools python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install imutils==0.5.3
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-pillow
RUN apt-get install -y python3-opencv

# install scipy and associated dependencies
RUN apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev
RUN apt-get install -y gfortran
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y python3-scipy

# download tensorflow 2.4 wheel and install
WORKDIR /root
RUN apt-get install -y wget
RUN apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
RUN pip3 install keras_applications==1.0.8 --no-deps
RUN pip3 install keras_preprocessing==1.1.0 --no-deps
RUN apt-get install -y python3-h5py
RUN apt-get install -y openmpi-bin 
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y libatlas-base-dev
RUN pip3 install six==1.15.0
RUN pip3 install wheel==0.36.2
RUN pip3 install mock==4.0.3
RUN wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
RUN pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl --no-deps
RUN pip3 install absl-py==0.11.0
RUN pip3 install --upgrade google-api-python-client
RUN pip3 install wrapt==1.12.1
RUN pip3 install typing-extensions==3.7.4.3
RUN pip3 install opt_einsum==3.3.0
RUN pip3 install gast==0.4.0
RUN pip3 install astunparse==1.6.3
RUN pip3 install termcolor==1.1.0
RUN pip3 install flatbuffers==1.12

# install tflite runtime
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

# install some additional tools for debugging
RUN apt-get install -y feh
RUN apt-get install -y vim

# install git
RUN apt-get install -y git

# install deep sort
RUN pip3 install git+https://github.com/mk-michal/deep_sort.git@v1.2.0

# install umt
RUN pip3 install git+https://github.com/nathanrooy/rpi-urban-mobility-tracker --no-deps
