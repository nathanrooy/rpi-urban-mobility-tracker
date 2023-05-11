# set base image
FROM arm64v8/debian:bullseye-20230502-slim

# install python 3.9 and create alias
RUN apt-get update -y
RUN apt install -y python3.9
RUN echo 'alias python="/usr/bin/python3.9"' >> ~/.bashrc

# install python dependencies
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

# install python modules
RUN pip install matplotlib
RUN pip install numpy
