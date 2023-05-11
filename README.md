# UMT-VAMOS

## Installation
- Install Docker
- Add non-root user to Docker user group:
```sh
sudo usermod -aG docker pi
```

2) Open a terminal and create a directory for the UMT output:
```sh
UMT_DIR=${HOME}/umt_output && mkdir -p ${UMT_DIR}
```

3) Move into the new directory:
```sh
cd ${UMT_DIR}
```

4) Download the Dockerfile and build it:
```sh
wget https://raw.githubusercontent.com/nathanrooy/rpi-urban-mobility-tracker/master/Dockerfile

docker build . -t umt
```

5) Start the Docker container:
```sh
docker run --rm -it --privileged --mount type=bind,src=${UMT_DIR},dst=/root umt
```

6) Test install by downloading a video and running the tracker:
```sh
wget https://github.com/nathanrooy/rpi-urban-mobility-tracker/raw/master/data/videos/highway_01.mp4

umt -video highway_01.mp4
```
