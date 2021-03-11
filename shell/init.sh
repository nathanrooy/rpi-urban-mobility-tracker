UMT_DIR=${HOME}/umt_output
rm ${UMT_DIR}/.bash_history
rm -rf ${UMT_DIR}/.cache
cd ${UMT_DIR}
docker build . -t umt --build-arg CACHEBUST=$(date +%s)
docker run --rm -it --privileged --mount type=bind,src=${UMT_DIR},dst=/root umt