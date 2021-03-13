sudo apt install wget -y
wget https://raw.githubusercontent.com/hainesdata/rpi-urban-mobility-tracker/master/shell/init.sh
chmod +x init.sh
UMT_DIR=${HOME}/umt_output && mkdir -p ${UMT_DIR}
./init.sh
