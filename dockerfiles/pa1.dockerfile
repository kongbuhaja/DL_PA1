# sudo docker build --force-rm -f pa1.dockerfile -t pa1:1.0 .
# sudo docker run --gpus all --cpuset-cpus=0-31 -m 120g --shm-size=16g -it -v /home/dblab/hs/DL_PA1:/home/DL_PA1 --name dl_pa1 pa1:1.0

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update 

RUN echo "== Install Basic Tools ==" &&\
    apt install -y --allow-unauthenticated \
    openssh-server vim nano htop tmux sudo git unzip build-essential iputils-ping net-tools ufw \
    python3 python3-pip curl dpkg libgtk2.0-dev \
    cmake libwebp-dev ca-certificates gnupg git 

# githup
RUN cd /home/ &&\
    git clone https://github.com/kongbuhaja/DL_PA1.git &&\
    git config --global --add safe.directory /home/DL_PA1
    
RUN echo "== Install Dev Tolls ==" &&\
    cd /home/DL_PA1/ &&\
    pip3 install -r requirements.txt

