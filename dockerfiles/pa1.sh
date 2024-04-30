sudo docker build --force-rm -f pa1.dockerfile -t pa1:1.0 .
sudo docker run --gpus "all" --cpuset-cpus=0-15 -m 60g --shm-size=8g -it -v /home/hs/ML/DL_PA1:/home/DL_PA1 --name dl_pa1 pa1:1.0
