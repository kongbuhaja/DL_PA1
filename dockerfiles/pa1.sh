sudo docker build --force-rm -f pa1.dockerfile -t pa1:1.0 .
sudo docker run --gpus all --cpuset-cpus=0-31 -m 120g --shm-size=16g -it -v /home/dblab/hs/DL_PA1:/home/DL_PA1 --name dl_pa1 pa1:1.0
