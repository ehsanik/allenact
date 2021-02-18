#!/bin/zsh
# Use deep learning instance, with g4dn.12xlarge, change storage
#curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#sudo apt-get install python3-distutils
#sudo apt-get install python3-apt
#sudo python3 get-pip.py
cd ~/allenact
sudo apt-get install xinit
sudo python3 xstart.py&
#sudo apt-get install python3-dev
#sudo apt-get install libffi-dev
sudo pip3 install -r small_requirements.txt
#sudo mount /dev/sda1 ~/storage
git init
git remote add origin https://github.com/ehsanik/dummy.git
git add README.md
git commit -am "something"
python3.6 main.py -o experiment_output -b projects/armnav_baselines/experiments/ithor/ armnav_ithor_rgb_simplegru_ddppo
#ssh -NfL 6006:localhost:6006 ubuntu@ec2-34-220-231-225.us-west-2.compute.amazonaws.com

