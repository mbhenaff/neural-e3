#!/bin/bash

sudo apt-get -y update
sudo apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get -y install libglib2.0-0
sudo apt-get install -y libsm6 libxext6 libxrender-dev
sudo apt-get install -y python-cloudpickle
sudo apt-get install -y python-dill
sudo apt-get install -y libharfbuzz-dev
python -m pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html --user
python -m pip install --upgrade pip --user
python -m pip install opencv-python --user
python -m pip install progressbar2 --user
python -m pip install tqdm --user
python -m pip install cloudpickle --user
python -m pip install dill --user
python -m pip install click --user
python -m pip install tensorflow --user
#python -m pip install tensorflow-gpu --user
python -m pip install pandas --user

python -m pip install -e . --user


#cd stable-baselines/
#python -m pip install -e . --user
#cd ../


python -m pip install PyHamcrest --user
python -m pip install gym --user
python -m pip install gym[atari] --user
python -m pip install scikit-image --user 
python -m pip install sklearn --user 
python -m pip install -U matplotlib --user
python -m pip install tensorboardX --user
python -m pip install matplotlib --user
python -m pip install scipy --user
