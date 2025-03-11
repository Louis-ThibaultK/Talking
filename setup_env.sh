#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
# Install ffmpeg
conda install -y -c conda-forge ffmpeg
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
sudo apt -y install libgl1

# Soft links for the auxiliary models
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s $(pwd)/latentsync/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
ln -s $(pwd)/latentsync/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
ln -s $(pwd)/latentsync/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth