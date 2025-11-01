#!/bin/bash


sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y \
python3-pip \
python3-opencv \
libgl1-mesa-glx \
libglib2.0-0 \
libatlas-base-dev \
libjpeg-dev \
libpng-dev


python3 -m pip install --upgrade pip


pip install \
ultralytics \
torch \
torchvision \
tflite-runtime \
onnxruntime \
numpy \
pillow \
opencv-python \
scikit-learn \
scipy \
pandas \
matplotlib \
tqdm \
psutil

echo "Installation complete!"
