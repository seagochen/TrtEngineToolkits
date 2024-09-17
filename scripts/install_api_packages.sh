#!/bin/bash

# 更新软件包列表
sudo apt update

# 安装 OpenCV 所需依赖项
echo "Installing OpenCV dependencies..."
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev

# 安装 OpenCV
echo "Installing OpenCV..."
sudo apt install -y libopencv-dev

# 检查安装情况
echo "Installation complete. Checking versions:"
opencv_version=$(pkg-config --modversion opencv4)
echo "OpenCV version: $opencv_version"
