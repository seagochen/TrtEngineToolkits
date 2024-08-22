- [1. 安装Docker Desktop for Windows](#1-安装docker-desktop-for-windows)
- [2. 安装WSL 2](#2-安装wsl-2)
- [3. 安装NVIDIA Container Toolkit](#3-安装nvidia-container-toolkit)
- [4. 拉取并运行Nvidia CUDA容器](#4-拉取并运行nvidia-cuda容器)
  - [4.1. 拉取镜像](#41-拉取镜像)
  - [4.2. 创建Dockerfile](#42-创建dockerfile)
  - [4.3. 构建镜像](#43-构建镜像)
  - [4.4. 运行容器](#44-运行容器)
  - [4.5. 验证PyTorch的GPU支持](#45-验证pytorch的gpu支持)
  - [4.6. （可选）保存镜像](#46-可选保存镜像)


# 1. 安装Docker Desktop for Windows

1. 从[Docker官网](https://www.docker.com/products/docker-desktop)下载Docker Desktop。
2. 运行安装程序并按照提示完成安装。
3. 在安装过程中确保启用了Windows的WSL 2（Windows Subsystem for Linux）。

# 2. 安装WSL 2

1. 打开PowerShell或命令提示符，运行以下命令来启用WSL和虚拟机平台：

    ```sh
    wsl --install
    ```

2. 重新启动计算机以完成安装。
3. 安装Ubuntu或其他你喜欢的Linux发行版。可以从Microsoft Store中选择并安装，例如Ubuntu

    ```sh
    wsl --install -d Ubuntu-22.04
    ```

# 3. 安装NVIDIA Container Toolkit

1. 确保你的Windows机器上安装了NVIDIA GPU驱动。
2. 打开WSL 2的Linux终端，配置生产环境库：

    ```sh
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

3. 可选：配置库以使用实验包：

    ```sh
    sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

4. 更新库中的包列表：

    ```sh
    sudo apt-get update
    ```

5. 安装NVIDIA Container Toolkit包：

    ```sh
    sudo apt-get install -y nvidia-container-toolkit
    ```

# 4. 拉取并运行Nvidia CUDA容器

## 4.1. 拉取镜像

打开WSL 2的Linux终端，拉取Nvidia CUDA Docker镜像，版本号尽量不要选择太新，因为我们很多现有的库和工具都是基于旧版本开发的。

    ```sh
    docker pull nvidia/cuda:11.4.3-devel-ubuntu20.04
    ```

如果需要其他版本的，可以在下述网页中找到相关信息：

https://hub.docker.com/r/nvidia/cuda/tags


## 4.2. 创建Dockerfile

然后我们要来创建一个基于上述镜像的自定义镜像

创建一个Dockerfile，用于定义你需要的环境和工具。在你的工作目录下创建一个名为 Dockerfile 的文件，并在其中添加以下内容：

    ```sh
    # 使用NVIDIA的CUDA基础镜像
    FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

    # 设置时区
    ENV TZ=Asia/Tokyo
    RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

    # 设置工作目录
    WORKDIR /workspace

    # 将此目录设置为默认启动目录
    CMD ["bash"]
    ```

## 4.3. 构建镜像

在Dockerfile所在的目录中，通过以下命令构建你的Docker镜像：

    ```sh
    docker build -t new_image .
    ```

这个命令会根据Dockerfile中的指令创建一个名为 pytorch-cuda 的Docker镜像。


## 4.4. 运行容器

使用以下命令运行这个镜像，并确保GPU可以正常使用：

    ```sh
    docker run --gpus all -it --privileged -v $(pwd):/workspace -e DISPLAY=$DISPLAY --rm pytorch-cuda
    ```



**其他一些比较常用的Docker命令选项说明**

- `-it`: 以交互模式运行
- `--gpus all`: 使GPU可在容器中访问
- `--rm`: 退出时删除容器
- `--privileged`: 赋予容器对主机资源的访问权限
- `-v`: 指定挂载目录
- `-e DISPLAY=$DISPLAY`: 将显示变量传递给容器


## 4.5. 验证PyTorch的GPU支持

进入容器后，可以通过以下Python代码验证PyTorch是否正确检测到了GPU：

    ```sh
    python3 -c "import torch; print(torch.cuda.is_available())"
    ```

## 4.6. （可选）保存镜像

如果你希望在镜像中保存你的更改，可以在退出容器前运行以下命令：

    ```sh
    docker commit <CONTAINER_ID> pytorch-cuda:latest
    ```

这样你以后就可以直接使用这个镜像了。
