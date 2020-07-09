# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libav-tools \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -q

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

ENV LANG C.UTF-8

### setup Mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/mjkey.txt
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# If you're curious what the DEBIAN_FRONTEND is all about:
# https://askubuntu.com/questions/506158/unable-to-initialize-frontend-dialog-when-using-ssh
#RUN apt-get purge nvidia*
#RUN add-apt-repository --yes ppa:graphics-drivers && apt-get update

#RUN apt-get update && apt-get install -y libav-tools zlib1g-dev swig cmake

# For some reason this works despite an error showing up...
#RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
#ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384

RUN mkdir -p /usr/lib/nvidia-000
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-000

WORKDIR /mujoco_py
COPY ./requirements.txt /mujoco_py/
RUN pip install -r requirements.txt

RUN mkdir /root/code
WORKDIR /root/code

### install mujoco-py
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN pip install mujoco-py==2.0.2.2
RUN python -c 'import mujoco_py'

### additional python dependencies
RUN pip install gym==0.12.1
RUN pip install pybullet==2.5.5
RUN pip install psutil==5.7.0
RUN pip install torch==1.0.0
