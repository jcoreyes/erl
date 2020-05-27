# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

# Now let's download python 3 and all the dependencies
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
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
COPY ./files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Not sure why this is needed
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
