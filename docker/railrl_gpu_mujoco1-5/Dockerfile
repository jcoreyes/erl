# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/
COPY ./requirements.dev.txt /mujoco_py/
RUN pip install -r requirements.txt
RUN pip install -r requirements.dev.txt
RUN pip install cloudpickle==0.5.2
RUN pip install cached-property==1.3.1
# For atari-py
RUN apt-get install -y zlib1g-dev swig cmake
RUN pip install gym[all]==0.10.5
RUN pip install gitpython==2.1.7
RUN pip install gtimer==1.0.0b5
RUN pip install pygame
RUN pip install awscli==1.11.179
RUN pip install boto3==1.4.8
RUN pip install dominate==2.3.1
RUN pip install ray==0.2.2
RUN pip install path.py==10.3.1

RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl

# Delay moving in the entire code until the very end.
#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
#CMD ["pytest"]
#COPY . /mujoco_py
#RUN python setup.py install
# If you're curious what the DEBIAN_FRONTEND is all about:
# https://askubuntu.com/questions/506158/unable-to-initialize-frontend-dialog-when-using-ssh
#RUN apt-get purge nvidia*
#RUN add-apt-repository --yes ppa:graphics-drivers && apt-get update
# For some reasno this works despite an error showing up...
RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384

#WORKDIR /root
#RUN git clone https://github.com/openai/mujoco-py.git
#WORKDIR /root/mujoco-py
#RUN git clone https://github.com/openai/mujoco-py.git

RUN mkdir /root/code
WORKDIR /root/code

# Forgot to install these. If you're going to rebuild this docker from scratch,
# I recommend moving these to where all the other pip installs were, but I
# was too impatient to do that now.
RUN pip install joblib==0.9.4
