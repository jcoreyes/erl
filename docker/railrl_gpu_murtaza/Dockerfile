FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2.5.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
RUN apt-get -y install git make cmake unzip
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
RUN pip install gym
#RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Add codebase stub ==========
RUN mkdir /root/code
ADD railrl-env.yml /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

RUN echo "source activate railrl-env" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

# gpu theanno
ENV THEANO_FLAGS mode=FAST_RUN,device=gpu,floatX=float32

ENV PATH /opt/conda/envs/railrl-env/bin:$PATH

RUN  apt-get install libglfw3

ENV DISPLAY :0


# https://github.com/NVIDIA/nvidia-docker/issues/136
# install EGL-Gears
RUN apt-get update && apt-get install -y \
    mesa-utils mesa-utils-extra

# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV DEFAULT_DOCKCROSS_IMAGE thewtex/opengl

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  libgl1-mesa-dri \
  menu \
  net-tools \
  openbox \
  python-pip \
  sudo \
  supervisor \
  tint2 \
  x11-xserver-utils \
  x11vnc \
  xinit \
  xserver-xorg-video-dummy \
  xserver-xorg-input-void \
  websockify && \
  rm -f /usr/share/applications/x11vnc.desktop && \
  apt-get remove -y python-pip && \
  wget https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  pip install supervisor-stdout && \
  apt-get -y clean

COPY etc/skel/.xinitrc /etc/skel/.xinitrc

RUN useradd -m -s /bin/bash user
USER user

RUN cp /etc/skel/.xinitrc /home/user/
USER root
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user


RUN git clone https://github.com/kanaka/noVNC.git /opt/noVNC && \
  cd /opt/noVNC && \
  git checkout 6a90803feb124791960e3962e328aa3cfb729aeb && \
  ln -s vnc_auto.html index.html

# noVNC (http server) is on 6080, and the VNC server is on 5900
EXPOSE 6080 5900

COPY etc /etc
COPY usr /usr

ENV DISPLAY :0

WORKDIR /root

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]