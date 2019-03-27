# Taken from https://github.com/halhenke/docker-sniper/blob/master/docker-master/Dockerfile

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

USER root

RUN apt-get update \
  && apt-get -y install wget locales git bzip2 curl \
  && rm -rf /var/lib/apt/lists/*

RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen en_US.utf8 \
  && /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8%

WORKDIR /root

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64

# Install sudo
RUN apt-get update && \
  apt-get -y install sudo \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m docker \
  && echo "docker:docker" | chpasswd && adduser docker sudo

RUN git clone --recursive https://github.com/mahyarnajibi/SNIPER.git

WORKDIR /root/SNIPER/SNIPER-mxnet

RUN apt-get update && \
  apt-get -y install \
    libatlas-base-dev \
    libopencv-dev \
    libopenblas-dev \
    gcc-5 \
    g++-5
  && rm -rf /var/lib/apt/lists/*

# Not using any extra processes because it would always blow up due to memory usage
RUN make USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
# RUN make -j [NUM_OF_PROCESS] USE_CUDA_PATH=[PATH_TO_THE_CUDA_FOLDER]

WORKDIR /root/SNIPER
