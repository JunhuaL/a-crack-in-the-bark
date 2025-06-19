FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# bug in MacOS Docker: http://github.com/docker/for-mac/issues/7025
#RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy

# Set time-zone programatically
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install system dependencies with Python and MIG support
RUN apt-get update && \
    apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    git \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc-12 \
    g++-12 \
    curl \
    sudo \
    sshfs \
    bzip2 \
    libgtk2.0-dev \
    wget \
    python3.10 \
    python3-pip \
    python3-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && apt-get -y purge manpages-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure CUDA paths and MIG capabilities
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_VERSION=12.6
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_DISABLE_REQUIRE=1
ENV NVIDIA_VISIBLE_DEVICES=all

# custom user, useful if mounting remote storage
ARG NB_USER="appuser"
ARG NB_UID="123456"
ARG NB_GID="1234567"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    HOME="/home/${NB_USER}"
RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" && \
    mkdir -p "${HOME}" && \
    chown "${NB_USER}:${NB_GID}" "${HOME}"
RUN adduser ${NB_USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# sshfs - only needed if mount remote storage
RUN echo "user_allow_other" >> /etc/fuse.conf
RUN mkdir /mnt/fast && chmod 777 /mnt/fast

# conda
WORKDIR "${HOME}"
ARG CONDA_FILE="Miniconda3-py39_24.11.1-0-Linux-x86_64.sh"
ARG PYTHON_VER="3.10.14"
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=${HOME}/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/${CONDA_FILE} \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && conda install -y python==${PYTHON_VER} \
    && conda clean -ya

# CUDA and pytorch
RUN conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda==11.7 -c pytorch -c nvidia \
    && conda clean -ya

RUN conda install typing_utils -c conda-forge

ADD ./ ./

RUN pip install --no-cache-dir torch==1.13.0 transformers==4.42.4 diffusers==0.21.1 numpy==1.26.3 datasets ftfy matplotlib accelerate==0.32.1 scikit-image scikit-learn
RUN pip install huggingface_hub==0.28.0

# fix huggingface_hub issue: https://github.com/easydiffusion/easydiffusion/issues/1851
RUN sed -i 's/from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info/from huggingface_hub import HfFolder, hf_hub_download, model_info/' /home/appuser/miniconda/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py

ARG CACHEBUST=1

# Create workspace directories
RUN mkdir -p /workspace/kubernetes/results

# Set working directory
WORKDIR /workspace/kubernetes
CMD ["true"]
