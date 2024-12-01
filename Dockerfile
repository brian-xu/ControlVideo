# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=20.04
ARG NVIDIA_CUDA_VERSION=11.6.1
# CUDA architectures, required by Colmap and tiny-cuda-nn. Use >= 8.0 for faster TCNN.
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"

# Pull source either provided or from git.
FROM scratch as source_copy

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_XCB_GL_INTEGRATION=xcb_egl

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y 'ppa:deadsnakes/ppa' && \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libpython3.8-dev \
        python3.8-dev \
        python3-dev \
        python3-pip

# Upgrade pip and install dependencies.
# pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118 && \
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70.0.0' && \
    pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 'numpy<2.0.0' --extra-index-url https://download.pytorch.org/whl/cu116
    
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir accelerate==0.17.1 addict==2.4.0 basicsr==1.4.2 bitsandbytes==0.35.4 clip==0.1.0 cmake==3.25.2 controlnet-aux==0.0.6 decord==0.6.0 deepspeed==0.8.0

RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir diffusers==0.14.0 easydict==1.10 einops==0.6.0 ffmpy==0.3.0 ftfy==6.1.1 imageio==2.25.1 imageio-ffmpeg==0.4.8 moviepy==1.0.3 numpy==1.24.2

RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir omegaconf==2.3.0 opencv-python==4.7.0.68 pandas==1.5.3 pillow==9.4.0 scikit-image==0.19.3 scipy==1.10.1 tensorboard==2.12.0

RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir tensorboard-data-server==0.7.0 tensorboard-plugin-wit==1.8.1 termcolor==2.2.0 thinc==8.1.10 timm==0.6.12 tokenizers==0.13.2

RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export FORCE_CUDA=1 && \
    pip install --no-cache-dir tqdm==4.64.1 transformers==4.26.1 wandb==0.13.10 xformers==0.0.16 modelcards matplotlib mediapipe positional-encodings

        
# Fix permissions
RUN chmod -R go=u /usr/local/lib/python3.8

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y 'ppa:deadsnakes/ppa' && \
    apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        python3.8 \
        python3.8-dev \
        build-essential \
        python-is-python3 \
        ffmpeg

# Copy packages from builder stage.
COPY --from=builder /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/

# Bash as default entrypoint.
CMD /bin/bash -l
