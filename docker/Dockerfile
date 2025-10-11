# ==============================================================================
# ===== STAGE 1: The "Builder" Base =====
# ==============================================================================
# This stage installs all heavy, slow-to-compile, and infrequently changing
# dependencies as root. It will be cached and reused in subsequent builds.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS builder

ARG PYTHON_VERSION=3.8

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    OMP_NUM_THREADS=16

# --- Install Base OS and Python Dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential cmake ninja-build sudo \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    libopenblas-pthread-dev && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# --- Install Core Python Frameworks (The Slow Parts) ---
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

ENV LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# Install mmcv-full from source to ensure CUDA compatibility.
# force mmcv to install with cuda
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
RUN pip install --no-cache-dir mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html --no-binary mmcv-full

# Install PyTorch3D from source for Pulsar support
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d && \
    cd /tmp/pytorch3d && \
    pip install --no-cache-dir . && \
    cd / && \
    rm -rf /tmp/pytorch3d && \
    rm -rf /var/lib/apt/lists/*

# ==============================================================================
# ===== STAGE 2: The Final Development Image (MODIFIED TO RUN AS ROOT) =====
# ==============================================================================
FROM builder

# --- Install MMDetection and its specific dependencies ---
RUN pip install --no-cache-dir \
    mmdet==2.24.0 \
    mmsegmentation==0.24.0 \
    yapf==0.31.0 \
    wandb \
    nuscenes-devkit \
    "networkx<2.3,>=2.2" \
    plyfile \
    scikit-image \
    "numba==0.53.0" \
    "trimesh>=2.35.39,<2.35.40" \
    "numpy<1.24"

# --- Environment Customization for root ---
# Re-enable the colored bash prompt for the root user. This is the line you wanted.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# --- Project Setup ---
# Copy the entire project context into the workspace
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Install the project in editable mode. This will be a system-wide install for the root user.
# RUN pip install -v -e .

# --- Container Defaults ---
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
CMD ["bash"]