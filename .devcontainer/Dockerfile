FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up basic utilities
RUN apt-get update && \
    apt-get install -y git git-lfs wget curl vim build-essential ca-certificates unzip bash-completion screen && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -all

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Switch to non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode -s /bin/bash
USER vscode
WORKDIR /home/vscode

# Accept conda ToS and initialize conda
RUN conda tos accept && conda init bash

# Create conda environment from environment.yaml (uses channels aihabitat + conda-forge)
COPY --chown=vscode:vscode environment.yaml /home/vscode/environment.yaml
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f /home/vscode/environment.yaml && \
    conda clean -afy && \
    rm /home/vscode/environment.yaml

# Install additional Python packages using pip
COPY --chown=vscode:vscode requirements.txt /home/vscode/requirements.txt
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate etpr1 && \
    pip install -r /home/vscode/requirements.txt && \
    rm /home/vscode/requirements.txt

# Ensure the environment is activated on shell startup
RUN echo "conda activate etpr1" >> ~/.bashrc
