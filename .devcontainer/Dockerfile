FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV_NAME=etpr1-uv
ARG MINICONDA_VERSION=py310_25.11.1-1
ARG MINICONDA_SHA256=f6174765edd75e1ef2bb3f16e99266d291deebf6532f2f7e800569afc084d0eb

# Set up basic utilities
RUN apt-get update && \
    apt-get install -y git git-lfs wget curl vim build-essential ca-certificates unzip bash-completion screen && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" -O /tmp/miniconda.sh && \
    echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum -c - && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
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

# Create conda environment from an explicit linux-64 lock.
COPY --chown=vscode:vscode conda-linux-64.lock /home/vscode/conda-linux-64.lock
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n "${CONDA_ENV_NAME}" --file /home/vscode/conda-linux-64.lock -y && \
    mkdir -p "/home/vscode/.conda/envs/${CONDA_ENV_NAME}/etc/conda/activate.d" && \
    echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' > "/home/vscode/.conda/envs/${CONDA_ENV_NAME}/etc/conda/activate.d/env_vars.sh" && \
    conda clean -afy && \
    rm /home/vscode/conda-linux-64.lock

# Install additional Python packages from the project dependency manifest.
# Use pip install, not sync, so uv does not remove packages owned by Conda.
COPY --chown=vscode:vscode pyproject.toml /home/vscode/pyproject.toml
COPY --chown=vscode:vscode uv.lock /home/vscode/uv.lock
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}" && \
    uv export --directory /home/vscode --frozen --no-emit-project --format requirements.txt --no-hashes --no-annotate --no-header --index-url https://mirrors.aliyun.com/pypi/simple --output-file /tmp/etpr1-uv-requirements.txt && \
    uv pip install --python "$CONDA_PREFIX/bin/python" --directory /home/vscode --index-url https://mirrors.aliyun.com/pypi/simple -r /tmp/etpr1-uv-requirements.txt && \
    python -m pip install nvidia-tensorrt==7.2.3.4 && \
    rm /home/vscode/pyproject.toml /home/vscode/uv.lock /tmp/etpr1-uv-requirements.txt
# nvidia-tensorrt: https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802

# Ensure the environment is activated on shell startup
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc
