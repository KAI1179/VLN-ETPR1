#!/usr/bin/env bash
set -euo pipefail

wheel_dir="${1:-/tmp/etpr1-nvidia-wheels}"
mkdir -p "${wheel_dir}"

download_wheel() {
    local url="$1"
    local sha256="$2"
    local filename="$3"
    local path="${wheel_dir}/${filename}"

    if [[ -f "${path}" ]] && echo "${sha256}  ${path}" | sha256sum -c - >/dev/null 2>&1; then
        return
    fi

    curl -L --fail --retry 3 --retry-delay 5 --output "${path}" "${url}"
    echo "${sha256}  ${path}" | sha256sum -c -
}

# Use developer.download.nvidia.com directly. The nvidia-pyindex redirect through
# developer.nvidia.com/w/ can return corrupt wheel bytes in some network routes.
download_wheel \
    "https://developer.download.nvidia.com/compute/redist/nvidia-cuda-runtime/nvidia_cuda_runtime-11.1.74-py3-none-manylinux1_x86_64.whl" \
    "a267c8563e68a7841f4e13aa9cb2d6d8c590649d9cc4fd59d349d6e1a2a1e0d0" \
    "nvidia_cuda_runtime-11.1.74-py3-none-manylinux1_x86_64.whl"
download_wheel \
    "https://developer.download.nvidia.com/compute/redist/nvidia-cuda-nvrtc/nvidia_cuda_nvrtc-11.1.105-py3-none-manylinux1_x86_64.whl" \
    "dfeef66560f12495ac10dfa94d4cb912550ce6097231e3acaaeaac32c7664c58" \
    "nvidia_cuda_nvrtc-11.1.105-py3-none-manylinux1_x86_64.whl"
download_wheel \
    "https://developer.download.nvidia.com/compute/redist/nvidia-cublas/nvidia_cublas-11.5.1.101-py3-none-manylinux1_x86_64.whl" \
    "c3b8dc2d9c4a22ca93fea4bdf15d7b861b89247aa33b6848fd3bfe3320cb9323" \
    "nvidia_cublas-11.5.1.101-py3-none-manylinux1_x86_64.whl"
download_wheel \
    "https://developer.download.nvidia.com/compute/redist/nvidia-cudnn/nvidia_cudnn-8.2.0.51-py3-none-manylinux1_x86_64.whl" \
    "ca579e66254abb13bd81b1d55cfc1d81ad9a4db2bc5f6b97b4c9d8d597df37df" \
    "nvidia_cudnn-8.2.0.51-py3-none-manylinux1_x86_64.whl"
download_wheel \
    "https://developer.download.nvidia.com/compute/redist/nvidia-tensorrt/nvidia_tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl" \
    "419a1d9c6c62b5a419d67d2730a618ec0658adcc5e1080450ae351ef9fe0088c" \
    "nvidia_tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl"

python -m pip install --no-deps \
    "${wheel_dir}/nvidia_cuda_runtime-11.1.74-py3-none-manylinux1_x86_64.whl" \
    "${wheel_dir}/nvidia_cuda_nvrtc-11.1.105-py3-none-manylinux1_x86_64.whl" \
    "${wheel_dir}/nvidia_cublas-11.5.1.101-py3-none-manylinux1_x86_64.whl" \
    "${wheel_dir}/nvidia_cudnn-8.2.0.51-py3-none-manylinux1_x86_64.whl" \
    "${wheel_dir}/nvidia_tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl"
