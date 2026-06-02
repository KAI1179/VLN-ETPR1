# Devcontainer Setup

This documentation provides a way to setup the environment with Dev Containers, which should be easier and reproducible.

## Habitat-Lab Download

Download [habitat-lab@0.1.7](https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.1.7.zip) and unzip it under `data/` before reopening in the devcontainer:

```bash
cd data/
wget https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.1.7.zip
unzip v0.1.7.zip
rm v0.1.7.zip
```

<details><summary>What post-create will do</summary>

The devcontainer post-create step will install this checkout into `etpr1-uv` with:

```bash
python setup.py develop --all --no-deps
```

It also patches Habitat-Lab's local `habitat_baselines/rl/requirements.txt` so `tensorflow==1.13.1` is only requested on Python versions below 3.8. This avoids the unavailable TensorFlow 1.13.1 wheel on Python 3.8 while keeping the rest of the Habitat-Lab baseline package installed.

The post-create step also installs `.devcontainer/sitecustomize.py` into the
environment's `site-packages` directory. Python imports this file at startup, so
legacy Habitat-Lab and Habitat-Sim worker processes see `np.float` mapped to the
builtin `float` without rewriting files under `data/` or `site-packages`. It also
sets default Habitat-Sim native logging controls, `GLOG_minloglevel=2` and
`MAGNUM_LOG=quiet`, before Habitat-Sim is imported.

</details>

## Build Image

1. Install Visual Studio Code
2. Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this folder in VSCode, reopen in devcontainer and wait for the image to build

<details><summary>Env details</summary>

The image creates and activates the `etpr1-uv` conda environment from
`conda-linux-64.lock`. Conda owns Python, Habitat-Sim, Gym 0.21.0, and
binary/system-level packages, while uv installs the Python package layer from a
temporary export of `uv.lock`. Human-edited Python dependencies live in the PEP 621
dependency list in `pyproject.toml`; the lockfile records the resolved artifacts.
`scripts/install-conda-activation-hooks.sh` installs activation hooks that put
`$CONDA_PREFIX/lib` first in `LD_LIBRARY_PATH`; this makes compiled Conda packages
such as `llvmlite` load Conda's `libstdc++.so.6` instead of an older system copy.
The same hook sets default native Habitat-Sim logging controls so shell-launched
training does not print C++ INFO messages such as `SemanticScene.h` teardown logs.
Gym stays in Conda because its published PyPI extras metadata is invalid under
modern package parsers. The `[tool.uv]` settings in `pyproject.toml` keep the
PyTorch CUDA wheel source and legacy build constraints next to the dependency
manifest.
TensorRT remains outside uv because NVIDIA serves the 7.2.3.4 wheels from its
own binary host. `.devcontainer/install-tensorrt.sh` downloads exact wheel files
from `developer.download.nvidia.com`, verifies their SHA-256 hashes, and installs
them with `pip --no-deps`. This avoids the `nvidia-pyindex` redirect through
`developer.nvidia.com/w/`, which can return corrupt wheel bytes on some network
routes.

</details>

## Dataset Download

### 1. Matterport3D (MP3D) Dataset

> [!WARNING]
> We assume that you have obtained [permission from Matterport](https://niessner.github.io/Matterport/#:~:text=Please%20fill%20and%20sign%20the%20Terms%20of%20Use%20agreement%20form%20and%20send%20it%20to%20matterport3d%40googlegroups.com%20to%20request%20access%20to%20the%20dataset.).

We recommend using [Matterport3D-Dataset-Downloader](https://github.com/wtzmx/Matterport3D-Dataset-Downloader) for faster download speed. Clone the repo, install dependencies as instructed and run in its folder:

```shell
python3 download_mp.py --task_data habitat --scans scans.txt -o /path/to/data/scene_datasets/mp3d/
```

Then, move the files so that the file structure follows this format: `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be **90 scenes** in total:

```shell
ls ./data/scene_datasets/mp3d | wc
     90      90    1080
```

### 2. Extra Files

```shell
wget "https://huggingface.co/datasets/cepillarskeira/ETP-R1-extra-files/resolve/main/extra_files.zip?download=true" --content-disposition
unzip extra_files.zip
# Before running the script below, please open `copy_extra_files.py` and modify the `source_root` and `target_root` variables to match your local absolute paths.
python copy_extra_files.py

# Clean up (Optional)
rm -rf extra_files extra_files.zip
```

## Model Download

### Llama

1. Request access on [Llama website](https://www.llama.com/llama-downloads/).
2. Request access on [HuggingFace (again)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), with the same email and info.
3. After you've got access, download the HuggingFace model with the tool of your choice.
     - Save under `data/models/`.
     - Remember to configure username and token.
     - Personally I use [`hfd.sh`](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f):

          ```shell
          hfd.sh meta-llama/Llama-3.1-8B-Instruct -x 8 --hf_username $HF_USERNAME --hf_token $HF_TOKEN
          ```

## Working With uv

### Adding Packages

```shell
$ uv add --no-sync --bounds exact pytest
$ uv pip install --python "$CONDA_PREFIX/bin/python" pytest==<locked-version>
```
