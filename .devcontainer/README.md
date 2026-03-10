# Devcontainer Setup

This documentation provides a way to setup the environment with Dev Containers, which should be easier and reproducible.

## Habitat-Lab Download

Download [habitat-lab@0.1.7](https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.1.7.zip) and unzip it under `data/`:

```bash
cd data/
wget https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.1.7.zip
unzip v0.1.7.zip
rm v0.1.7.zip
```

## Build Image

1. Install Visual Studio Code
2. Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this folder in VSCode, reopen in devcontainer and wait for the image to build

Note: You may continue to the next section while the image is building.

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
