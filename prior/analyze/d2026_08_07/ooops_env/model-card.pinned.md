---
license: apache-2.0
datasets:
- JunweiZheng/s2d3d
- JunweiZheng/WildPASS
- JunweiZheng/matterport3d
language:
- en
metrics:
- mean_iou
pipeline_tag: image-segmentation
tags:
- code
---

# ECCV'24 OPS

This repository contains the pretrained CLIP model and the OOOPS checkpoints for our ECCV'24 paper: [OPS](https://junweizheng93.github.io/publications/OPS/OPS.html)

If you're interested in my research topics, feel free to check [my homepage](https://junweizheng93.github.io/) for more findings!

If you're interested in this model, please cite the following paper:

```text
@inproceedings{zheng2024open,
title={Open Panoramic Segmentation},
author={Zheng, Junwei and Liu, Ruiping and Chen, Yufan and Peng, Kunyu and Wu, Chengzhi and Yang, Kailun and Zhang, Jiaming and Stiefelhagen, Rainer},
booktitle={European Conference on Computer Vision (ECCV)},
year={2024}
}
```

## Pretrained CLIP Model

You can download the pretrained CLIP model used by OOOPS model: [pretrained CLIP](https://huggingface.co/JunweiZheng/OPS/blob/main/ViT-B-16.pt)

## OOOPS Checkpoints

You can find the two checkpoints in this repository: [OOOPS_without_REPR.pt](https://huggingface.co/JunweiZheng/OPS/blob/main/OOOPS_without_REPR.pt) and [OOOPS_with_REPR.pt](https://huggingface.co/JunweiZheng/OPS/blob/main/OOOPS_with_REPR.pt)
