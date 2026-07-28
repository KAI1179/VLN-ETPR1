# RGB-D Segmenter Static Candidate Audit

## Status

This is the P5.1 primary-source audit required by the frozen RGB-D segmenter
benchmark contract. No repository, checkpoint, or dependency has been
downloaded or installed.

The static outcome is:

- **ESANet: conditional primary, blocked at the weight-license gate**;
- **DFormerv2-S: blocked challenger** until its weight-license and depth-
  encoding ambiguities are resolved;
- **DFormerv2-B/L and original DFormer: deferred** because their additional
  compute or integration cost is not justified before the Small checkpoint
  establishes value.

“Conditional primary” does not authorize checkpoint download or benchmark
inference. Weight permission, the checkpoint byte hash, extracted
configuration, isolated environment, and frozen preprocessing must pass before
the first model load.

## ESANet Primary

### Pinned source and checkpoint

- official repository:
  <https://github.com/TUI-NICR/ESANet>;
- revision:
  `820c5bb633e49e69dcd075d4330165bb540a0cc9`;
- candidate architecture: ESANet-R34-NBt1D;
- source dataset: NYUv2, 40 classes;
- selected checkpoint archive:
  `nyuv2_r34_NBt1D_scenenet.tar.gz`;
- official Google Drive ID:
  `1w_Qa8AWUC6uHzQamwu-PAqA7P00hgl8w`;
- expected extracted checkpoint:
  `nyuv2/r34_NBt1D_scenenet.pth`;
- official reported NYUv2 mIoU: `51.58`;
- downloaded archive and checkpoint SHA-256: pending, and mandatory before
  model loading.

The SceneNet-pretrained variant is frozen because it has the best officially
reported NYUv2 endpoint for the same architecture and runtime. The ordinary
`50.30` checkpoint is not an adaptive fallback.

The repository received a December 2025 maintenance update and now documents a
Python 3.12 / PyTorch 2.9.1 environment. It supplies evaluation, sample
inference, ONNX, TensorRT, and latency paths. Its maintained recipe used
validation batch 24 at `480×640`, so a twelve-view `256×256` inference batch is
plausible but remains an OOM-gated smoke rather than an assumed fact.

### License gate

The source license is Apache-2.0. The official README discusses use of network
weights and requests citation, but it does not attach an explicit license or
permission grant to the externally hosted checkpoint. Therefore the code
license passes and the weight license remains unspecified. Under the frozen
contract this blocks download and inference until either permission is
clarified or an explicit project risk exception is separately recorded. A
citation request is not treated as a license grant, and the checkpoint is never
redistributed.

### Frozen preprocessing

The official test path:

- reads RGB in RGB order;
- bilinearly resizes RGB and nearest-neighbour resizes depth when resizing is
  requested;
- converts RGB to `[0,1]` and applies ImageNet mean/std
  `[0.485,0.456,0.406] / [0.229,0.224,0.225]`;
- applies checkpoint-dataset depth mean/std;
- restores invalid raw-depth zeros after normalization;
- bilinearly resizes logits with `align_corners=False`.

The maintained ESANet recipe normally resizes to its configured `480×640`
test shape. The benchmark deliberately disables that resize to preserve the
shared square camera geometry and proposes native `256×256` inference. This is
an experiment-specific deviation, not the official default, and must pass a
technical input/logit alignment smoke before inference is accepted.

Habitat metric depth follows the pinned NYUv2 preparation exactly:
`(depth_m * 1000).astype(uint16)`, followed by conversion to float and the
checkpoint's raw-depth normalization; zero remains invalid. This preserves the
official millimetre quantization over the benchmark's `0–10 m` range. The exact
depth mean/std and model arguments must be read from the selected official
archive and frozen in the candidate manifest before inference. No scale is
selected from oracle labels or benchmark quality.

Primary sources:

- repository setup and official checkpoint table:
  <https://github.com/TUI-NICR/ESANet/tree/820c5bb633e49e69dcd075d4330165bb540a0cc9>;
- preprocessing:
  <https://github.com/TUI-NICR/ESANet/blob/820c5bb633e49e69dcd075d4330165bb540a0cc9/src/preprocessing.py>;
- sample inference and explicit depth scaling:
  <https://github.com/TUI-NICR/ESANet/blob/820c5bb633e49e69dcd075d4330165bb540a0cc9/inference_samples.py>;
- Apache-2.0 license:
  <https://github.com/TUI-NICR/ESANet/blob/820c5bb633e49e69dcd075d4330165bb540a0cc9/LICENSE>.

## DFormerv2 Challenger

### Pinned source and checkpoint

- official repository:
  <https://github.com/VCIP-RGBD/DFormer>;
- revision:
  `814799bb1f39eb380f72fdea1cd591f2cc27b6aa`;
- candidate: DFormerv2-S;
- source dataset: NYUv2, 40 classes;
- checkpoint: `DFormerv2_Small_NYU.pth`;
- official host:
  <https://huggingface.co/bbynku/DFormerv2/blob/main/DFormerv2/NYU/DFormerv2_Small_NYU.pth>;
- advertised size: 107 MB;
- checkpoint SHA-256: pending;
- official reported NYUv2 mIoU: `56.0`;
- reported model cost: 26.7M parameters and 33.9 GFLOPs per `480×640`
  view.

DFormerv2-B and -L require about 216 MB and 383 MB of weights, while the paper
reports substantially higher per-view compute for modest NYUv2 gains. They do
not enter the current twelve-view benchmark. Original DFormer is also deferred
because DFormerv2-S is the maintained family member with the best accuracy-cost
case.

### Blocking license and preprocessing findings

The repository `LICENSE` file is MIT, but the repository README separately says
the code is for non-commercial use only. The Hugging Face model card declares
no checkpoint license. This benchmark will not infer with the checkpoint until
the applicable internal-research permission is made unambiguous and recorded.
No weight is redistributed.

The official data recipe converts depth arrays to grayscale PNG through
`plt.imsave(..., cmap="Greys_r")`. DFormerv2 constructs geometry priors from
depth differences, so an unverified substitution of Habitat metric metres
would change the model contract. Before DFormerv2 can leave `blocked`, its exact
NYUv2 depth decode, normalization, invalid-value handling, and geometry-prior
input must be reproduced from the pinned code and frozen without consulting
benchmark labels.

Primary sources:

- setup, checkpoints, and data recipe:
  <https://github.com/VCIP-RGBD/DFormer/tree/814799bb1f39eb380f72fdea1cd591f2cc27b6aa>;
- NYUv2 vocabulary/config:
  <https://github.com/VCIP-RGBD/DFormer/blob/814799bb1f39eb380f72fdea1cd591f2cc27b6aa/local_configs/_base_/datasets/NYUDepthv2.py>;
- repository license:
  <https://github.com/VCIP-RGBD/DFormer/blob/814799bb1f39eb380f72fdea1cd591f2cc27b6aa/LICENSE>;
- DFormerv2 paper:
  <https://arxiv.org/abs/2504.04701>;
- official checkpoint directory:
  <https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/NYU>.

## Frozen NYUv2-to-Canonical Mapping

ESANet and DFormerv2 NYUv2 checkpoints share the same ordered 40-class source
vocabulary. The primary 23-channel mapping is frozen below before any
checkpoint inference.

| NYUv2 source | canonical target | kind |
| --- | --- | --- |
| wall | — | diagnostic `structure` only |
| floor | — | diagnostic `free-space` only |
| cabinet | cabinet | direct |
| bed | bed | direct |
| chair | chair | direct |
| sofa | sofa | direct |
| table | table | direct |
| door | door | direct |
| window | — | diagnostic `structure` only |
| bookshelf | — | ignored |
| picture | picture | direct |
| counter | counter | direct |
| blinds | — | ignored |
| desk | table | many-to-one |
| shelves | — | ignored |
| curtain | — | ignored |
| dresser | chest_of_drawers | synonym |
| pillow | cushion | synonym |
| mirror | — | ignored |
| floor mat | — | diagnostic `free-space` only |
| clothes | clothes | direct |
| ceiling | — | diagnostic `free-space` only |
| books | — | ignored |
| refridgerator | appliances | many-to-one; source spelling preserved |
| television | tv_monitor | synonym |
| paper | — | ignored |
| towel | towel | direct |
| shower curtain | — | ignored; not a shower |
| box | — | ignored |
| whiteboard | — | ignored |
| person | — | ignored |
| night stand | cabinet | many-to-one |
| toilet | toilet | direct |
| sink | sink | direct |
| lamp | — | ignored |
| bathtub | bathtub | direct |
| bag | — | ignored |
| otherstructure | — | diagnostic `structure` only |
| otherfurniture | — | ignored |
| otherprop | — | ignored |

This covers 17 of 23 deployable target categories:

```text
appliances, bathtub, bed, cabinet, chair, chest_of_drawers, clothes,
counter, cushion, door, picture, sink, sofa, table, toilet, towel, tv_monitor
```

The six explicitly unsupported targets are:

```text
fireplace, gym_equipment, plant, seating, shower, stool
```

The mapping passes the preregistered 14-category static count gate. Its 80%
target-cell support gate is computed only after the cohort manifest is sealed.
Ignored classes remain abstentions and are never mapped to `other`.

## Priority

1. Seal and validate the 50-observation cohort.
2. Resolve the ESANet checkpoint permission gate; do not download while it is
   unspecified.
3. Download and hash only the selected ESANet archive after dependency and
   license records are ready.
4. Extract and freeze its model arguments and depth statistics.
5. Run one-view and twelve-view technical smokes without inspecting semantic
   quality.
6. Keep DFormerv2-S blocked unless both blockers are resolved independently of
   benchmark labels.
