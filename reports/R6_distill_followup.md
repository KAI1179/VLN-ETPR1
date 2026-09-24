# R6 GT teacher distillation follow-up

## Preparation

- Branch: `exp/refiner`.
- User GPU constraint: only GPUs 0–3 may be used. Therefore all GPU work below is limited to those four physical GPUs; six-run and three-run groups are scheduled in batches when needed. The eight-GPU step is ineligible under this constraint.
- Pre-existing tracked changes were saved as `stash@{0}` (`R6 pre-existing changes`): `scripts/distill/eval_dagger_distill.sh` and `scripts/distill/summarize_eval.py` (the former had staged and unstaged changes).
- Remote task branch was fetched and its three patch files were applied with `git am`: `ec5fde0`, `d0f820e`, `6a98eb9`.
- The first `git am` invocation encountered a stale incomplete `.git/rebase-apply` state; `git am --abort` cleared it and the required patch sequence was then applied successfully. No patch conflict occurred.

## Step 0 — pretrained map loading (CPU)

The literal command from the task book was attempted first. It failed before importing the checker because the default Python 3.13 NumPy extension requires `GLIBCXX_3.4.29`, which is unavailable in the system `libstdc++.so.6`. The complete traceback is in `reports/step0_map_loading.txt`. The task book names `/home/xukai/anaconda3/envs/etpr1-py38`, so the same command is being retried with that interpreter.

## Step 1 — teacher map ablations and LLM map controls

GPU assignment and results pending. Only physical GPUs 0–3 are permitted; runs will be batched if all four are occupied.

## Step 2 — student GT-map evaluation

Command (corrected interpreter):

```bash
PYTHONPATH=. /home/xukai/anaconda3/envs/etpr1-py38/bin/python scripts/distill/check_pretrained_map_loading.py --dagger-ckpts data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter14000.pth /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth data/logs/checkpoints/s4_try5_refiner/ckpt.iter15000.pth 2>&1 | tee -a reports/step0_map_loading.txt
```

The checker reached the model-loading checks but failed. Raw output is in `reports/step0_map_loading.txt`.

| Check | Result |
|---|---:|
| Checkpoint total tensors | 522 |
| Pretrained `map_encoder.*` tensors | 37 |
| Pretrained `bert.global_encoder.graph_map_attention.*` tensors | 6 |
| `load_pretrained_map_modules=False` landed fusion tensors | 3/6 |
| `load_pretrained_map_modules=False` residual projection norm | 15.3716 |
| `load_pretrained_map_modules=True` strict navigation transfer | failed |

The exact terminal exception was:

```text
RuntimeError: Pretrained graph_map_attention tensors were remapped but did not land in the navigation model: ['graph_map_attention.attention.in_proj_weight', 'graph_map_attention.attention.in_proj_bias', 'graph_map_attention.attention.out_proj.weight', 'graph_map_attention.attention.out_proj.bias', 'graph_map_attention.residual_projection.weight', 'graph_map_attention.residual_projection.bias']
```

Gate G0: **not satisfied** (not 6/6 and strict transfer failed). Per the task book, the loader variants are ineligible for step 4.

## Step 3 — paired CPU analysis

Command and result pending.

## Step 4 — eight-GPU training decision

Not eligible under the user constraint allowing only GPUs 0–3. This step will not be started.
