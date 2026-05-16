# OccWorld-Style Map Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive instruction-to-map predictor with an OccWorld-style latent map prior and verify it with predictor-only training.

**Architecture:** Frozen VLN text encoder feeds learned latent map tokens through cross-attention and transformer blocks. A CNN decoder upsamples the latent grid into dense `(37, SIZE, SIZE)` logits. Trainer uses all visible GPUs, sparse-map metrics, and sparse prior calibration.

**Tech Stack:** PyTorch, existing VLN-BERT text encoder, `DataParallel`, cognitive map `.npz` training pairs.

---

### Task 1: Predictor Architecture

**Files:**
- Modify: `vlnce_baselines/models/etp_imagined/instruction_map_predictor.py`
- Modify: `vlnce_baselines/models/etp_imagined/policy.py`
- Modify: `vlnce_baselines/models/etp_imagined/train_map_predictor.py`

- [ ] Replace `InstructionCognitiveMapPredictor` internals with latent-grid tokens, cross-attention to text, transformer blocks, and progressive CNN upsampling.
- [ ] Keep constructor compatible with existing call sites: `hidden_size`, `num_heads`, `num_layers`, `dropout`.
- [ ] Add optional args for `latent_grid_size` and `decoder_channels`, with safe defaults.
- [ ] Keep output shape `(B, NUM_MAP_CATEGORIES, SIZE, SIZE)`.
- [ ] Update trainer and policy to pass same architecture defaults.

### Task 2: Trainer Metrics And GPU Use

**Files:**
- Modify: `vlnce_baselines/models/etp_imagined/train_map_predictor.py`

- [ ] Wrap frozen text encoder plus predictor in one `DataParallel` module.
- [ ] Use only user-selected cards through `CUDA_VISIBLE_DEVICES`; run commands with `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- [ ] Report `target_pos`, `prob_mean`, `prob_pos`, `prob_neg`, `pred_pos@t`, `iou@t`, `precision@t`, `recall@t`, `top1pct_recall`, and `top5pct_recall`.
- [ ] Use per-channel weighted BCE/focal support and sparse prior output bias.

### Task 3: Verification And Milestone Commit

**Files:**
- Modify: changed source files from Tasks 1-2

- [ ] Run `python -m compileall vlnce_baselines/models/etp_imagined`.
- [ ] Run bounded trainer smoke test on GPUs 0-3.
- [ ] Tune default positive weight/thresholds if predictions collapse all-positive or all-negative.
- [ ] Commit once architecture compiles and smoke training produces non-degenerate metrics.
