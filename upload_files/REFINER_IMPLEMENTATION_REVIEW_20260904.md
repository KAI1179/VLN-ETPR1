# Refiner implementation review package

## 1. Scope

- Repository: `/home/xukai/code/ETP-R1-snapshot/ETP-R1`
- Branch: `exp/refiner`
- Clean Try5 base: `75358899544441988be73d7d49cfb74e7b2bef97`
- Current implementation: `019e0c8f352d9628efa40076c464f354237657a6`
- Commits:
  - `b12eac4 feat: add cognitive map refiner training core`
  - `019e0c8 feat: integrate online evidence refiner with Try5`

The implementation is additive to clean Try5. When
`MODEL.MAP_ENCODER.refiner_ckpt` is empty, the original Try5 path is retained.

## 2. Implemented components

1. `OnlineEvidence`
   - Accumulates 27 semantic channels plus observed/free/blocked masks.
   - Uses depth, semantic observations and sensor poses to project evidence into
     the start-aligned 100 x 100 grid at 0.5 m resolution.

2. `CognitiveMapRefiner`
   - Four-level U-Net, input 67 channels (`P0[37] + E[30]`) and output 37
     sigmoid probabilities.
   - Uses base width 64 and GroupNorm(8).

3. Refiner dataset and training
   - Reads packed trajectory evidence and the corresponding P0/GT maps.
   - Samples six steps per training trajectory.
   - Applies object loss in observed cells and region loss globally.
   - Reports P0/refined mIoU at 100 x 100 and 10 x 10 for seen, unseen, route and
     route-seen areas.
   - Selects `best.pt` by val-unseen route-seen object mIoU.

4. Evidence collection
   - Supports teacher and perturbed collection passes.
   - Uses the existing Try5 waypoint and panorama stages to construct reachable
     ghost candidates; it does not use language/map/navigation logits to choose
     actions.
   - Saves one compressed trajectory file per episode and collection kind.

5. DAgger/evaluation integration
   - Adds the 12-view semantic sensors only when a refiner checkpoint is set.
   - Loads the refiner separately from the policy, freezes it and excludes it
     from policy optimization/checkpointing.
   - Updates accumulated evidence before map-input construction, refines P0 and
     supplies the composed cognitive map to the unchanged Try5 fusion path.

## 3. Changed code files

1. `prior/online_evidence.py`
2. `scripts/refiner/collect.py`
3. `scripts/refiner/train.py`
4. `vlnce_baselines/common/environments.py`
5. `vlnce_baselines/config/default.py`
6. `vlnce_baselines/models/refiner/__init__.py`
7. `vlnce_baselines/models/refiner/dataset.py`
8. `vlnce_baselines/models/refiner/model.py`
9. `vlnce_baselines/ss_trainer_ETP_PriorGT.py`

Total delta from clean Try5: 1,238 inserted lines across nine files.

## 4. Verification completed

- Python 3.8 compilation passed for all nine changed Python files.
- `git diff --check 7535889..HEAD` passed.
- CPU forward check passed for `CognitiveMapRefiner`, producing shape
  `(1, 37, 100, 100)` with finite sigmoid output.
- Collection and training command-line parsers were imported successfully.
- The exact Try5 checkpoint was found and loaded:
  - Path: `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`
  - SHA-256: `3ea8868fcb6d5d225b906cbe4ea6781fd7fc76697842ca5ed7a29761fce0fffb`
  - Raw state-dict entries: 522
  - Try5 `map_encoder` and `graph_map_attention` parameters are present.
  - `bert.global_encoder.graph_map_attention.residual_projection.weight` has
    shape `(768, 768)`, finite values and norm `13.726471900939941`.

`ruff` and `ty` executables were not available in either the current shell or
the `etpr1-py38` environment, so those checks were not run.

## 5. Not yet completed

- The bounded 100-episode collection run and its three visual reports.
- Full train/val data collection.
- Refiner training and metric evaluation.
- DAgger/evaluation with a trained refiner checkpoint.

Reason: the current dev container exposes no `/dev/nvidia*`; under
`etpr1-py38`, PyTorch reports `cuda_available=False` and `device_count=0`.
No CPU or substitute-checkpoint run was used in place of the required GPU test.

## 6. Review points and known constraints

- `scripts/refiner/collect.py` still has the repository-relative old checkpoint
  as its default. For the verified checkpoint, explicitly pass:
  `--pretrained-path /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`.
- Collection cannot omit every policy forward pass because reachable ghost
  candidates require the existing waypoint and panorama encoders. Action
  selection remains independent of navigation logits.
- The taskbook composition preserves unseen P0 only for the 27 object channels;
  the 10 region channels are replaced globally by refiner output.
- Runtime correctness involving Habitat semantic IDs, sensor poses and
  VectorEnv alignment still requires the bounded GPU collection test.
- Refiner integration currently targets DAgger/evaluation. Inference-specific
  environment construction has not been separately verified.
- Although training uses the `train_90` dataset suffix, collection currently
  writes files under `data/refiner/train/`, rather than a `train_90/` directory.
  The training script expects the same current path, but it differs from the
  taskbook's naming.
- GT includes soft value `0.6`. The implementation applies the calculated
  positive-class weight to every positive soft-label element; this exact
  weighting behavior should be reviewed before the full training run.

## 7. Package contents

- This review report.
- The design/risk notes and two implementation taskbooks.
- All nine changed source files, preserving repository-relative paths.
- Two Git-format patches, one per implementation commit.
- No checkpoint, dataset, semantic cache, generated trajectory or training log.
