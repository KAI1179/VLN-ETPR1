# Integration Plan: Full Cognitive Map Input for ETP-PriorGT

## Status Update (2026-03-18)

- The old `data/prior` subtree has been removed from this repository.
- PriorGT now consumes precomputed cognitive maps directly from `data/cognitive_maps`.
- PriorGT now feeds the full map (target size `100 x 100`) into the map encoder instead of per-step local crops.

## Runtime Hotfixes (2026-03-22)

- Fixed eval checkpoint routing for GRPO trainers by propagating `EVAL.CKPT_PATH_DIR`
  to both `IL.ckpt_to_load` and `GRPO.ckpt_to_load` in base eval setup.
- Fixed eval/inference token preprocessing crash (`RuntimeError: Could not infer dtype of dict`)
  by ensuring instruction dicts are always converted and `txt_task_encoding` is
  always present even when `task_type` is not passed explicitly.
- Base eval/inference paths now pass explicit `task_type` (`r2r -> 1`, `rxr -> 2`)
  and `max_text_len` into `extract_instruction_tokens`.

## Goal

Use an episode-level cognitive map as an additional spatial signal for ETP-R1 navigation, with minimal changes to the original ETP-R1 code path.

## Data Contract

Each episode map is loaded from:

- `data/cognitive_maps/{scene_id}/episode_{episode_id}.npz`

Expected keys inside each `.npz`:

- `grid`: `(num_categories, H, W)` float32
- `offset_x`: float (metadata)
- `offset_z`: float (metadata)
- `range_y`: float or array (metadata)

Runtime behavior:

- PriorGT uses `grid` as input features.
- `offset_x`, `offset_z`, `range_y` are preserved for compatibility but are not required for full-map mode.

## Runtime Pipeline (Full-Map Mode)

1. Load per-episode map (`grid`) at rollout start.
2. Convert to fixed shape `(C, 100, 100)` by center-crop or zero-pad if needed.
3. Stack per-env maps into `(B, C, 100, 100)`.
4. Apply weighted category embedding and CNN encoder.
5. Fuse resulting `map_embeds` into navigation branch.

## Implementation Notes

- Shared utility: `vlnce_baselines/models/etp_prior_gt/map_utils.py`
  - `full_cognitive_map(...)` returns fixed-size full map tensors.
  - `make_zero_map(...)` provides fallback when map file is missing.
- Trainers:
  - `vlnce_baselines/ss_trainer_ETP_PriorGT.py`
  - `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`
  - Both now use full-map tensors for `mode='map_encoding'`.
- Config:
  - `MODEL.MAP_ENCODER.map_size = 100`
  - `MODEL.MAP_ENCODER.crop_radius` is deprecated in full-map mode.

## Probe Mode Compatibility

Probe mode (`MODEL.MAP_ENCODER.freeze_base=True`) remains supported:

- Base model is frozen.
- Only `map_encoder.*` parameters are trainable.
- DDP uses unused-parameter handling in probe runs.

## Validation Checklist

- Training starts and passes early iterations without tensor shape mismatch.
- No `torch.stack` shape errors for map tensors.
- Probe mode runs without DDP unused-parameter reduction errors.
- Log confirms map encoder path is active when `MODEL.MAP_ENCODER.enabled=True`.

## Next Optional Improvements

- Remove deprecated `crop_radius` key after all launch scripts/configs migrate.
- Add one-time startup logging of loaded map shape per rank for faster diagnostics.
- Add a tiny unit test for full-map center crop / zero-pad behavior.
