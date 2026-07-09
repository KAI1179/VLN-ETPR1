# LLM-Grid-Probe Design

## Purpose

Create a predictor-only LLM milestone that tests whether an instruction-tuned
causal language model can generate the grid portion of a Try5-style cognitive
map. This is an ablation probe, not a navigation candidate: it predicts only
the raster grid target and does not claim comparability with PriorGT or
LLM-Navigation.

The first run should answer whether compact grid JSON is learnable and
parsable before we spend effort on VLN integration or non-grid metadata
prediction.

## Naming

- `LLM-Grid-Probe`: grid-only predictor milestone. Non-grid metadata is not
  part of the candidate comparison.
- `LLM-Grid`: reserved for a non-cheating candidate that predicts the grid plus
  the non-observation map metadata needed by Try5-style consumption.

## Location

Put implementation under the existing LLM package:

```text
vlnce_baselines/models/etp_llm/train_llm_grid_probe.py
vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md
tests/etp_llm/test_train_llm_grid_probe.py
scripts/submit/llm-grid-probe-train.sh
```

This keeps the probe next to LLM-Boxes and lets it reuse the existing LLM
training conventions: chat templating, LoRA setup, EOS-safe labels/generation,
checkpoint layout, metrics output, and per-example artifacts.

## Dataset

Use VLN-CE cached maps only for v1:

```text
data/cognitive_maps/gt.legacy.r1p5.direction5.v1
```

Training should mirror LLM-Boxes and load VLN-CE examples first. ETP-R1
pretraining caches under `data/cognitive_maps_etp_r1` are out of scope for v1.

## Prompt

Follow the LLM-Boxes convention:

- The system prompt defines the keyed compact JSON schema and the canonical
  object/region category names.
- The user prompt contains only per-example inputs: dataset tag, instruction,
  level-local start position, and start direction.
- Scene id and per-example scene inventories are excluded.

## Target

The target is compact JSON at `scale=2`:

```json
{"region_candidates":["living/social space"],"object_candidates":["chair"],"regions":{"living/social space":{"cells":[[1,2],[1,3,0.5]],"mentioned":false}},"objects":{"chair":{"cells":[[4,5]],"mentioned":true}}}
```

Candidate lists come first and match the region/object keys. Entity cells are
`[row,col]` for value `1.0` or `[row,col,value]` for soft cells. Other values
are emitted as compact floats, matching `prior.llm_grid_samples.serialize_grid_target`.

Full-resolution grid targets are retained for GT upper-bound and later
navigation comparison, but they are not the first LLM generation target. The
sample study showed `scale=2` is the practical v1 target: 100 sampled VLN-CE
targets averaged about 1831 tokens for keyed JSON, with 9/500 above
`max_new_tokens=4096`.

## Training

Reuse the LLM-Boxes trainer mechanics:

- Causal LM fine-tuning with LoRA.
- LoRA rank `r=32`, alpha `64`, dropout `0.05`.
- Target modules: q/k/v/o projections plus MLP projections.
- `max_new_tokens=4096`; train-time examples over this target budget are
  excluded and counted.
- EOS-safe causal labels that mask prompt and padding positions.
- Generation config passes EOS and pad token ids.
- Batch size starts at 1 or 2 depending on GPU memory.

Do not add rank sweeps, decoding constraints, or repetition-stop policies in
v1. Those are follow-up experiments if the baseline shows invalid JSON,
truncation, or repetition failures.

## Evaluation

Evaluate predictions against the `scale=2` GT grid directly. Do not upsample
for predictor metrics.

Report:

- JSON parse validity.
- Schema validity: `grid` is a list of `[category,row,col]` or
  `[category,row,col,value]` records.
- Cell precision, recall, and F1.
- Category-aware raster IoU and recall.
- Token length, target truncation rate, and generated completion length.
- Repetition diagnostics sufficient to detect tail loops.

Invalid JSON or invalid records should be counted explicitly rather than
silently repaired. Deterministic partial salvage can be added later, but v1
should first expose raw generation quality.

## Artifacts

Use the same run layout as LLM-Boxes:

```text
<output-dir>/
  artifacts/
    system_prompt.md
    <example-id>.txt or <example-id>.json
  checkpoints/
    epoch-1/
    ...
    final/
  metrics.json
```

Per-example artifacts should preserve generated text for debugging malformed
JSON and repetition. Checkpoints should be saved after each epoch and at
`final`.

## Out Of Scope

- VLN policy integration.
- Navigation cache generation.
- Direction-vector prediction.
- Using GT metadata in a reported navigation candidate.
- ETP-R1 pretraining cache training.
- Rank, target-module, prompt-format, or decoding sweeps.
- Grammar-constrained decoding.

## Follow-Up Path

If LLM-Grid-Probe produces valid and useful grids, the next design should
promote toward LLM-Grid by adding prediction of the non-observation map
metadata needed by Try5-style consumption, then generating navigation-ready
caches. A `scale=2` prediction can be expanded back to `(37,100,100)` with
deterministic nearest-neighbor upsampling when integration starts, so the Try5
VLN code path can remain unchanged.
