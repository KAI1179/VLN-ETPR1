# LLM-Grid Candidate Design

## Purpose

Promote the grid-only LLM-Grid-Probe into a navigation candidate by making the
LLM predict every non-observation cognitive-map input needed by the Try5-style
map path. The candidate predicts sparse grid anchors plus five route direction
vectors from the instruction and start metadata. It no longer borrows
`direction_vectors` from ground truth, so it is eligible for comparison with
other non-cheating map predictors.

LLM-Grid-Probe remains the grid-only predictor milestone. It is useful for
debugging whether the sparse grid format is learnable, but it is not a
navigation candidate because it omits required metadata.

## Target Schema

The LLM-Grid candidate target is compact JSON with exactly these top-level keys
in this order:

```json
{"predicted_regions":["dining/food"],"predicted_objects":["chair","counter"],"regions":{"dining/food":{"cells":[[12,18]],"mentioned":false}},"objects":{"chair":{"cells":[[20,28]],"mentioned":true},"counter":{"cells":[[22,31]],"mentioned":true}},"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]}
```

Contract:

- `predicted_regions` replaces `region_candidates`.
- `predicted_objects` replaces `object_candidates`.
- `predicted_regions` must match the `regions` object keys in the same order.
- `predicted_objects` must match the `objects` object keys in the same order.
- `regions` and `objects` are keyed by canonical category name.
- `cells` are binary `[row,col]` entries in the downsampled `50x50` grid.
- `mentioned` stays on each entity and carries instruction mention status.
- `direction_vectors` is exactly five `[dx,dz]` vectors in the level-local
  x-z frame. Valid entries are unit vectors or `[0.0,0.0]` padding.
- No compatibility aliases are accepted.

The schema uses `direction_vectors`, not `motion_vectors`, because that is the
existing cognitive-map cache and model input name.

## Prompt Contract

The system prompt should use candidate language because the first arrays are
predicted semantic categories, not inventory dumps:

```text
Generate sparse cognitive-map anchors and direction vectors for VLN.
Each cell is [row,col] in a 50x50 grid with integers 0-49.
Grid columns follow the x axis; grid rows follow the z axis.
Return only compact valid JSON with keys: predicted_regions, predicted_objects, regions, objects, direction_vectors.
Use only categories from Allowed object categories and Allowed region categories.
Select categories that are explicitly mentioned or can be inferred from the instruction and route context.
Use canonical category names; predicted_regions/predicted_objects must match the keys of regions/objects.
direction_vectors contains exactly five [dx,dz] vectors in the x-z frame, ordered by route progress, with [0.0,0.0] padding when needed.
No markdown, prose, comments, or extra keys.
```

The user prompt should keep the existing LLM-Boxes-style per-example inputs:

```text
dataset R2R | start x = 11.7 | start z = 1.3 | direction x = -0.0 | direction z = -1.0 | instruction Turn right. Take another right after the white chairs and wait next to the counter with the bar stools.
```

This is slightly more verbose than `start=(x,z)` but matches the existing
LLM-Boxes convention and avoids ambiguity between metric coordinates and grid
indices.

## Candidate Integration

Training and evaluation use the `train_llm_grid` code path and produce the
LLM-Grid candidate contract:

- The serializer reads `grid` and `direction_vectors` from each raster cache.
- The parser validates all five top-level keys and rejects old Probe keys.
- The evaluator keeps grid metrics and adds direction-vector metrics.
- Navigation cache generation converts a valid LLM-Grid JSON output into the
  same cognitive-map tensors consumed by the Try5-style policy path.
- Invalid JSON or invalid schema is counted explicitly; no repair model or
  silent fallback is introduced.

GT remains the upper-bound candidate because it can use ground-truth grid and
ground-truth `direction_vectors`. LLM-Grid is the non-cheating candidate because
it must predict both.

## Evaluation Metrics

Keep the current grid metrics:

- JSON validity.
- Schema validity.
- Cell precision, recall, and F1.
- Category-aware raster IoU and recall.
- Target and generation token counts.

Add direction-vector metrics:

- `direction_vector_valid_rate`: schema-valid examples with exactly five valid
  vectors.
- `direction_vector_l2`: mean L2 distance over the five vectors.
- `direction_vector_cosine`: mean cosine similarity over non-padding target
  vectors.
- `direction_vector_padding_accuracy`: accuracy of predicting padding positions.

## Out Of Scope

- Keeping aliases for `region_candidates` or `object_candidates`.
- Using `motion_vectors` as a generated key.
- Predicting full-resolution `100x100` grids in the first candidate run.
- Grammar-constrained decoding.
- Rank, category-list, or prompt sweeps before establishing the candidate
  baseline.
