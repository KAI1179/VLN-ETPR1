# LLM-Grid Sanity Analysis Design

## Purpose

Run the remaining predictor-quality sanity checks before selecting an RGB-D
semantic perception model. The analysis asks whether adding English RxR
training data improved the R2R LLM-Grid predictor, whether the current category
F1 is informative, and how checkpoint behavior changes from early training to
the final epoch.

The analysis uses existing R2R validation prediction caches and evaluator
outputs only. It must not train a model or generate predictions. If a requested
comparison lacks cached predictions or evaluator outputs, that comparison is
reported as postponed rather than reconstructed from checkpoints.

## Package Boundary

Downstream analysis does not belong in
`vlnce_baselines/models/etp_llm/`. That package retains training, prediction,
schema, and evaluator primitives.

The analysis lives in a focused subpackage:

```text
prior/analyze/llm_grid/
├── __main__.py
├── metrics.py
├── plots.py
└── samples.py
```

`metrics.py` owns completed evaluator-artifact loading, population validation,
diagnostic baselines, and tabular exports. `plots.py` owns quantitative
figures. `samples.py` owns fixed qualitative-example selection, rendering, and
comparison sheets. `__main__.py` provides the `tap.Tap` orchestration CLI.
`prior/analyze/batch_vis.py` may gain the smallest reusable API needed to
render an explicit set of example IDs. No new analysis module is added to
`etp_llm`.

Tests mirror those responsibilities:

```text
tests/analyze/
├── test_llm_grid_metrics.py
├── test_llm_grid_plots.py
└── test_llm_grid_samples.py
```

Complete generated artifacts go under:

```text
outputs/llm_grid_analysis/r2r_rxr_sanity/
```

Selected figures and comparison sheets referenced by the experiment notes go
under:

```text
docs/images/llm_grid_r2r_rxr_sanity/
```

## Fixed Inputs

The mixed R2R plus English RxR run is:

```text
outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag
```

The representative checkpoints are fixed to:

| Epoch | Role |
| --- | --- |
| 1 | Earliest checkpoint and initial baseline |
| 2 | Best observed `val_unseen` IoU candidate |
| 5 | Mid-training checkpoint with an isolated schema-validity regression |
| 10 | Final checkpoint and late-training behavior |

Their authoritative evaluator inputs are:

```text
outputs/llm_grid_eval/checkpoint-sweep-r1p5/runs/000/episodes.csv
outputs/llm_grid_eval/checkpoint-sweep-r1p5/runs/001/episodes.csv
outputs/llm_grid_eval/checkpoint-sweep-r1p5/runs/004/episodes.csv
outputs/llm_grid_eval/checkpoint-sweep-r1p5/runs/009/episodes.csv
```

The corresponding cached rasters are:

```text
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-1
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-5
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
```

Every quantitative check uses the complete and identical R2R populations:

- 778 `val_seen` episodes.
- 1,839 `val_unseen` episodes.

Missing or invalid predictions remain in the denominator as empty predictions,
matching the evaluator contract. The analysis fails explicitly if checkpoint
coverage, split membership, or episode identities differ.

## Qualitative Sample Comparison

The surviving historical R2R-only visualization directory contains 20
`val_unseen` examples:

```text
docs/images/llm-grid-s2.legacy.r1p5.direction5.comparison/
```

Seventeen of those examples have valid raster predictions in all four selected
mixed-model checkpoints. Those 17 fixed IDs form the qualitative comparison
set. Reusing them makes the new mixed-model images directly comparable to the
historical R2R-only images while avoiding cherry-picking by the new analysis.

For every selected example, render the mixed-model prediction against the same
ground truth at epochs 1, 2, 5, and 10. Also produce a comparison sheet with
five labeled columns:

1. Historical R2R-only image.
2. Mixed epoch 1.
3. Mixed epoch 2.
4. Mixed epoch 5.
5. Mixed epoch 10.

The R2R-only column is labeled as a historical sample whose exact checkpoint
provenance is unavailable. It must not be presented as an epoch-matched
quantitative comparison.

Restricting visualization to schema-valid common examples could hide the epoch
5 regression. The report therefore includes per-epoch invalid counts from the
full validation populations next to the qualitative samples.

## IoU Distribution

Use the per-episode `category_aware_raster_iou` values already present in each
`episodes.csv`. Invalid predictions retain their evaluator-provided zero IoU.

Produce:

- A fixed-bin histogram grid with split as rows and epoch as columns. The
  x-axis is IoU in `[0, 1]`; the y-axis is the number of episodes.
- A tolerant-IoU survival plot for each split. For threshold `t`, plot the
  number and fraction of episodes satisfying `IoU >= t` for all four epochs.

All epochs share bin edges and threshold values so visual differences are not
plotting artifacts. Export the underlying histogram and threshold counts to
CSV as well as PNG.

## Predict-All-Categories Baseline

The baseline predicts the presence of all 27 object categories and all 10
region categories for every episode. It does not fill spatial cells and does
not alter spatial metrics.

For an episode with `t_o` target object categories and `t_r` target region
categories:

```text
all_object_f1 = 2 * t_o / (27 + t_o)
all_region_f1 = 2 * t_r / (10 + t_r)
all_combined_f1 = 2 * (t_o + t_r) / (37 + t_o + t_r)
```

For a model prediction, calculate the comparable episode-wise combined
category F1 from existing per-episode counts:

```text
actual_combined_f1 =
    2 * (object_tp + region_tp)
    / (object_predicted + region_predicted + t_o + t_r)
```

A zero denominator produces F1 zero.

Report object, region, and combined episode-wise F1 distributions and their
unweighted per-split means for the four model checkpoints and the all-category
baseline. The figure uses split columns and object/region/combined rows so the
baseline can be compared against each checkpoint without mixing supports.

This is deliberately an episode-macro diagnostic requested for the sanity
check. It does not replace or silently reinterpret the evaluator's existing
split-pooled object and region category metrics.

## Predicted Category-Count Distribution

Use the distinct-category counts already exported per episode. Produce two
histogram figures:

- Objects, with total, instruction-mentioned, and instruction-unmentioned rows.
- Regions, with total, instruction-mentioned, and instruction-unmentioned rows.

Columns correspond to epochs 1, 2, 5, and 10. `val_seen` and `val_unseen` are
shown with distinguishable traces in every panel. Object axes use the fixed
range `0..27`; region axes use `0..10`. Export exact histogram counts to CSV.

Mentioned and unmentioned partitions follow the evaluator's instruction
extractor. They do not trust a model-generated `mentioned` field.

## R2R-Only Overfitting Comparison

The requested epoch-wise comparison against the earlier R2R-only LLM-Grid
cannot currently be performed:

- R2R-only checkpoints exist locally for epochs 1 through 10.
- The documented R2R-only prediction cache is absent locally and remotely.
- No R2R-only per-episode evaluator outputs remain locally or remotely.

Running those checkpoints would require new prediction, which is outside the
approved task. The quantitative overfitting comparison is therefore
postponed and documented as missing-data work.

The surviving 20-image R2R-only visualization set may be used only for the
17-example qualitative comparison described above. It cannot recover epoch
curves or full-population metrics.

## Outputs and Reproducibility

The analysis CLI uses `tap.Tap` and accepts explicit input and output roots.
Defaults may point to the fixed experiment paths above, but calculations do not
discover arbitrary model runs or fall back to similarly named caches.

The output directory contains:

- `summary.json` with inputs, population checks, invalid counts, F1 summaries,
  and the postponed comparison.
- CSV files containing IoU histograms, tolerant-IoU curves, F1 distributions,
  and predicted category-count histograms.
- PNG figures for each quantitative check.
- A manifest of the 17 qualitative example IDs and source paths.
- Per-epoch renderings and five-column comparison sheets.

Every artifact is deterministic for identical inputs. Existing caches and
evaluator results remain read-only.

## Experiment Notes

Add a Chinese section to `docs/daily/2026-07-25.md` containing:

- Input paths and fixed population sizes.
- Why epochs 1, 2, 5, and 10 were selected.
- Links to the qualitative samples and quantitative figures.
- Exact predict-all-category F1 results and whether category F1 remains
  informative.
- IoU and predicted-category distribution findings.
- The invalid-output caveat, especially epoch 5.
- The missing-data reason for postponing the R2R-only epoch comparison.

In `docs/daily/2026-07-24.md`:

- Leave the first sampling-analysis checkbox unchecked for the user.
- Link the sanity-check section to the 2026-07-25 note.
- Check the completed IoU-distribution, all-category-F1, and predicted-category
  count children.
- Leave the R2R-only epoch-comparison child unchecked.
- Leave the parent sanity-check checkbox unchecked while that child remains
  postponed.

## Verification

Tests cover:

- Predict-all-category object, region, and combined F1 formulas.
- Comparable model combined F1, including a zero denominator.
- Fixed population and episode-identity validation.
- Histogram counts conserve the full split population.
- Invalid predictions retain zero IoU.
- Mentioned and unmentioned count columns are mapped correctly.
- Explicit qualitative example selection is deterministic and rejects missing
  required rasters.

Verification runs targeted Pytest, Ruff, and `ty` on the changed Python files.
The analysis itself runs on cached CSV, text, NPZ, and PNG files and requires no
GPU.

## Out of Scope

- Training, checkpoint evaluation, or navigation-cache generation.
- Reconstructing missing R2R-only predictions.
- Changing the LLM-Grid evaluator's established metric contract.
- Selecting a semantic segmentation or object-detection model.
- Migrating existing sweep, plotting, or bootstrap modules out of `etp_llm`;
  that cleanup may be designed separately.
