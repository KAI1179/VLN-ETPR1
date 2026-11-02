# D1 P1 full cross-scene probe: completed result

## Conclusion

The **full five-field donor swap did not reduce performance**. On R2R `val_unseen`, the same Try5 DAgger `iter28000` checkpoint reached **SR 65.42 / SPL 54.64**, versus the saved own-map control **64.33 / 54.16**. Across 1,839 paired episodes, full gained 42 successes and lost 22 (exact McNemar **p=0.0169**). The result contradicts the proposed “full drops by ~2 points” branch. It also **does not establish input insensitivity**: the measured difference is positive and episode outcomes change. Which swapped field causes it remains unknown.

## What was run

The launcher `scripts/distill/eval_probe_llm_full.sh` evaluated `data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/store/ckpt.iter28000.pth` with `LLMGridTry5Policy`, `architecture=try5`, `source=llm_grid`, and `MAP_ENCODER.enabled=True`. It used four ranks on physical GPUs 1,5,6,7, `NUM_ENVIRONMENTS=4`, `IL.back_algo=teleport`, and `MODEL.MAP_ENCODER.load_pretrained_map_modules=False` to avoid the dead *pretraining* spatial tokenizer; the nonzero DAgger map/fusion weights load afterward. The checkpoint's embedded config records `pretrained_path=.../llm_grid_try5/store2/model_step_460000.pt`, `MAP_ENCODER.enabled=True`, `architecture=try5`, and seed 100 (CPU `torch.load` audit, 2026-10-08).

The seed-0 cross-scene assignment contains 1,833 distinct available episodes and `same_scene=0` for every row (`data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0/r2r/val_unseen/donor_assignment.csv`). `scripts/distill/make_probe_metadata_donor.py` copies all five NPZ fields from each donor: `grid`, `range_y`, `direction_vectors`, `start_direction_vector`, and `start_position`. The generated model key is `probe.llm_full_s0.v1`. The six episodes without source LLM cache—308, 349, 455, 804, 1318, 1331—remain missing and are scored zero, as in own. The actual LLM cache report found 1,833 available and six missing.

Source results: `data/logs/checkpoints/{probe_llm_full_val_unseen,llm_try5_control_val_unseen,probe_llm_gtmeta_val_unseen}/eval_results/stats_ckpt_28000_val_unseen.json` and their `stats_ep_ckpt_28000_val_unseen_r*_w*.json` files. The paired table below is computed directly from those episode files using `scripts/distill/summarize_d1_probes.py` (full vs own) and the same paired-count formula (full vs GT metadata).

| Input | Ranks | Episodes | Missing | SR | SPL | OSR | nDTW |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM own | 1 | 1839 | 6 | 64.33 | 54.16 | 69.11 | 66.19 |
| Same-episode GT directions, own LLM grid/position | 4 | 1839 | 6 | 64.27 | 54.11 | 69.00 | 66.16 |
| Full cross-scene donor | 4 | 1839 | 6 | **65.42** | **54.64** | **69.93** | **66.65** |

| Pair: full versus control | Both succeed | Full only | Control only | Both fail | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Own LLM map | 1161 | 42 | 22 | 614 | 0.0169 |
| GT-direction probe | 1161 | 42 | 21 | 615 | 0.0111 |

The second pair has a matched four-rank harness and differs in cache content, but not in **only** one field: full uses cross-scene values for all five fields, whereas the GT-direction probe keeps own LLM grid/position/range and uses same-episode GT directions. An own-map four-rank rerun under the **same current code** would close the rank/code-version mismatch in the primary comparison; a repeat of one identical cache would quantify evaluation stochasticity. The saved own aggregate was written on **2026-09-22**, whereas full was written on **2026-10-08** (`stat` on the two aggregate JSONs), and the repository's map-loading checks changed in between. Treat the historical own row as a reference, not a perfectly synchronized control.

## Interpretation and limits

1. **Do not attribute the result to `start_position`.** Full changes five fields at once. The result is a gain, not the hypothesized loss. A position-only swap is still a valid causal probe, but its predicted sign is no longer supported by this result.
2. **Do not infer a pure attention-sink/register effect from “full did not drop.”** A cross-scene full swap changes 64 success outcomes versus own (42 gain, 22 loss); it may be beneficial by chance, by a particular metadata field, or through another input interaction. It does not show that episode content is irrelevant. Conversely, the paired result alone cannot prove a specific field is used.
3. **The historical “~63 SR baseline” is not an established matched control here.** `docs/NOTE.md` lists “Baseline (Dagger)” as **63.13/54.23** but leaves its checkpoint column blank; `reports/distill_eval_summary.md` and `scripts/distill/summarize_eval.py` repeat that as “Baseline DAgger (author).” These are reference rows, not a local run with a traceable checkpoint/config. Its pretraining checkpoint, seed, and `MAP_ENCODER.enabled` cannot be verified from those rows. A separate local **63.13** run in `reports/R3_smoke_report.md` is a *GT teacher try5* `iter16000` evaluation, whose checkpoint embeds `PriorGTPolicy`, `MAP_ENCODER.enabled=True`, seed **100**, and `pretrained_path=.../prior_gt_r1p5/.../try-5-r1p5_step_387500.pt` (CPU checkpoint audit). The LLM `iter28000` checkpoint also embeds seed **100**, but is `LLMGridTry5Policy`, uses a different pretraining checkpoint (`llm_grid_try5/model_step_460000.pt`), and is evaluated with LLM maps. Equal seeds do not make the two experiments otherwise matched. Neither 63.13 row establishes that structure and pretraining were held constant against LLM Try5. `MAP_ENCODER.enabled=False` would itself change the graph and therefore is not a content-only intervention.

Within **this** own/full comparison, the saved outputs refer to the same `iter28000` weights, so policy architecture, training seed, and pretraining lineage are fixed at the checkpoint level. The intended intervention is the cache; however the evaluation **rank count and run date/code state are not matched** (1 rank on Sep 22 vs 4 ranks on Oct 8). Thus the +1.09-point observed gain should not be presented as a clean estimate of any single field's contribution or of the historical +2-point Try5-vs-author-baseline gap.

## Most informative next comparisons

- Rerun **own** with the same four-rank launcher settings as full (only cache key/run name/port changed) to settle the primary harness mismatch.
- Evaluate cross-scene `metadata_only` with that same harness, then a `start_position_only` cache if the spatial-range-prior hypothesis is still of interest. Those tests separate donor direction metadata from position; `grid` still needs its own swap if field-level attribution is required.
- Do **not** claim the “LLM-predicted route spatial-range prior” mechanism until the position-only paired test shows it.
