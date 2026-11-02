# D5 — Cognitive map as a training-time privileged input

**Status:** The user completed the two matched 4-rank `val_unseen` evaluations on 2026-10-09; the CPU paired summary is below. Removing the raster and generated direction vectors leaves overall SR at 64.27%, but the 1839-episode equality is partly due to filling six previously missing cache entries. On the 1833 originally shared entries, deploy SR is 0.22 percentage points lower, with no significant paired difference. The historical mapless “~63 SR” checkpoint is still not identifiable, so the asserted gain over ETP-R1 remains unverified. The full attention-alignment probe is also not ready; see T3.

## T0. Provenance and input semantics

| Item | Confirmed finding |
|---|---|
| 28000 checkpoint | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/store/ckpt.iter28000.pth` |
| Original DAgger config | `/data/xukai/etp-r1-snapshot/checkpoints/data_try5_local/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml`: pretrain 460000, `LLMGridTry5Policy`, `SS-ETP-LLM`, try5/llm_grid, key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`, teleport, tryout true, seed 100, 30k iterations, LR 1e-5. `MODEL.elevation_axis` is **not recorded** in this historical dump; HEAD evaluation defaults to `y`, but historical training axis is not established by that dump. |
| D2 controls | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_zero_grid.sh` uses `probe.llm_zero_grid.v1`, 4 ranks and yielded 64.06 SR. Historical own 64.33 used `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_llm_try5_control_val_unseen.sh`, 1 rank. They are **not a matched same-rank pair**; D5 reruns both arms. |
| Mapless ETP-R1 “~63” | No traceable checkpoint + training config + same-harness per-episode output was found. The historical launcher's `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth` and `/home/xukai/code/ETP-R1-snapshot/ETP-R1/pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt` are absent locally. A separate 63.13 figure belongs to a GT-teacher run and must not be substituted. Thus E-base is blocked on checkpoint provenance. |
| Pretraining | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/submit/llm-grid-try5-pretrain.sh:7-12` starts the LLM-map Try5 stage from ETP-R1 `model_step_367500.pt`, uses the above LLM cache key, and requests four GPUs. The 460000 checkpoint is 92,500 step numbers later; an uninterrupted additional 92,500-step run is an **inference**, not verified from the missing original pretraining log. Mapless ETP-R1 starts from the 367500 baseline family, but its DAgger checkpoint is unverified. |
| Direction usage | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_utils.py:136-148` maps the five `direction_vectors` into `map_trajectory_metadata`. `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_encoder.py:338-361` concatenates their ten values with two start-direction and two start-position values into one **always-valid** metadata token. All-zero direction has no “absent” mask/padding interpretation; it is an ordinary input. No separate `deploy-nodir` cache is needed. |
| Prompt-start audit | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/make_deploy_cache.py` reconstructed the same start metadata rule as `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/llm_boxes_navigation_cache.py:425-469` and `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/llm_grid_navigation_cache.py:278-287`. Among 1833 original val_unseen NPZs, `start_direction_vector` mismatch **0** and `start_position` mismatch **0** after float32 conversion. Original raster is absent for IDs 308, 349, 455, 804, 1318, 1331; their start fields are reconstructed from the same prompt-input rule. |

**Deployment caveat:** the cache-generation prompt's `start_position` is obtained from semantic-level local coordinates using the *ground-truth trajectory* when calling `SceneSemanticBoxes.relevant_to` (`llm_boxes_navigation_cache.py:425-450`, `prior/bbox/_relevance.py:27-63`). Its equality with the cached prompt value does **not** prove it can be generated from start pose alone in a genuinely new episode. D5 tests “no LLM output” under the existing benchmark metadata contract, not a fully verified no-GT-input deployment pipeline. Also, the current `metadata_only` switch still loads a complete LLM NPZ before zeroing the grid; D5's separate cache tests the proposed values without removing that loader dependency.

## T1. Matched deploy-view evaluation

The new model key is `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__deploy-zero`, created only under `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation/`. It contains **1839/1839** val_unseen rasters and status JSONs; the LLM cache report says available=1839, missing=0. Each NPZ field is: `grid` float32 `(37,100,100)` all zero; `direction_vectors` float32 `(5,2)` all zero; `start_direction_vector` float32 `(2,)` from prompt input; `start_position` float32 `(2,)` from prompt input; `range_y` copied from the original NPZ where present, otherwise `[None,None]` (object array). Twenty files passed the actual direction5 tensor loader. Original cache files were not touched.

Evaluation scripts, to be run serially on GPUs **1,5,6,7** in one tmux session:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/run_eval.sh
```

This calls `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/eval_own.sh` (port 29660), then `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/eval_deploy.sh` (port 29661). Both are copied from `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_zero_grid.sh`. `diff -u` confirmed exactly these three changed lines in each:

| Argument | D2 zero-grid parent | E-own | E-deploy |
|---|---|---|---|
| `--master_port` | 29650 | 29660 | 29661 |
| `--exp_name` | `probe_llm_zero_grid_val_unseen` | `d5_own_val_unseen` | `d5_deploy_val_unseen` |
| `llm_cache_model_key` | `probe.llm_zero_grid.v1` | original full key | original full key + `__deploy-zero` |

All other arguments—including 4 ranks, GPU selection, checkpoint, policy, source, teleport and configured tryout—remain identical. The two completed aggregate result files were written at 02:15:03 and 02:19:28 local time on 2026-10-09; the scripts did not record an overall start timestamp, so exact total wall time is unavailable.

After both finish, CPU summary (1839 and 1833 shared-original-cache episodes; SR/SPL/OSR/nDTW/NE/PL, exact McNemar and scene-cluster bootstrap 95% CI):

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/summarize.py
```

| Arm | N | SR | SPL | OSR | nDTW | NE | PL |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-own | 1839 | 64.27 | 54.17 | 69.06 | 66.20 | 3.994 | 13.688 |
| E-deploy | 1839 | 64.27 | 53.46 | 68.90 | 66.71 | 3.888 | 13.911 |
| E-own, original-cache subset | 1833 | 64.48 | 54.35 | 69.29 | 66.41 | 4.007 | 13.733 |
| E-deploy, original-cache subset | 1833 | 64.27 | 53.48 | 68.90 | 66.73 | 3.889 | 13.901 |
| E-base | blocked: checkpoint/harness unidentified | — | — | — | — | — | — |

Source: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d5_{own,deploy}_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r[0-3]_w4.json`, merged by `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/summarize.py`. Four rank files and **1839 distinct episode IDs** were present in each arm. The aggregate JSON independently reports 6 own-cache misses and 0 deploy-cache misses.

| Deploy vs own | N | SR difference (pp) | Scene-bootstrap 95% CI (pp) | Deploy-only / own-only successes | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| All episodes | 1839 | +0.00 | [−0.97, +1.02] | 65 / 65 | 1.000 |
| Original-cache subset | 1833 | −0.22 | [−1.14, +0.78] | 61 / 65 | 0.789 |

Of the six episodes missing from the original LLM cache, the deploy cache succeeds on **308, 349, 455 and 1318**, while own is assigned failure on all six; deploy still fails on 804 and 1331. Hence full-set SR parity conceals four fewer deploy successes on the 1833 truly paired original-cache episodes. SPL falls by 0.71 pp on all episodes and 0.87 pp on the common subset; the McNemar p value does not test SPL. Compared with the old 64.33 own reference, the matched rerun is 64.27, confirming that the reference alone was not an adequate pair.

**T1 inference:** at this checkpoint, there is no detectable SR benefit from the raster plus predicted direction vectors on these episodes under the same 4-rank harness. The 1833-episode SR CI still permits about a one-point effect in either direction; this is not proof of exact equivalence or “zero cost” for all metrics. Without E-base, it cannot establish a significant gain over mapless ETP-R1.

## T2. Training-content control

No existing run was verified to share iter28000's Try5 map-enabled structure, 460000 initialization and DAgger hyperparameters while receiving zero map fields throughout DAgger. The previous “structure difference excluded” assertion lacks an identifiable run/config pair. A new `__zero-all` cache sets both **LLM-generated** fields (`grid`, `direction_vectors`) to zero and preserves `range_y` and the prompt-supplied start fields. It preserves the original available-episode set to keep DAgger sampling matched; it is not the 1839-complete deployment cache. `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/make_zero_all_cache.py` generated train/val_seen/val_unseen counts **10707/768/1833**; `llm_navigation_cache_report` confirmed those counts and identical available episode-ID lists to the source key (train's 112, val_seen's 10, val_unseen's 6 missing remain missing). Twenty val_unseen files passed the actual direction5 loader with both generated fields zero. The DAgger `train_90` subset lies within the train split.

Prepared control launcher: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/train_zero_all_dagger.sh`. Its call to `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_d3.sh` differs from the prepared HEAD full-map control `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_full_dagger_control.sh` only in run name, cache key and port (29664). **Do not compare this HEAD run directly to historical 28000 as if code were controlled.** The D3 common launcher explicitly sets `load_pretrained_map_modules=False`, elevation `y`, checkpoint interval 400 and other HEAD settings; for content causality, run the already prepared HEAD full-map control alongside the new zero-all control.

No training was started. If the user elects the matched pair, each needs four selected GPUs; for the zero-map arm only:

```bash
D3_GPUS=1,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/train_zero_all_dagger.sh
```

This isolates **DAgger-stage** predicted-map content only. To isolate pretraining content, repeat the Try5 pretraining from 367500 with the LLM-generated fields zeroed, then start matched DAgger runs from its endpoint. The original stage spans checkpoint indices 367500→460000 (at most an inferred 92,500 additional optimizer steps) and the maintained Slurm script requests four GPUs. Thus a rough compute budget is `92,500 × four-GPU step time`, plus DAgger; no trustworthy wall-time estimate is available from this audit. No pretraining launcher was created.

## T3. Language–vision alignment probe: feasibility, not a result

`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:1019-1030` uses `policy.net.vln_bert.graph_query_text` immediately before the action head. A forward hook can capture pre-softmax graph→text attention scores of shape `(batch, 12 heads, graph nodes, text tokens)`; softmax over valid text tokens is required. The crossmodal encoder also has four graph→text layers (`global_encoder.encoder.x_layers[0..3].visual_attention.att`), each with 12 heads. Attention entropy (A1) is computable, but lower entropy by itself does not prove better alignment. A2 requires selected-node/episode/pose bookkeeping during rollout. More decisively, `/home/xukai/code/ETP-R1-snapshot/ETP-R1/prior/grid_map/_cognitive.py:220` `extract_categories` returns category IDs, **not tokenizer spans**; an exact A3 “landmark-token attention share” cannot be computed from its output. Guessing spans from category names would silently omit synonym/semantic matches. Therefore no `attn_probe.py` or GPU command was emitted as though A1–A3 were implemented. The T3 claim remains **untested** pending a defined token-span annotation/alignment rule and a reviewed eval-time recorder. The missing mapless baseline also blocks the requested paired comparison.

## T4. Optional GT-noise design (not executed)

For a future controlled arm, use GT legacy maps and calibrate two independent operations on `train_90`: (i) episode/category deletion and insertion to match LLM region-presence precision/recall near 0.92; (ii) within-map region translation/block jitter to lower native 100×100 region IoU toward 0.33 while preserving the calibrated presence counts. Retain start metadata and the same schema, and reject calibrations that match only pooled IoU while markedly changing per-category support. `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d5/check_gt_noise.py` is a CPU-only validator for a **future, already-created** candidate namespace; it reports region IoU and episode-wise presence precision/recall over train_90, using the D3 targets 0.3339/0.9192/0.9128. No noisy cache or training launcher was produced under this optional item.

## Decision boundary

The matched evaluation is consistent with map-field removal having little effect on **iter28000's SR**, subject to the ±~1 pp scene-bootstrap interval and the SPL decrease. It does **not** yet prove genuinely LLM-free deployment, because the retained prompt-local start position was constructed with the benchmark's GT trajectory/semantic-level preprocessing (T0). The stronger claim “training-time maps cause a gain over mapless ETP-R1” additionally needs a traceable E-base and the matched HEAD full-versus-zero DAgger control; pretraining contribution is separate. Those causal claims remain open.
