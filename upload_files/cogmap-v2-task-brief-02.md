# Task brief 02 — M1 serialisation + LLM data, whole-box raster, G1 registration experiment

Workspace `/home/xukai/code/cogmap-VLN-V2`, branch `exp/cogmap-v2-graph-gt` (continue on it). Brief 01 artefacts and `docs/cogmap_v2/INTERFACES.md` are the inputs; its leakage rules (strip every `gt` recursively, no scene / region-instance IDs in prompts, `mentioned` re-derived from the instruction per ADR 0002) are binding for everything below. CPU parts first; GPU parts end in ready-to-run launchers plus dry-runs — the user submits.

## Why

Brief 01 produced the GT cognitive graph, a renderer and registration labels. Two things follow immediately: (A) the LLM generator M1 can now be trained on graph targets instead of raster targets, and (B) the registration head M2 can be tested in the cleanest possible setting — GT graph raster + coordinate fusion (the recipe that reached 86 SR on `gt.legacy.r1p5`) with and without registration supervision. One fact from brief 01 shapes (B): the graph raster still carries the 1.5 m route corridor (`route_points_start` clipping), so a second, corridor-free rendering is needed to separate "route corridor" from "region layout".

## Part A — M1: compact graph text and LLM training data (CPU)

1. `prior/cogmap_v2/serialize.py`: `graph_to_text(graph) -> str` and `text_to_graph(text, instruction) -> CognitiveGraph` (round-trip exact on semantic fields). Format: one line per route node in order, then side nodes, then edges, e.g.
   `N0 start bedroom | obj: bed(m) lamp` / `N1 on_route hallway(unknown-mention) | obj: picture(m)` / `E N0>N1 short door straight none` / `S2 side bathroom | attach N1 right`. Categories are canonical strings from `prior.constants`; unknown regions serialise as `unknown`. Keep unmentioned route nodes. No coordinates, no `gt`, no IDs beyond episode-local N/S indices.
2. Token statistics with the Llama-3.1-8B tokenizer over train targets (mean / P95 / P99 / max), as `prior.llm_grid_samples` reports today; pick `--max-new-tokens` so ≤ 0.5% of targets are dropped and report the number.
3. Training files in the format the existing LLM-Grid launcher consumes (`scripts/submit/llm-grid-train-r1p5.sh` → its data loader): prompt = instruction + start pose line; completion = graph text. Splits: R2R train (+ RxR-EN train if the graph CLI supports it cheaply; otherwise R2R only and say so). Write `scripts/submit/llm-graph-train.sh` as a copy of the LLM-Grid launcher pointing at the new data and a new output dir; dry-run only.
4. `scripts/cogmap_v2/eval_graph_text.py`: given generated texts, parse with `text_to_graph`, derive `mentioned`, and score against GT graphs: route node sequence edit distance (category level), node precision/recall split into **mentioned vs unmentioned** regions, edge-label accuracy on matched edges, side-node F1, parse-failure rate. This is both the M1 metric and the future GRPO reward; make the reward function a pure callable `graph_reward(pred_text, gt_graph, instruction) -> float`.

## Part B — second rendering: whole boxes, no corridor (CPU)

Add a renderer option / namespace `gt.graph.box.direction5.v1`: `render_gt` without `route_points_start`, i.e. every graph node's region and object boxes drawn in full; metadata unchanged. Generate for the three R2R splits. Report its IoU against legacy (it will be low; what matters is that it exists). This is the honest target for M5 — a graph→layout model produces boxes, not corridors.

## Part C — G1: registration head under coordinate fusion (GPU launchers + dry-runs)

Locate the coordinate-fusion GT run that reached 86 SR (`scripts/distill/run_dagger_coord_fusion.sh`, GT arm, `COORD=True`; find run dir, iter and its `eval_args.txt` / `config.yaml`) and use its exact recipe as the template.

1. **Registration head (M2, supervision only).** In the Try5 policy, after graph–map fusion, add a small head on each topological-graph node feature predicting (a) the graph route node index it lies in (softmax over up to `N_MAX=16` route nodes, padded; side nodes excluded) and (b) progress in [0,1]. Labels during DAgger rollout: node world xz → level xz (`graph.gt.level_origin` from the registration NPZ; the trainer already knows the scene) → `RegistrationLabels.locate_position(level_xz, path_index_hint=<last label of that node's predecessor>)`; use `selected_level_mask` semantics: nodes with `distance_m > 2.0` get label `-1` and are ignored in the loss. Loss weight `IL.reg_weight` (default 0.1) added to the DAgger CE (+ KD if enabled). Config flags default off; with them off the policy is byte-identical to the template.
2. **Feedback variant.** `IL.reg_feedback True`: the head's predicted progress and the soft route-node distribution are projected to one extra token appended to the map tokens (same width), so the policy can use its own registration estimate. Off by default.
3. **Launcher** `scripts/distill/run_dagger_g1.sh ARM` with arms
   - `corridor_noreg` — `gt.graph.r1p5.direction5.v1`, COORD=True, head off (should reproduce ≈ 86; the control),
   - `corridor_reg` — same + head (supervision only),
   - `corridor_reg_fb` — same + head + feedback token,
   - `box_noreg`, `box_reg`, `box_reg_fb` — the same three on `gt.graph.box.direction5.v1`.
   Same iters / seed / GPUs as the template; `eval_args.txt` written so `eval_iters.sh` works unchanged; one sbatch wrapper for ln01 in `scripts/submit/`. Dry-run all six.
4. **Training-free probe (optional, cheap):** evaluate the existing 86-SR checkpoint on `gt.graph.box.direction5.v1` val_unseen as-is. It is out-of-distribution for that model, so read it only as a lower bound on what the box raster carries.

Tests: head shapes / masking / `-1` ignore; feedback token width; `reg_weight=0` and head off reproduce the template forward bit-for-bit on a fixed input; serializer round-trip; `graph_reward` on a GT text equals its maximum.

## Reporting

Commit per part; `docs/daily/` (Chinese) with the token statistics, drop rate, box-raster IoU, the located template run (path, iter, SR), the six dry-run command lines, and anything in the template recipe that had to change. Final message: those numbers plus the list of launch commands for the user.
