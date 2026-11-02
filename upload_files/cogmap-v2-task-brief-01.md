# Task brief 01 — GT cognitive graph, graph→raster renderer, registration labels

Workspace: `/home/xukai/code/cogmap-VLN-V2` — a full copy of the V1 checkout (`KAI1179/VLN-ETPR1`, with its uncommitted patch state and 205k symlinks preserved; many symlinks and some scripts still point into the old repo, which stays untouched). Work only inside this copy. Repo conventions in `AGENTS.md` apply (tap.Tap CLIs, ruff/ty, English in code, Chinese in `docs/daily/`). CPU-only work.

Setup before coding:
1. `git status`; commit the inherited uncommitted state as one `wip: snapshot of V1 workspace` commit (or stash it) so later diffs are clean, then `git checkout -b exp/cogmap-v2-graph-gt` from the current HEAD (not from `main`: the current tree carries the coordinate-fusion patches that G1 will need).
2. `data/`, `pretrained/` etc. are symlinks into the old repo — read from them freely (scene boxes, GT caches, connectivity), but **write every V2 output into a real directory in this copy**: `/home/xukai/code/cogmap-VLN-V2/data_v2/` (graphs, registration labels, and the new raster namespace via `--output-dir data_v2/cognitive_maps`). Nothing new is written through the symlinks.
3. Any new script resolves paths from the repo root at runtime (as the later launchers in `scripts/distill/` already do), never hard-codes `/home/xukai/...`.
4. `features_val_unseen.json` lives on branch `data/pa-dumps`: `git show origin/data/pa-dumps:features_val_unseen.json > data_v2/features_val_unseen.json`.

## Why

V2 upgrades the cognitive-map representation from "LLM-predicted raster" to "cognitive graph G → metric layout M", adds a registration head that places the agent's topological nodes on G, and maintains G/M online. All of these are trained from one GT source, which does not exist yet: the GT cognitive graph per episode, a renderer from graph to the V1 raster format, and per-point registration labels. Build these three.

## Facts to design against

- val_unseen: GT path crosses median 3 regions (30% ≤ 2), median 1 turn, median 2 three-way forks; 79.7% of episodes mention fewer rooms than the path crosses (`data_v2/features_val_unseen.json`, from branch `data/pa-dumps`, from `scripts/analysis/episode_features.py` on `claude/awesome-ride-axuad6`, patches 0066/0078). So nodes are region × (turn | fork) sub-segments (median ≈ 6 anchors), and unmentioned regions are first-class nodes.
- The consumer stays V1 code: Try5 raster cache (`raster/<scene>/<episode>.npz`: `grid` 37×100×100, `direction_vectors`, `start_direction_vector`, `start_position`), as written by `prior.cognitive_map_generation` / `prior.etp_r1`. Render into exactly that format so V1 encoders, coordinate fusion (patch 0025 on `exp/refiner`), KD and the eval scripts consume V2 data unchanged.
- Reuse: `prior.bbox.SceneSemanticBoxes.from_scene_id` (already cached), `prior.bbox._relevance._first_touched_level_points` (level pick + level-local xz path), `prior.cognitive_map_generation.trajectory_direction_vectors` (turn rule, cos < 0.8), `prior.vlnce.VLNCEEpisodeEntry`, `prior.etp_r1.ConnectivityEntry.map_for` (cached), the ADR 0002 mention extractor, and the fork rule in `episode_features.py` (`n_forks3_graph`).

## Deliverable 1 — `prior/cogmap_v2/graph.py`, CLI `python -m prior.cogmap_v2.graph --split ...`

`CognitiveGraph` (pydantic) → `data_v2/cogmap_v2/graphs/<split>/<scene>/<episode_id>.json`, plus `stats.json` per split.

```
nodes: id, region_category, role {start|on_route|side|goal}, anchor_kind {region_entry|turn|fork}, order,
       mentioned (region category appears in instruction),
       objects: [{category, mentioned, offset_bin {left|right|ahead|behind}}],
       gt: {center_level_xz, aabb_level, path_index_range}        # GT-only, never shown to the LLM
edges: src, dst, distance_bin {short|medium|long}, passage {door|opening|stairs|unknown},
       turn {straight|left|right|back}, vertical {none|up|down}
route: ordered node ids
meta: scene, episode_id, instruction, level_idx, start_position_level_xz, start_direction_vector
```

Construction, as a starting point — adjust where the data says otherwise and note it:
- Walk the dense GT path on the selected level; a new `region_entry` node when the path enters another region AABB (overlaps → smallest area; points outside any region stay with the current node). Split within a region at turns (`trajectory_direction_vectors` rule, no 5-cap) and at forks; merge triggers closer than ~1 m.
- `side` nodes: mentioned region categories whose AABB comes within ~1.5 m of the path but is never entered.
- Objects: OBBs within ~1.5 m of the sub-segment, attached to the nearest node; `offset_bin` from the path heading there.
- `distance_bin` from train-split terciles of inter-anchor path length (store the thresholds in `constants.py`). `passage`: stairs if |Δy| > 0.5 m, door if a door OBB sits within ~1 m of a region-boundary crossing, opening for other crossings, else unknown. `turn` from heading change (±30° straight, beyond ±150° back).
- Episodes that raise `InsufficientTrajectoryPointsError` go to `skipped.json`.

Sanity check to include: `scripts/cogmap_v2/check_graph_vs_features.py` comparing the graph's region count with `n_rooms` from `features_val_unseen.json`; large disagreements mean the containment rule needs a look. A unit test on a small hand-built scene (two regions, a door, a turn) is enough.

## Deliverable 2 — `prior/cogmap_v2/render.py`

- `render_gt(graph) → npz`: from the `gt` fields; should land close to the existing `gt.legacy.r1p5.direction5.v1` raster for the same episode (report the per-episode IoU on a val_unseen sample; metadata should match).
- `render_from_layout(graph, layout) → npz`: `layout` = per node an AABB (region) / OBB (object) in the start frame — the M5 output interface, also what M3 re-renders through. Same writer, no GT access.
- New namespace `gt.graph.r1p5.direction5.v1` with the usual `boxes/` + `raster/` layout.

## Deliverable 3 — `prior/cogmap_v2/registration.py`

Per episode `data_v2/cogmap_v2/registration/<split>/<scene>/<episode_id>.npz`: `path_node_id[i]` for each dense GT point, `anchor_xz[k]`, `progress[i] = order/n_route`, and a helper `label_position(xz) → node_id` that projects onto the route polyline (so online topological nodes get labelled by route position, not Euclidean nearest). Round-trip test on GT points.

## Also write

`docs/cogmap_v2/INTERFACES.md`: what M1 (compact text serialisation of G for the LLM), M5 (layout target) and M2 (label lookup during DAgger rollout) will need from these files, as you see it after building them.

Commit per deliverable; record numbers and any rule changes in `docs/daily/` (Chinese). Final message: node-count histogram on val_unseen, distance-bin thresholds, render IoU summary, skipped episodes.
