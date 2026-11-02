# Cognitive graph V2 interfaces

These CPU-generated R2R artifacts share one source: the dense VLN-CE ground-truth
trajectory and cached MP3D semantic boxes. The implementation lives in
`prior/cogmap_v2/`. All outputs are under a real, repository-local `data_v2/`;
V1 datasets, semantic caches, and cognitive maps are read-only inputs.

## Artifact locations and identifiers

| Artifact | Path relative to repository root |
| --- | --- |
| Graph | `data_v2/cogmap_v2/graphs/<split>/<scene>/<episode_id>.json` |
| Graph distribution and rules | `data_v2/cogmap_v2/graphs/<split>/stats.json` |
| Insufficient selected-level paths | `data_v2/cogmap_v2/graphs/<split>/skipped.json` |
| Train distance calibration | `data_v2/cogmap_v2/graphs/train/distance_bins.json` |
| Registration | `data_v2/cogmap_v2/registration/<split>/<scene>/<episode_id>.npz` |
| V1 boxes payload | `data_v2/cognitive_maps/gt.graph.r1p5.direction5.v1/boxes/<scene>/R2R_<split>_<episode_id>.npz` |
| V1 raster payload | `data_v2/cognitive_maps/gt.graph.r1p5.direction5.v1/raster/<scene>/R2R_<split>_<episode_id>.npz` |

Graph and registration filenames use numeric episode IDs within a split. Raster
and box filenames use `VLNCEEpisodeEntry.unique_id`, matching V1 consumers and
preventing collisions across splits. Nodes use episode-local integer IDs; route
IDs are ordered, unique, and separate repeated visits to the same region.
`gt.region_id` identifies a region instance as `category_index:box_index` within
the selected scene level. It is diagnostic supervision, not a model input.

## Graph construction and GT boundary

`CognitiveGraph.load(path)` validates the Pydantic schema. Each node has the
category, role, anchor kind, order, mention flag, and attached object categories
and relative offset bins specified in the task brief. Edges carry distance,
passage, turn, and vertical bins. `route` contains only route nodes; side nodes
have `order=-1` and attach to their closest route sub-segment by an edge.

The smallest containing region AABB owns a path point. Points outside all
regions retain the previous region; an initial uncovered segment is `unknown`.
Region-entry triggers survive even when less than 1 m apart so short visits
remain visible. Other turn and fork triggers merge into the preceding anchor
within 1 m. Start and final path point are always separate anchors, and the
final anchor has role `goal`. Fork detection follows the existing
`n_forks3_graph` rule: snap within 1.5 m, exclude the previous connectivity node,
keep onward edges within 3.5 m, and require three bearing clusters separated
by more than 60 degrees. Turns use cosine similarity below 0.8, without a cap.

Mention extraction reuses ADR 0002's canonical extractor, including its noun
resolution. Unmentioned entered regions and nearby objects remain first-class
entries. Mentioned unentered region instances within 1.5 m become side nodes.
Objects within 1.5 m of dense selected-level samples attach once to the nearest
route sub-segment; this is a sampled approximation to continuous distance.
Door passage tests use the crossing sample and a 1 m OBB distance threshold.
Vertical bins use anchor height differences above 0.5 m. Inter-anchor distance
uses selected-level polyline arc length; train route edges alone determine the
two terciles in `constants.py`. Side-edge distance uses the side center's
Euclidean distance to the route attachment point and the same bin thresholds.

Every key named `gt` is supervision only, at every nesting level:

- `node.gt`: actual path anchor center, region AABB, inclusive selected-level
  `path_index_range`, and region instance ID.
- `object.gt`: metric object OBB.
- `graph.gt`: dense selected-level path, its original world point indices,
  full world trajectory, selected-level origin and height range, and V1
  direction5 metadata computed from the full world trajectory.

The selected-level path may omit points from other levels. Consecutive retained
points are connected in X/Z for this initial single-level representation;
stairs are edge annotations, not a complete multi-level graph.

## M1: compact graph text for the LLM

M1 should serialize only semantic node fields, edges, and the route in a
deterministic ID/order scheme. Keep unmentioned route regions and side nodes.
Do not recursively serialize a graph object and merely remove the top-level
`gt`: node and object GT must also be removed. Do not include scene identity,
region instance IDs, dense paths, or metric boxes in prompts or graph targets.
Instruction text is the prompt; start pose can be supplied as explicit known
metadata. Categories are canonical strings from `prior.constants`.

For generated graphs, derive `mentioned` from the instruction using the same
extractor, as ADR 0002 requires. The stored GT flags are training diagnostics,
not a requirement to predict flags. Unknown regions have no GT region box and
can be represented as `unknown`; do not invent a canonical region index for
them. A compact serializer and LLM training are future M1 work.

## M5: metric layout targets and rendering

`graph_to_layout(graph) -> GraphLayout` builds the supervised metric target.
`render_gt(graph) -> RenderedMap` uses that target and the same renderer as
`render_from_layout(graph, layout) -> RenderedMap`. Both optionally accept a
raster output path. `.save(path)` writes the NPZ; `.save_boxes(path,
instruction, level_idx)` writes a V1 `RelevantSemanticBoxes` payload.

`GraphLayout` contains:

- `nodes`: exactly one `NodeLayout(node_id, region, objects)` per graph node.
  Objects align by list index with `graph.nodes[].objects`. `region` may be
  absent. Every `LayoutBox` has `center`, `half_extents`, and `rotation`.
- `start_position_level_xz` and unit `start_direction_vector`: known output
  frame pose, supplied explicitly rather than read from GT.
- `direction_vectors_start`: exactly five unit `(right, forward)` vectors,
  followed by zero padding. These must be predicted or otherwise supplied
  without GT for a deployable layout.
- Optional `range_y`, default `(None, None)` for predicted layouts.
- Optional `route_points_start` and `route_radius_m` (default 1.5 m): explicitly
  supplied route samples used for V1 square-neighborhood raster clipping.
  `graph_to_layout` supplies the selected-level dense GT path. M5/M3 must supply
  predicted or independently estimated samples to obtain the same clipping at
  deployment; omitting the samples renders whole boxes. Sparse samples are not
  automatically interpolated, so sample the predicted polyline densely enough
  to avoid gaps in the neighborhood mask.

Positions and extents are meters. In the start frame, +x is right and +z is
forward. Given V1 start direction `(s,c)`, level-local forward is `(-c,-s)`
and right is `(s,-c)`. These basis columns transform start coordinates back
to level coordinates after adding the level-local start position. Box rotation
is the angle of its local x axis, `atan2(z,x)` in its own coordinate frame.
Regions support oriented rectangles because a level AABB becomes an OBB under
an arbitrary start-frame rotation. This task's explicit start-frame M5 target
extends the level-local output choice in ADR 0001; V1 output remains level-local.

`render_from_layout` reads no GT, including node/object GT. It rasterizes the
provided geometry with V1 category channels and mention confidence (1.0 or 0.6).
Repeated region geometry is deduplicated. Its output is float32
`grid[37,100,100]`, float32 `direction_vectors[5,2]`, float32
`start_direction_vector[2]`, float32 `start_position[2]`, and V1 `range_y`.
Raster rows increase with level x; columns increase with level z.

V1 boxes support region AABBs only: a rotated predicted region is saved as its
enclosing level AABB in `boxes/`, while `raster/` retains the oriented footprint.
Treat the raster as authoritative for rotated M5 predictions. Saved boxes carry
no dense trajectory and only a start keypoint plus padding; they cannot recover
the direction5 trajectory metadata by themselves. Use `GraphLayout` as M5's
lossless metric target rather than the V1 boxes compatibility artifact.

Graph GT rendering fills represented boxes and applies the same V1 square
neighborhood around each selected-level dense path sample. Nearby unentered,
unmentioned regions are absent from the graph, and Euclidean object selection
differs from square raster neighborhoods. Therefore graph and legacy raster IoU
is measured, not assumed to be one. The renderer
report records per-episode occupied category-channel IoU, confidence-weighted
IoU, distribution summaries, and exact equality of all four metadata arrays.

## M2: online registration during DAgger

`build_registration(graph) -> RegistrationLabels` creates a self-contained,
pickle-free NPZ (`schema_version=1`). Load with `RegistrationLabels.load(path)`.

| Arrays | Index convention |
| --- | --- |
| `path_node_id`, `progress`, `path_xz`, `selected_level_mask` | Full original dense GT trajectory |
| `route_xz` | Dense selected-level route |
| `selected_world_indices` | Selected-level index to full dense index |
| `anchor_xz`, `route_node_id`, `node_start_index` | `graph.route` order |

Node IDs and indices are int64, geometry/progress float64, and the mask is bool.
`progress = zero_based_route_order / n_route`; the goal is `(n_route-1)/n_route`.
Selected-level point labels use the graph's inclusive range partition. Other
level points project in X/Z onto the retained route, using temporal position
only to break equal-distance ties. The mask allows M2 to omit those projected
labels if its rollout supervision requires actual level membership.

`labels.label_position(level_xz, path_index_hint=None)` returns the node at the
projection onto dense route segments. It does not choose the nearest anchor.
`locate_position` additionally returns progress, fractional selected-level path
index, and projection distance in meters. Convert a world X/Z query by
subtracting `graph.gt.level_origin`; M2 needs that known scene coordinate
transform or an equivalent deployment frame transform.

At exact crossings/repeated positions, X/Z alone cannot identify route
occurrence. Without a hint lookup chooses the first occurrence. Pass the last
returned selected-level path index for continuity; ties within 1e-7 m choose
the occurrence closest to that hint. A hint never overrides a spatially closer
branch. The dense-point round-trip audit passes the known selected-level index;
M2 should track history and inspect `distance_m` for off-route observations.

## CPU generation and checks

Run with the repository's Python environment containing Habitat, Pydantic v2,
typed-argument-parser, and existing mention-extraction dependencies. Prefix
commands with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
python -m prior.cogmap_v2.graph --split train --calibrate-distance-bins
python -m prior.cogmap_v2.graph --split val_seen
python -m prior.cogmap_v2.graph --split val_unseen
python -m prior.cogmap_v2.render --split val_unseen --output-dir data_v2/cognitive_maps
python -m prior.cogmap_v2.registration --split val_unseen
python scripts/cogmap_v2/check_graph_vs_features.py
python scripts/cogmap_v2/audit_registration.py --split val_unseen
```

Repeat rendering/registration for the other splits. Calibration writes its
measured thresholds and source counts to `train/distance_bins.json`; update
`constants.py` to those thresholds before generating validation splits. Generated
data is intentionally Git-ignored; committed daily notes preserve run counts,
thresholds, checks, and limitations. `check_graph_vs_features.py` compares region
instances rather than category count, flags absolute count differences of at
least two, and lists feature rows without graphs (including skipped episodes).
