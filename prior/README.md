# Helpers

Following modules could be run with `python -m`:

- `prior`: Generate cognitive maps for all VLNCE entries (R2R + RxR)
- `prior.bbox`: Show / export bounding box info for given scene or episode
- `prior.etp_r1`: Generate cognitive maps for all ETP-R1 entries
- `prior.grid_map`: Visualizes given cognitive map (`.npz`)

For analyzing, see `./analyze/README.md`.

# Cognitive Map Generators

Both `prior` and `prior.etp_r1` generate paired cache files:

```text
<output-dir>/<namespace>/boxes/<scene-id>/<episode-or-instr-id>.npz
<output-dir>/<namespace>/raster/<scene-id>/<episode-or-instr-id>.npz
```

The `boxes/` file stores the selected relevant semantic boxes sidecar. The
`raster/` file stores the cognitive map consumed by PriorGT-style navigation
models.

Shared generator options:

- `--map-source {bbox,legacy}`: `bbox` rasterizes relevant semantic boxes;
  `legacy` builds a Try5-style path-neighborhood raster from the selected
  full-level semantic grid.
- `--radius-m FLOAT`: Path-neighborhood radius in meters. Default is `1.5`.
- `--namespace NAME`: Cache namespace under `--output-dir`. If omitted, the
  namespace is derived as `<map-source>_r<radius>`, such as `bbox_r1p5` or
  `legacy_r2p5`.
- `--output-dir PATH`: Cache root. Defaults to `data/cognitive_maps` for
  VLN-CE and `data/cognitive_maps_etp_r1` for ETP-R1.

Examples:

```bash
python -m prior --map-source legacy --radius-m 1.5
python -m prior --map-source legacy --radius-m 2.5
python -m prior --map-source bbox --radius-m 2.5
python -m prior.etp_r1 --map-source legacy --radius-m 1.5
python -m prior.etp_r1 --map-source bbox --radius-m 2.5 42_0
```

The optional positional arguments to `prior.etp_r1` are sampled instruction IDs.
When provided, only those instructions are regenerated and visualized.

Level selection is shared by `bbox` and `legacy`: the first semantic level
touched by the ground-truth trajectory is selected. If that level has fewer
than two selected-level points, generation raises
`InsufficientTrajectoryPointsError` and skips the episode instead of falling
back to a later level.

# API

- `SceneSemanticBoxes.from_scene_id` -> `SceneSemanticBoxes` (collection of `LevelSemanticBoxes`)
- `SceneSemanticBoxes` -`relevant_to`-> `RelevantSemanticBoxes`
- `RelevantSemanticBoxes` -`to_cognitive_map`-> `CognitiveGridMap`

# Saved Raster Data

Raster cognitive maps are saved as compressed `.npz` files under `raster/`.

## Raster NPZ Overview

Each raster npz file contains:

- `grid`: Grid data of dimension (OBJECT_CATEGORIES + REGION_CATEGORIES) x ROWS x COLS.
- `range_y`: Y range of the floor, stored as `[min_y, max_y]`; either value may be `null`. Not useful for our job.
- `trajectory_keypoints`: Five level-local trajectory keypoints as `[x, z]`,
  zero-padded when needed.
- `start_direction_vector`: Direction vector of the start position.

Together they showcase navigation trajectory and semantic surroundings for a
navigation instruction. The covered path neighborhood is controlled by
`--radius-m`.

## Grid

- Each cell is 0.5 x 0.5m. ROWS and COLS set to $100$. OBJECT_CATEGORIES = 27, REGION_CATEGORIES = 10.
- Each element in cell is `float` in the range [0.0, 1.0], representing the confidence level of the presence of a specific category at the grid cell.
    - For generated ground-truth, if the category is mentioned in instruction, the confidence level is set as $1$; otherwise $0.6$.
- In visualizations, the upper-right corner is the origin (row=0, col=0). Column coord increases going left, and row coord increases going down.

## Angles

Angles follow standard mathematical convention in visualization space, increasing as you go counter-clockwise. Examples:

| Angle | Direction Vector | Visualized Direction | Grid direction |
| - | - | - | - |
| 0° | (cos=1, sin=0) | Right | -col |
| 90° | (cos=0, sin=1) | Up | -row |

# Saved Relevant-Box Data

Relevant semantic boxes can be saved as `.json`. Generator sidecars are saved as
compressed `.npz` files under `boxes/`, with the same serialized payload.

## JSON Overview

Each json file contains:

- `level_idx`: First ground-truth-trajectory level touched by the episode.
- `level`: Relevant boxes on that selected level.
- `instruction`: Episode instruction text.
- `ground_truth_trajectory`: Dense selected-level positions as `[x, z]`.
- `trajectory_keypoints`: Five selected-level keypoints as `[x, z]`, zero-padded.
- `start_direction_vector`: Direction vector of the start position, in the format of (cos, sin).

The nested `level` object contains:

- `objects`: 27 arrays of object boxes, indexed by mapped object category.
- `regions`: 10 arrays of region AABBs, indexed by mapped region category.
- `range_y`: Y range of the floor, stored as `[min_y, max_y]`; either value may be `null`. When element `null`, indicates no bound. Not useful for our job.

Together they showcase navigation trajectory and semantic surroundings for a
navigation instruction, covering a local neighborhood around each waypoint.
Every position in each bounding box should be positive.

## Objects

The `objects` field is an array of length 27. `objects[i]` contains 2D boxes of objects of category $i$. Each 2D object box consists of:

- `center`: Center coordinate of the box.
- `half_extents`: Half size along X/Z.
- `rotation`: Box rotation in radians.
- `mentioned`: Whether category $i$ is mentioned in the instruction.

## Regions

The `regions` field is an array of length 10. `regions[i]` contains 2D axis-aligned bounding boxes of regions of category $i$. Each 2D axis-aligned bounding box consists of:

- `min`: Minimum X/Z coordinate of the box.
- `max`: Maximum X/Z coordinate of the box.
- `mentioned`: Whether category $i$ is mentioned in the instruction.

## Angles

Angles follow the reverse of standard mathematical convention, but still increasing as you go counter-clockwise. Examples:

| Angle | Direction Vector | Coordinate direction |
| - | - | - |
| 0° | (cos=1, sin=0) | -Z |
| 90° | (cos=0, sin=1) | -X |

# Categories

## Object Categories

```python
MAPPED_OBJECT_NAMES = [
    "void",  # 0
    "chair",  # 1
    "door",  # 2
    "table",  # 3
    "cushion",  # 4
    "sofa",  # 5
    "bed",  # 6
    "plant",  # 7
    "sink",  # 8
    "toilet",  # 9
    "tv_monitor",  # 10
    "shower",  # 11
    "bathtub",  # 12
    "counter",  # 13
    "appliances",  # 14
    "structure",  # 15
    "other",  # 16
    "free-space",  # 17
    "picture",  # 18
    "cabinet",  # 19
    "chest_of_drawers",  # 20
    "stool",  # 21
    "towel",  # 22
    "fireplace",  # 23
    "gym_equipment",  # 24
    "seating",  # 25
    "clothes",  # 26
]
"""Names of the 27 mapped object categories, indexed by mapped category ID."""
```

## Region Categories

```python
MAPPED_REGION_NAMES = [
    "outdoor/semi-outdoor",  # 0
    "living/social space",  # 1
    "recreation/fitness",  # 2
    "utility/service",  # 3
    "work/study",  # 4
    "circulation",  # 5
    "private room",  # 6
    "bathroom/sanitary",  # 7
    "dining/food",  # 8
    "other/miscellaneous",  # 9
]
"""Names of the 10 mapped region categories, indexed by mapped category ID."""
```
