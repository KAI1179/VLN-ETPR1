## Data Overview

Each npz file contains:

- `grid`: Grid data of dimension (OBJECT_CATEGORIES + REGION_CATEGORIES) x ROWS x COLS.
- `offset_x`: X offset to transform world coordinates to grid coordinates, in order to keep indexing positive. Not useful for our job.
- `offset_z`: Z offset to transform world coordinates to grid coordinates, in order to keep indexing positive. Not useful for our job.
- `range_y`: Y range of the floor. Not useful for our job.
- `positions`: Waypoint positions along the path.
- `direction_vectors`: 5 distinct path directions as normalized (cos, sin) tuples.
    - Zeroes used if distinct directions < 5.
    - At most 5 directions are stored. Extra directions are discarded.
- `start_direction_vector`: Direction vector of the start position.

Together they showcase navigation trajectory and semantic surroundings for a navigation instruction, covering a local neighborhood of radius 5 cells around each waypoint.

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

## Categories

Object and region categories:

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
