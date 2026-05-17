"""Constants used in prior-based mapping and navigation."""

from magnum import Quaternion, Vector4
from habitat_sim.geo import FRONT, GRAVITY
from typing import cast

# OBJECT MAPPING

OBJECT_MAPPING = [
    0,  # void -> void
    15,  # wall -> structure
    17,  # floor -> free-space
    1,  # chair -> chair
    2,  # door -> door
    3,  # table -> table
    18,  # picture -> picture
    19,  # cabinet -> cabinet
    4,  # cushion -> cushion
    15,  # window -> structure
    5,  # sofa -> sofa
    6,  # bed -> bed
    16,  # curtain -> other
    20,  # chest_of_drawers -> chest_of_drawers
    7,  # plant -> plant
    8,  # sink -> sink
    17,  # stairs -> free-space
    17,  # ceiling -> free-space
    9,  # toilet -> toilet
    21,  # stool -> stool
    22,  # towel -> towel
    16,  # mirror -> other
    10,  # tv_monitor -> tv_monitor
    11,  # shower -> shower
    15,  # column -> structure
    12,  # bathtub -> bathtub
    13,  # counter -> counter
    23,  # fireplace -> fireplace
    16,  # lighting -> other
    16,  # beam -> other
    16,  # railing -> other
    16,  # shelving -> other
    16,  # blinds -> other
    24,  # gym_equipment -> gym_equipment
    25,  # seating -> seating
    16,  # board_panel -> other
    16,  # furniture -> other
    14,  # appliances -> appliances
    26,  # clothes -> clothes
    16,  # objects -> other
    16,  # misc -> other
]
"""
Mapping (id-to-id) for object categories, from 1 (void) + 40 classes to 27 classes, following prior works:
- [CM2](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/datasets/util/viz_utils.py#L57-L59)
- [WS-MGMap](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/habitat_extensions/sensors.py#L324-L328)

Note: Categories 0 (void), 15 (structure), and 17 (free-space) represent environmental elements
rather than discrete objects and are typically excluded from object-level visualizations.
"""

OBJECT_NAMES = [
    "void",  # -> void
    "wall",  # -> structure
    "floor",  # -> free-space
    "chair",  # -> chair
    "door",  # -> door
    "table",  # -> table
    "picture",  # -> picture
    "cabinet",  # -> cabinet
    "cushion",  # -> cushion
    "window",  # -> structure
    "sofa",  # -> sofa
    "bed",  # -> bed
    "curtain",  # -> other
    "chest_of_drawers",  # -> chest_of_drawers
    "plant",  # -> plant
    "sink",  # -> sink
    "stairs",  # -> free-space
    "ceiling",  # -> free-space
    "toilet",  # -> toilet
    "stool",  # -> stool
    "towel",  # -> towel
    "mirror",  # -> other
    "tv_monitor",  # -> tv_monitor
    "shower",  # -> shower
    "column",  # -> structure
    "bathtub",  # -> bathtub
    "counter",  # -> counter
    "fireplace",  # -> fireplace
    "lighting",  # -> other
    "beam",  # -> other
    "railing",  # -> other
    "shelving",  # -> other
    "blinds",  # -> other
    "gym_equipment",  # -> gym_equipment
    "seating",  # -> seating
    "board_panel",  # -> other
    "furniture",  # -> other
    "appliances",  # -> appliances
    "clothes",  # -> clothes
    "objects",  # -> other
    "misc",  # -> other
]
"""Names of the 41 original object categories, indexed by category ID."""

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

MAPPED_OBJECT_OTHER_INDEX = MAPPED_OBJECT_NAMES.index("other")
"""Index of the "other" category in the mapped object names."""

OBJECT_CATEGORIES = len(set(OBJECT_MAPPING))
"""Number of unique object categories after mapping."""

ENVIRONMENTAL_OBJECT_CATEGORIES = {0, 15, 17}
"""Set of mapped object category IDs that represent environmental elements rather than discrete objects."""

MAPPED_OBJECT_COLORS = [
    "#1a1a1a",  # 0: void - black
    "#654321",  # 1: chair - dark brown
    "#B8860B",  # 2: door - dark goldenrod
    "#8B4513",  # 3: table - saddle brown
    "#F5DEB3",  # 4: cushion - wheat
    "#CC6633",  # 5: sofa - terra cotta
    "#4682B4",  # 6: bed - steel blue
    "#228B22",  # 7: plant - forest green
    "#87CEEB",  # 8: sink - sky blue
    "#F0F0F0",  # 9: toilet - very light gray
    "#2F4F4F",  # 10: tv_monitor - dark slate gray
    "#B0E0E6",  # 11: shower - powder blue
    "#E0FFFF",  # 12: bathtub - light cyan
    "#DEB887",  # 13: counter - burly wood
    "#708090",  # 14: appliances - slate gray
    "#A9A9A9",  # 15: structure - dark gray
    "#D3D3D3",  # 16: other - light gray
    "#F5F5F5",  # 17: free-space - white smoke
    "#FFD700",  # 18: picture - gold
    "#704214",  # 19: cabinet - sepia
    "#BC8F8F",  # 20: chest_of_drawers - rosy brown
    "#D2B48C",  # 21: stool - tan
    "#FFB6C1",  # 22: towel - light pink
    "#FF6347",  # 23: fireplace - tomato
    "#4B0082",  # 24: gym_equipment - indigo
    "#DAA520",  # 25: seating - goldenrod
    "#9370DB",  # 26: clothes - medium purple
]
"""RGB hex colors for the 27 mapped object categories, designed to be intuitive and distinctive."""

# REGION MAPPING

REGION_MAPPING = [
    8,  # bar -> dining/food
    4,  # classroom -> work/study
    8,  # dining booth -> dining/food
    7,  # spa/sauna -> bathroom/sanitary
    9,  # junk -> other/miscellaneous
    7,  # bathroom -> bathroom/sanitary
    6,  # bedroom -> private room
    6,  # closet -> private room
    8,  # dining room -> dining/food
    5,  # entryway/foyer/lobby -> circulation
    1,  # familyroom/lounge -> living/social space
    3,  # garage -> utility/service
    5,  # hallway -> circulation
    4,  # library -> work/study
    3,  # laundryroom/mudroom -> utility/service
    8,  # kitchen -> dining/food
    1,  # living room -> living/social space
    4,  # meetingroom/conferenceroom -> work/study
    1,  # lounge -> living/social space
    4,  # office -> work/study
    0,  # porch/terrace/deck -> outdoor/semi-outdoor
    2,  # rec/game -> recreation/fitness
    5,  # stairs -> circulation
    7,  # toilet -> bathroom/sanitary
    3,  # utilityroom/toolroom -> utility/service
    2,  # tv -> recreation/fitness
    2,  # workout/gym/exercise -> recreation/fitness
    0,  # outdoor -> outdoor/semi-outdoor
    0,  # balcony -> outdoor/semi-outdoor
    9,  # other room -> other/miscellaneous
]
"""
Mapping (id-to-id) for region categories, from 30 classes to 10 classes.

Usage:
```python
category_id = region.category.index()
mapped_id = REGION_MAPPING[category_id]
mapped_name = MAPPED_REGION_NAMES[mapped_id]
```
"""

REGION_NAMES = [
    "bar",  # -> dining/food
    "classroom",  # -> work/study
    "dining booth",  # -> dining/food
    "spa/sauna",  # -> bathroom/sanitary
    "junk",  # -> other/miscellaneous
    "bathroom",  # -> bathroom/sanitary
    "bedroom",  # -> private room
    "closet",  # -> private room
    "dining room",  # -> dining/food
    "entryway/foyer/lobby",  # -> circulation
    "familyroom/lounge",  # -> living/social space
    "garage",  # -> utility/service
    "hallway",  # -> circulation
    "library",  # -> work/study
    "laundryroom/mudroom",  # -> utility/service
    "kitchen",  # -> dining/food
    "living room",  # -> living/social space
    "meetingroom/conferenceroom",  # -> work/study
    "lounge",  # -> living/social space
    "office",  # -> work/study
    "porch/terrace/deck",  # -> outdoor/semi-outdoor
    "rec/game",  # -> recreation/fitness
    "stairs",  # -> circulation
    "toilet",  # -> bathroom/sanitary
    "utilityroom/toolroom",  # -> utility/service
    "tv",  # -> recreation/fitness
    "workout/gym/exercise",  # -> recreation/fitness
    "outdoor",  # -> outdoor/semi-outdoor
    "balcony",  # -> outdoor/semi-outdoor
    "other room",  # -> other/miscellaneous
]
"""Names of the 30 original region categories, indexed by category ID."""

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

REGION_CATEGORIES = len(MAPPED_REGION_NAMES)
"""Number of unique region categories after mapping."""

MAPPED_REGION_COLORS = [
    "#90EE90",  # 0: outdoor/semi-outdoor - light green
    "#FFA07A",  # 1: living/social space - light salmon
    "#FF6B6B",  # 2: recreation/fitness - coral red
    "#808080",  # 3: utility/service - gray
    "#6495ED",  # 4: work/study - cornflower blue
    "#E6E6E6",  # 5: circulation - very light gray
    "#DDA0DD",  # 6: private room - plum
    "#ADD8E6",  # 7: bathroom/sanitary - light blue
    "#FFD966",  # 8: dining/food - light orange/yellow
    "#D3D3D3",  # 9: other/miscellaneous - light gray
]
"""RGB hex colors for the 10 mapped region categories, designed to be intuitive and distinctive."""

# GRID MAP PARAMETERS

WIDTH = 50
"""Width of the grid map in meters."""

DEPTH = 50
"""Depth of the grid map in meters."""

CELL_SIZE = 0.5
"""Length of each grid cell in meters."""

ROWS = int(WIDTH / CELL_SIZE)
"""Number of rows in the grid map."""

COLS = int(DEPTH / CELL_SIZE)
"""Number of columns in the grid map."""

# COGNITIVE MAP PARAMETERS

MAX_DISTANCE_CELLS = 5
"""Maximum distance (in cells) to consider as related to the position/path."""

GAUSSIAN_SIGMA = 1
"""Sigma parameter for gaussian_filter."""

DIRECTION_VECTOR_CNT = 5
"""Number of direction vectors."""

DIRECTION_VECTOR_SIM = 0.8
"""Threshold of considering two direction vectors the same."""

# INTERNAL UTILITY CONSTANTS

_HABITAT_MP3D_ROTATION_QUATERNION = Quaternion.rotation(FRONT, GRAVITY)  # type: ignore
HABITAT_MP3D_ROTATION_VECTOR = cast(Vector4, _HABITAT_MP3D_ROTATION_QUATERNION.xyzw)  # type: ignore
"""Rotation vector used for MP3D scenes. Translated from [CPP source](https://github.com/facebookresearch/habitat-sim/blob/6c26c4f9ae10ad7f534aa9e59a84dff00782741e/src/esp/scene/SemanticScene.h#L157-L161), which evaluates to `Vector(-0.707107, 0, 0, 0.707107)`."""
