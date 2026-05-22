from sys import argv
from . import construct_bounding_boxes_from_scene_id
from ..constants import MAPPED_REGION_NAMES


scenes = argv[1:]
for scene in scenes:
    print(f"Scene {scene}")
    levels = construct_bounding_boxes_from_scene_id(scene)
    for level_idx, level in enumerate(levels):
        print(f"  Level {level_idx} ({level.range_y})")
        for region_cat, regions in enumerate(level.regions):
            if len(regions) == 0:
                continue
            region_name = MAPPED_REGION_NAMES[region_cat]
            print(f"    Region {region_name}")
            for region in regions:
                print(f"      {region.id} {region.min} ~ {region.max}")
