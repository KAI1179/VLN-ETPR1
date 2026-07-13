from pathlib import Path
from prior.grid_map import BaseGridMap

SPLIT = "val_unseen"
llm_base = (
    Path("data/llm_navigation/llm-grid-r2r-legacy-r1p5-direction5-scale2/r2r/")
    / SPLIT
    / "cognitive_maps/raster/"
)
gt_base = Path("data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1/raster/")
llm_ns = "llm-grid-s2.legacy.r1p5.direction5"
gt_ns = "gt.legacy.r1p5.direction5.blurred"

samples = set()

for scene in llm_base.iterdir():
    for llm_npz in scene.iterdir():
        samples.add((scene.stem, llm_npz.stem))
        print(scene.stem, llm_npz.stem)
        llm_map = BaseGridMap.load(llm_npz)
        save_d = Path(f"data/samples/{llm_ns}/{scene.stem}")
        save_d.mkdir(parents=True, exist_ok=True)
        save_p = save_d / llm_npz.stem
        llm_map.visualize(save_p)

for scene in gt_base.iterdir():
    for gt_npz in scene.iterdir():
        if (scene.stem, gt_npz.stem) not in samples:
            continue
        gt_map = BaseGridMap.load(gt_npz)
        save_d = Path(f"data/samples/{gt_ns}/{scene.stem}")
        save_d.mkdir(parents=True, exist_ok=True)
        save_p = save_d / gt_npz.stem
        gt_map.visualize(save_p)
