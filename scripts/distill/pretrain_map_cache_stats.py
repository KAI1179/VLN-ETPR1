from pathlib import Path

import numpy as np
from tap import Tap


class Args(Tap):
    sample_size: int = 200


def summarize(files):
    zero = []
    active = []
    mins = []
    maxs = []
    dtypes = set()
    starts = []
    dirs = []
    for path in files:
        with np.load(path) as data:
            grid = data["grid"]
            zero.append(float(np.all(grid == 0)))
            active.append(float(np.any(grid >= 0.5, axis=0).mean()))
            mins.append(float(grid.min()))
            maxs.append(float(grid.max()))
            dtypes.add(str(grid.dtype))
            starts.append(data["start_position"])
            dirs.append(data["direction_vectors"])
    return {
        "n": len(files),
        "zero": float(np.mean(zero)) if zero else float("nan"),
        "active": float(np.mean(active)) if active else float("nan"),
        "min": (min(mins) if mins else float("nan")),
        "max": (max(maxs) if maxs else float("nan")),
        "dtype": ",".join(sorted(dtypes)),
        "start_range": (float(np.min(starts)), float(np.max(starts))) if starts else (float("nan"), float("nan")),
        "dir_range": (float(np.min(dirs)), float(np.max(dirs))) if dirs else (float("nan"), float("nan")),
    }


def main():
    args = Args().parse_args()
    roots = {
        "LLM train": Path("data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/train/cognitive_maps/raster"),
        "GT train": Path("data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster"),
    }
    print("source n zero_ratio active_ratio min max dtype start_range direction_range")
    for name, root in roots.items():
        files = sorted(root.glob("*/*.npz"))
        if name == "GT train":
            files = [p for p in files if "R2R_train_" in p.name]
        if not files:
            print(name, "MISSING")
            continue
        idx = np.linspace(0, len(files) - 1, min(args.sample_size, len(files))).astype(int)
        s = summarize([files[i] for i in idx])
        print(name, s["n"], f'{s["zero"]:.6f}', f'{s["active"]:.6f}', s["min"], s["max"], s["dtype"], s["start_range"], s["dir_range"])


if __name__ == "__main__":
    main()
