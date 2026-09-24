"""B1: fields and size of an R2R-CE split file.

Env: VLNCE_EPISODES (val_unseen.json.gz, or rand100.json.gz as fallback),
VLNCE_GT (optional matching *_gt.json.gz).
"""

from __future__ import annotations

import sys

from _common import env_path, episodes_of, load_json


def main() -> int:
    path = env_path("VLNCE_EPISODES")
    if path is None or not path.is_file():
        print("B1 SKIP: VLNCE_EPISODES not set or missing:", path)
        return 2
    print("file:", path)
    eps = episodes_of(path)
    print("n_episodes", len(eps))
    e = eps[0]
    print("top-level keys:", list(e.keys()))
    for k in ("episode_id", "trajectory_id", "scene_id"):
        print(k, e.get(k))
    print("instruction keys:", list(e["instruction"].keys()))
    print("instruction_id:", e["instruction"].get("instruction_id"))
    ref = e.get("reference_path", [])
    print("reference_path len:", len(ref), "first:", ref[0] if ref else None)
    print("goals:", e.get("goals"))
    print("has viewpoint ids in reference_path?", any(isinstance(x, dict) for x in ref))
    all_xyz = all(
        isinstance(x, list) and len(x) == 3
        for ep in eps
        for x in ep.get("reference_path", [])
    )
    print("all reference_path entries are 3-float lists?", all_xyz)
    lens = [len(ep.get("reference_path", [])) for ep in eps]
    print(
        "reference_path len min/mean/max:",
        min(lens),
        round(sum(lens) / len(lens), 2),
        max(lens),
    )
    print("unique scene_id:", len({ep["scene_id"] for ep in eps}))
    print("unique trajectory_id:", len({ep.get("trajectory_id") for ep in eps}))
    print(
        "episodes with instruction_id:",
        sum(1 for ep in eps if "instruction_id" in ep["instruction"]),
    )
    gt = env_path("VLNCE_GT")
    if gt is not None and gt.is_file():
        g = load_json(gt)
        first = next(iter(g.values()))
        print("GT file:", gt, "n keys", len(g), "value keys", list(first.keys()))
        print("GT locations len (first):", len(first.get("locations", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
