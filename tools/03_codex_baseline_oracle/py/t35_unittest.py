"""t35_unittest.py — replay rand100 GT trajectories through the vendored oracle injector (no simulator).

usage: t35_unittest.py <oracleES/mcp dir> <oracle_by_index_rand100.json> <rand100_gt.json.gz> <rand100.json.gz>
Checks, per episode: progress idx monotone from 0 to n-1 (under progress_mask_final_zero=0), offtrack never, commit
never. Prints a summary; exit 0 when all pass, 1 otherwise.
"""

from __future__ import annotations

import gzip
import importlib
import json
import math
import os
import sys
from pathlib import Path


def yaw_from_positions(p0: list[float], p1: list[float], prev: float) -> float:
    dx, dz = p1[0] - p0[0], p1[2] - p0[2]
    if abs(dx) + abs(dz) < 1e-6:
        return prev
    return math.atan2(-dx, -dz)  # habitat: forward = (-sin yaw, -cos yaw)


def main(argv: list[str]) -> int:
    mcp_dir, table_p, gt_p, eps_p = (
        Path(argv[1]),
        Path(argv[2]),
        Path(argv[3]),
        Path(argv[4]),
    )
    gt = json.load(gzip.open(gt_p))
    eps = json.load(gzip.open(eps_p))["episodes"]
    sys.path.insert(0, str(mcp_dir))
    bad: list[str] = []
    n_ok = 0
    for index, e in enumerate(eps):
        eid = str(e["episode_id"])
        os.environ.update({
            "BAREES_ORACLE": "all",
            "BAREES_ORACLE_JSON": str(table_p),
            "BAREES_EPISODE_INDEX": str(index),
            "BAREES_ORACLE_MASK_FINAL": "0",
        })
        os.environ.pop("BAREES_LIVE_DIR", None)
        for m in ("oracles", "oracle_inject"):
            sys.modules.pop(m, None)
        oi = importlib.import_module("oracle_inject")
        st = oi.STATE
        if st.ep is None:
            bad.append(f"{eid}: not loaded {st.warnings}")
            continue
        locs, acts = gt[eid]["locations"], gt[eid]["actions"]
        yaw = 0.0
        idxs: list[int] = []
        off = commit = 0
        prev = locs[0]
        for k, pos in enumerate(locs):
            if k > 0:
                yaw = yaw_from_positions(prev, pos, yaw)
            prev = pos
            q = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]
            act = acts[k] if k < len(acts) else 0
            oi.on_step(
                act,
                {"info": {"position": pos, "orientation": q, "distance_to_goal": None}},
            )
            idxs.append(st.idx)
            off += bool(st.dist_route > oi.OFFTRACK_M)
            commit = int(st.commit_fired)
        n = len(st.ep["clauses"])
        mono = all(a <= b for a, b in zip(idxs, idxs[1:], strict=False))
        end_ok = idxs[-1] == n - 1
        if mono and end_ok and off == 0 and commit == 0:
            n_ok += 1
        else:
            bad.append(
                f"{eid}: monotone={mono} end={idxs[-1]}/{n - 1} offtrack_steps={off} commit={commit}"
            )
    print(
        f"[t35 unittest] {n_ok}/{len(eps)} episodes pass (progress 0->n-1 monotone, no offtrack, no commit)"
    )
    for b in bad[:20]:
        print("  ", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
