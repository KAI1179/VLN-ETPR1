"""t33_export_traj.py — per-episode trajectories from an archived MIP run (poses.jsonl written by the oracleES injector,
or, for bareES runs, nothing: bareES logs no pose — the baseline of T3.3 therefore MUST run on the oracleES arm with
oracle=none, which is byte-identical to bareES for the model and logs poses).

usage: t33_export_traj.py <run_archive_dir> <traj_out_dir>
Writes <traj_out_dir>/ep<episode_id>.json: {episode_id, index, success, distance_to_goal, end_reason, steps: [...]}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    run, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    n = 0
    for e in summary.get("episodes") or []:
        idx, eid = e.get("index"), e.get("episode_id")
        poses = run / f"live_{idx}" / "poses.jsonl"
        if not poses.is_file():
            print(
                f"episode {eid} (index {idx}): no poses.jsonl (bareES arm logs no pose; use oracleES oracle=none)"
            )
            continue
        steps = [json.loads(ln) for ln in poses.read_text().splitlines() if ln.strip()]
        steps = [s for s in steps if s.get("source") == "step"]
        rec = {
            "episode_id": eid,
            "index": idx,
            "success": e.get("metrics", {}).get("success"),
            "distance_to_goal": e.get("metrics", {}).get("distance_to_goal"),
            "end_reason": e.get("agent", {}).get("end_reason"),
            "steps": steps,
        }
        (out / f"ep{eid}.json").write_text(json.dumps(rec))
        n += 1
    print(f"exported {n} trajectories to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
