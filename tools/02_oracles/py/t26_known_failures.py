"""T2.6 (optional): R1 / R6 hit rates on recorded agent runs, split by success.

Searches for run directories that hold a summary.json and episode_<i>.jsonl files whose events carry
agent positions, under $WORKDIR/MIP (paper-era logs, if the repo ships any) and T26_ROOTS
(space-separated extra dirs). Scripted fake runs (fake_*, fakeapi_*) are ignored. No model is called.
Env: OUT, WORKDIR, T26_ROOTS; CLAUSE_TOL_M (0.5). Needs oracle_progress_{rand100,val_unseen}.json,
rule_calibration_val_unseen.json. Exit 0 when something was replayed, 2 (SKIP) when no usable log exists.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ClauseTrack,
    env_float,
    env_path,
    load_json,
    md_path,
    md_table,
    out_dir,
)
from segments import replay
from t25_rules import r6_flags

IGNORE = ("fake_", "fakeapi_")


def positions_in(obj: Any) -> list[list[float]]:
    """Every 3-float list stored under a key named 'position' / 'agent_position', depth-first."""
    found: list[list[float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (
                k in ("position", "agent_position")
                and isinstance(v, list)
                and len(v) == 3
                and all(isinstance(x, (int, float)) for x in v)
            ):
                found.append([float(x) for x in v])
            elif k not in ("goal", "goals", "episode", "reference_path"):
                found += positions_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found += positions_in(v)
    return found


def read_episode(path: Path) -> tuple[str | None, np.ndarray]:
    eid, pts = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "episode_meta":
                eid = str(ev.get("episode_id"))
                continue
            pts += positions_in(ev)
    arr = np.asarray(pts, dtype=float)[:, [0, 2]] if pts else np.zeros((0, 2))
    if len(arr) > 1:  # drop repeats (observe events report the same pose)
        keep = np.concatenate([
            [True],
            np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-6,
        ])
        arr = arr[keep]
    return eid, arr


def find_runs(roots: list[Path]) -> list[Path]:
    runs = []
    for root in roots:
        if not root.is_dir():
            continue
        for s in root.rglob("summary.json"):
            d = s.parent
            if "envs" in d.parts or d.name.startswith(IGNORE):
                continue
            if any(d.glob("episode_*.jsonl")):
                runs.append(d)
    return runs


def main() -> int:
    workdir = env_path("WORKDIR")
    roots = [
        workdir / "MIP",
        *[Path(p) for p in os.environ.get("T26_ROOTS", "").split()],
    ]
    tol = env_float("CLAUSE_TOL_M", 0.5)
    lines = ["## T2.6（可选）R1 / R6 在已知失败上的命中率", ""]
    runs = find_runs(roots)
    oracle: dict[str, dict] = {}
    for split in ("rand100", "val_unseen"):
        p = out_dir() / f"oracle_progress_{split}.json"
        if p.exists():
            oracle.update({str(e["episode_id"]): e for e in load_json(p)["episodes"]})
    cal = load_json(out_dir() / "rule_calibration_val_unseen.json")
    budget = load_json(out_dir() / "clause_budget_train.json")["r1_budget"][
        cal["v0_defaults"]["R1"] or "P99"
    ]
    r6 = cal["v0_defaults"]["R6"] or "radius=1.0,gap=10"
    rad, gap = (
        float(r6.split(",")[0].split("=")[1]),
        int(r6.split(",")[1].split("=")[1]),
    )
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "r1": 0, "r6": 0})
    used_runs = []
    for d in runs:
        summary = load_json(d / "summary.json")
        succ = {
            str(e.get("episode_id")): e.get("metrics", {}).get("success")
            for e in summary.get("episodes", [])
        }
        n_before = sum(v["n"] for v in stats.values())
        for jl in sorted(d.glob("episode_*.jsonl")):
            eid, locs = read_episode(jl)
            if (
                eid is None
                or eid not in oracle
                or len(locs) < 2
                or succ.get(eid) is None
            ):
                continue
            ep = oracle[eid]
            r = replay(ClauseTrack(ep, tol_m=tol), locs)
            types = [ep["clauses"][c]["type"] for c in r.step_clause]
            r1 = bool((r.used_after > np.array([budget[t] for t in types])).any())
            r6_hit = bool(
                r6_flags(locs, np.array([h.idx for h in r.hits]), rad, gap).any()
            )
            key = "success" if succ[eid] else "failure"
            stats[key]["n"] += 1
            stats[key]["r1"] += r1
            stats[key]["r6"] += r6_hit
        if sum(v["n"] for v in stats.values()) > n_before:
            used_runs.append(str(d))
    if not used_runs:
        lines += [
            f"SKIP：在 {', '.join(str(r) for r in roots)} 下没有找到带位置信息的非脚本化 episode 日志（{len(runs)} 个候选 run 目录均不可用）。MIP 仓库不附带论文运行日志；未调用任何模型。"
        ]
        md_path("T2.6").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 2
    rows = [
        [k, v["n"], f"{100 * v['r1'] / v['n']:.1f}%", f"{100 * v['r6'] / v['n']:.1f}%"]
        for k, v in stats.items()
    ]
    lines += [
        f"runs: {used_runs}",
        f"R1 = {cal['v0_defaults']['R1']}, R6 = {r6}",
        "",
        md_table(["episodes", "n", "R1 触发", "R6 触发"], rows),
    ]
    md_path("T2.6").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
