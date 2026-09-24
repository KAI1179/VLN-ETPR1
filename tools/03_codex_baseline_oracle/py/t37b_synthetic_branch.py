"""T3.7(b): synthetic wrong-branch recall of R1 / R6 / O-offtrack / O-commit on rand100.

For every rand100 reference path and every turning vertex (direction change > turn_deg, see rules_v0
commit_oracle), 2-3 "wrong branch" trajectories are built: the GT trajectory up to the location nearest
the turning viewpoint, then a DIFFERENT connectivity edge (unobstructed, included, not on the R2R path),
extended greedily (smallest turn) to BRANCH_DEPTH_MAX viewpoints, interpolated at STEP_M. The branch
candidates are the neighbours sorted by their angle to the GT next segment, spread over that range
(first / middle / last), so small-angle (hard) and large-angle branches are both covered.

Inputs: oracle_progress_rand100 ($OUT3, else $OUT), rand100 GT + episodes, $R2R_DISC/R2R_val_unseen.json
(viewpoint ids of each path), $CONN/<scan>_connectivity.json, $OUT3/rules_v0.json (T3.0 + T3.6).
Coordinates: mp3d(x, y, z) = (hab_x, -hab_z, hab_y); all distances horizontal.
Outputs: $OUT3/t37b_synthetic_rand100.json, md/T3.7b.md. Exit 0 when branches were built, 2 SKIP otherwise.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import oracles as O
import t3common as t3
from common import ClauseTrack, hab_xz

RULES = ("R1", "R6", "offtrack", "commit")
ANGLE_BINS = ((0, 45), (45, 90), (90, 181))


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


@lru_cache(maxsize=64)
def connectivity(scan: str) -> dict[str, dict[str, Any]]:
    conn = t3.load_json(t3.env_path("CONN") / f"{scan}_connectivity.json")
    ids = [c["image_id"] for c in conn]
    return {
        c["image_id"]: {
            "xz": np.array([float(c["pose"][3]), -float(c["pose"][7])]),
            "included": bool(c["included"]),
            "nbrs": [
                ids[j]
                for j, ok in enumerate(c["unobstructed"])
                if ok and conn[j]["included"]
            ],
        }
        for c in conn
    }


def interpolate(points: list[np.ndarray], step_m: float) -> np.ndarray:
    """Dense polyline through `points` with spacing <= step_m (first point included)."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step_m)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


def wrong_branches(
    conn: dict[str, dict[str, Any]],
    path: list[str],
    k: int,
    depth_max: int,
    per_vertex: int,
) -> list[tuple[list[str], float]]:
    """(chain of viewpoint ids, angle to the GT next segment) for up to per_vertex branches at path[k]."""
    v = path[k]
    on_path = set(path)
    tangent = conn[path[k + 1]]["xz"] - conn[v]["xz"]
    cands = [n for n in conn[v]["nbrs"] if n not in on_path]
    scored = sorted(
        ((O.angle_deg(conn[n]["xz"] - conn[v]["xz"], tangent), n) for n in cands),
        key=lambda x: x[0],
    )
    if len(scored) > per_vertex:
        idx = np.linspace(0, len(scored) - 1, per_vertex).round().astype(int)
        scored = [scored[i] for i in dict.fromkeys(int(i) for i in idx)]
    out = []
    for ang, first in scored:
        chain, prev, cur = [first], v, first
        while len(chain) < depth_max:
            d0 = conn[cur]["xz"] - conn[prev]["xz"]
            nxt = [n for n in conn[cur]["nbrs"] if n not in on_path and n not in chain]
            if not nxt:
                break
            best = min(
                nxt, key=lambda n: O.angle_deg(conn[n]["xz"] - conn[cur]["xz"], d0)
            )
            chain.append(best)
            prev, cur = cur, best
        if len(chain) >= 2:
            out.append((chain, ang))
    return out


def replay_branch(
    ep: dict[str, Any], steps: list[O.AgentStep], branch_at: int, rules: dict[str, Any]
) -> dict[str, Any]:
    tol, mode = rules["clause_tol"]["m"], rules["clause_tol"]["mode"]
    track = ClauseTrack(ep, tol_m=tol, offtrack_m=rules["offtrack_m"], tol_mode=mode)
    co_p = rules.get("commit_oracle") or {}
    commit = O.CommitOracle(
        np.asarray(ep["ref_path_xz"]),
        O.CommitParams(
            theta_deg=co_p.get("theta_deg", 45.0),
            advance_m=co_p.get("advance_m", 1.0),
            vertex_radius_m=co_p.get("vertex_radius_m", 2.0),
            turn_deg=co_p.get("turn_deg", 30.0),
            ontrack_m=co_p.get("ontrack_m", 1.0),
            sustained=co_p.get("sustained", True),
        ),
    )
    r1 = O.BudgetRule(
        budget_m=rules["R1"]["type"], budget_steps=rules["R1"].get("steps")
    )
    r6 = O.LoopDetector(rules["R6"]["radius_m"], rules["R6"]["gap_forward_steps"])
    first: dict[str, int | None] = dict.fromkeys(RULES)
    prefix_hits: dict[str, int] = dict.fromkeys(RULES, 0)
    prev_idx, max_dist = -1, 0.0
    for i, s in enumerate(steps):
        h = track.locate(s.xz, prev_idx, monotone=True)
        prev_idx = h.idx
        max_dist = max(max_dist, h.dist)
        flags = {
            "R1": r1.step(s.xz, h.idx, track.types[h.idx], s.forward),
            "R6": r6.step(s.xz, h.idx, s.forward),
            "offtrack": h.offtrack,
            "commit": commit.step(s.xz, s.yaw),
        }
        for r, f in flags.items():
            if not f:
                continue
            if i < branch_at:
                prefix_hits[r] += 1
            elif first[r] is None:
                first[r] = i
    return {
        "first": first,
        "prefix_hits": prefix_hits,
        "max_dist_m": round(max_dist, 3),
    }


def main() -> int:
    t3.md3_path("T3.7b")
    t3.reset_issues3("T3.7b")
    head = "### T3.7(b) 合成错分支召回（rand100）"
    rules = t3.load_rules_v0()
    oracle, gt, eps = (
        t3.load_oracle3("rand100"),
        t3.load_gt("rand100"),
        t3.load_episodes("rand100"),
    )
    r2r_path = t3.env_path("R2R_DISC") / "R2R_val_unseen.json"
    missing = [
        name
        for name, ok in (
            ("rules_v0.json (T3.0)", rules is not None),
            ("oracle_progress_rand100", oracle is not None),
            ("rand100 GT", gt is not None),
            ("rand100 episodes", eps is not None),
            (str(r2r_path), r2r_path.is_file()),
            (
                f"{t3.env_path('CONN')}/*_connectivity.json",
                any(t3.env_path("CONN").glob("*_connectivity.json")),
            ),
        )
        if not ok
    ]
    if missing:
        t3.write_md("T3.7b", [head, "", f"SKIP：缺少输入 {missing}。"])
        return 2
    assert (
        rules is not None and oracle is not None and gt is not None and eps is not None
    )
    if "commit_oracle" not in rules:
        t3.issue3(
            "T3.7b: rules_v0.json has no commit_oracle (T3.6 not run); default theta 45 deg / d 1.0 m used"
        )
    depth_max, per_vertex = (
        env_int("BRANCH_DEPTH_MAX", 3),
        env_int("BRANCH_PER_VERTEX", 3),
    )
    step_m = t3.env_float("STEP_M", 0.25)
    turn_deg = float((rules.get("commit_oracle") or {}).get("turn_deg", 30.0))
    r2r = {int(x["path_id"]): x for x in t3.load_json(r2r_path)}
    records: list[dict[str, Any]] = []
    n_turn = n_ep_turn = n_skipped = 0
    for ep in oracle["episodes"]:
        eid = str(ep["episode_id"])
        g, e, item = gt.get(eid), eps.get(eid), r2r.get(int(ep["path_id"]))
        ref = np.asarray(ep["ref_path_xz"], dtype=float)
        if g is None or e is None or item is None or len(item["path"]) != len(ref):
            n_skipped += 1
            continue
        conn = connectivity(item["scan"])
        turns = O.turning_vertices(ref, turn_deg)
        n_turn += len(turns)
        n_ep_turn += bool(turns)
        gsteps = O.gt_steps(g["locations"], g["actions"], e["start_rotation"])
        locs = hab_xz(g["locations"])
        for k in turns:
            i_k = int(np.argmin(np.linalg.norm(locs - ref[k], axis=1)))
            prefix = [s for s in gsteps if s.loc_index <= i_k]
            # cut after the arrival at i_k (turn actions taken there belong to the GT continuation)
            arrive = next(i for i, s in enumerate(prefix) if s.loc_index == i_k)
            prefix = prefix[: arrive + 1]
            for chain, ang in wrong_branches(
                conn, item["path"], k, depth_max, per_vertex
            ):
                dense = interpolate(
                    [locs[i_k], *[conn[c]["xz"] for c in chain]], step_m
                )
                branch = O.steps_from_polyline(dense[1:], first_loc_index=i_k + 1)
                steps = [*prefix, *branch]
                rep = replay_branch(ep, steps, len(prefix), rules)
                records.append({
                    "episode_id": ep["episode_id"],
                    "vertex": k,
                    "chain": chain,
                    "depth": len(chain),
                    "angle_to_gt_deg": round(ang, 1),
                    "branch_len_m": round(
                        float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()), 2
                    ),
                    "branch_steps": len(branch),
                    "gt_offset_at_vertex_m": round(
                        float(np.linalg.norm(locs[i_k] - ref[k])), 3
                    ),
                    "delay_steps": {
                        r: (rep["first"][r] - len(prefix) + 1)
                        if rep["first"][r] is not None
                        else None
                        for r in RULES
                    },
                    "prefix_hits": rep["prefix_hits"],
                    "max_dist_m": rep["max_dist_m"],
                })
    t3.dump_json(
        t3.out3_dir() / "t37b_synthetic_rand100.json",
        {
            "rules": {
                k: rules[k]
                for k in ("R1", "R6", "offtrack_m", "clause_tol")
                if k in rules
            },
            "commit_oracle": rules.get("commit_oracle"),
            "depth_max": depth_max,
            "per_vertex": per_vertex,
            "step_m": step_m,
            "n_branches": len(records),
            "branches": records,
        },
    )
    if not records:
        t3.write_md(
            "T3.7b", [head, "", "SKIP：没有构造出任何错分支（拐点无可用邻接边）。"]
        )
        return 2

    def summary(recs: list[dict[str, Any]]) -> list[list[Any]]:
        rows = []
        for r in RULES:
            delays = [
                x["delay_steps"][r] for x in recs if x["delay_steps"][r] is not None
            ]
            rows.append([
                r,
                t3.pct(len(delays) / len(recs)),
                f"{np.median(delays):.0f} / {np.percentile(delays, 90):.0f} / {max(delays)}"
                if delays
                else "-",
                sum(x["prefix_hits"][r] for x in recs),
            ])
        any_hit = sum(any(x["delay_steps"][r] is not None for r in RULES) for x in recs)
        first_any = [
            min(v for v in x["delay_steps"].values() if v is not None)
            for x in recs
            if any(v is not None for v in x["delay_steps"].values())
        ]
        rows.append([
            "any",
            t3.pct(any_hit / len(recs)),
            f"{np.median(first_any):.0f} / {np.percentile(first_any, 90):.0f} / {max(first_any)}"
            if first_any
            else "-",
            "-",
        ])
        return rows

    hdr = [
        "规则",
        "触发率（分支点后）",
        "触发延迟 步 P50 / P90 / max",
        "分支点前（GT 前缀）触发数",
    ]
    depth_c = Counter(x["depth"] for x in records)
    corridor = sum(x["max_dist_m"] <= rules["offtrack_m"] for x in records)
    co = rules.get("commit_oracle") or {}
    lines = [
        head,
        "",
        (
            f"{oracle['n_episodes']} 条 rand100 中 {n_ep_turn} 条含拐点（> {turn_deg:g}°），拐点 {n_turn} 个，跳过 {n_skipped} 条；"
            f"错分支 {len(records)} 条（深度 {dict(sorted(depth_c.items()))}，每拐点最多 {per_vertex} 条，{step_m} m 插值）。"
            f"规则参数来自 rules_v0.json：R1 {rules['R1'].get('level', '')} 距离预算"
            f"{'+步数预算' if rules['R1'].get('steps') else ''}，R6 {rules['R6']['radius_m']} m / {rules['R6']['gap_forward_steps']} 步，"
            f"offtrack {rules['offtrack_m']} m，commit θ={co.get('theta_deg', 45)}° d={co.get('advance_m', 1.0)} m。"
            f"延迟 = 首次触发步 − 分支点步（每步 {step_m} m）。"
        ),
        "",
        t3.md_table(hdr, summary(records)),
        "",
        f"分支全程距参考路径 ≤ {rules['offtrack_m']} m（O-偏离原理上不可能触发）的分支：{corridor} / {len(records)}。",
        "",
        "按分支与 GT 下一段夹角分组：",
        "",
    ]
    for lo, hi in ANGLE_BINS:
        sub = [x for x in records if lo <= x["angle_to_gt_deg"] < hi]
        if sub:
            lines += [
                f"角度 [{lo}, {hi}) — {len(sub)} 条：",
                "",
                t3.md_table(hdr, summary(sub)),
                "",
            ]
    for depth in sorted(depth_c):
        sub = [x for x in records if x["depth"] == depth]
        lines += [
            f"深度 {depth} — {len(sub)} 条：",
            "",
            t3.md_table(hdr, summary(sub)),
            "",
        ]
    lines.append("结论：PASS（已构造并回放；召回数字见表）")
    t3.write_md("T3.7b", lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
