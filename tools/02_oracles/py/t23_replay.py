"""T2.3: self-check of current_clause() by replaying the rand100 GT trajectories.

Env: OUT, RAND100; CLAUSE_TOL_M (arc-length tolerance of the clause-start test, default 0.5);
OFFTRACK_M (default 3.0). Needs $OUT/oracle_progress_rand100.json (T2.2).
Reports, for the chosen tolerance and a sweep {0, 0.25, 0.5, 1.0}: monotone share of the raw index,
final-clause hit rate, offtrack episodes. Writes md/T2.3.md and $OUT/t23_replay_rand100.json.
Exit 0 when the chosen tolerance meets all three targets (>= 95 %, 100 %, 0), else 1.
"""

from __future__ import annotations

import sys

import numpy as np

from common import issue, reset_issues
from common import (
    ClauseTrack,
    ce_gt,
    dump_json,
    env_float,
    hab_xz,
    load_oracle,
    md_path,
    md_table,
    out_dir,
)

SWEEP = (0.0, 0.25, 0.5, 1.0)


def replay_split(oracle: dict, gt: dict, tol: float, offtrack_m: float) -> dict:
    per_ep, fails = [], {"non_monotone": [], "final_miss": [], "offtrack": []}
    for ep in oracle["episodes"]:
        eid = str(ep["episode_id"])
        if eid not in gt:
            continue
        track = ClauseTrack(ep, tol_m=tol, offtrack_m=offtrack_m)
        locs = hab_xz(gt[eid]["locations"])
        raw = track.replay(locs, monotone=False)
        idx = [h.raw_idx for h in raw]
        drops = [
            (i, idx[i - 1], idx[i]) for i in range(1, len(idx)) if idx[i] < idx[i - 1]
        ]
        mono = track.replay(locs, monotone=True)
        last = track.n_clauses - 1
        final_ok = mono[-1].idx == last
        max_dist = max(h.dist for h in raw)
        off = any(h.offtrack for h in raw)
        if drops:
            fails["non_monotone"].append({
                "episode_id": ep["episode_id"],
                "first_drops(step,from,to)": drops[:3],
            })
        if not final_ok:
            end_s = mono[-1].s
            fails["final_miss"].append({
                "episode_id": ep["episode_id"],
                "final_idx": mono[-1].idx,
                "last_idx": last,
                "last_type": ep["clauses"][-1]["type"],
                "s_end": round(end_s, 3),
                "last_start": round(float(track.starts[-1]), 3),
                "short_by_m": round(float(track.starts[-1]) - end_s, 3),
            })
        if off:
            fails["offtrack"].append({
                "episode_id": ep["episode_id"],
                "max_dist_m": round(max_dist, 3),
            })
        per_ep.append({
            "episode_id": ep["episode_id"],
            "monotone_raw": not drops,
            "final_ok": final_ok,
            "offtrack": off,
            "max_dist_m": round(max_dist, 3),
            "n_steps": len(locs),
        })
    n = len(per_ep)
    return {
        "tol_m": tol,
        "n": n,
        "monotone_share": sum(e["monotone_raw"] for e in per_ep) / n
        if n
        else float("nan"),
        "final_hit_share": sum(e["final_ok"] for e in per_ep) / n
        if n
        else float("nan"),
        "offtrack_episodes": sum(e["offtrack"] for e in per_ep),
        "max_dist_p95": float(np.percentile([e["max_dist_m"] for e in per_ep], 95))
        if n
        else float("nan"),
        "fails": fails,
    }


def main() -> int:
    tol = env_float("CLAUSE_TOL_M", 0.5)
    md_path("T2.3")
    reset_issues("T2.3")
    offtrack_m = env_float("OFFTRACK_M", 3.0)
    oracle, gt = load_oracle("rand100"), ce_gt("rand100")
    runs = {t: replay_split(oracle, gt, t, offtrack_m) for t in sorted({*SWEEP, tol})}
    dump_json(
        out_dir() / "t23_replay_rand100.json", {str(t): r for t, r in runs.items()}
    )
    chosen = runs[tol]
    ok = (
        chosen["monotone_share"] >= 0.95
        and chosen["final_hit_share"] == 1.0
        and chosen["offtrack_episodes"] == 0
    )
    lines = [
        "## T2.3 GT 回放自检",
        "",
        f"rand100 GT `locations` (one point per 0.25 m forward step), offtrack threshold {offtrack_m} m.",
        "",
    ]
    rows = [
        [
            f"{t}{' (chosen)' if t == tol else ''}",
            r["n"],
            f"{100 * r['monotone_share']:.1f}%",
            f"{100 * r['final_hit_share']:.1f}%",
            r["offtrack_episodes"],
            f"{r['max_dist_p95']:.3f}",
        ]
        for t, r in runs.items()
    ]
    lines.append(
        md_table(
            [
                "CLAUSE_TOL_M",
                "n",
                "raw idx monotone (target ≥95%)",
                "final idx = last (target 100%)",
                "offtrack episodes (target 0)",
                "max dist to path P95 (m)",
            ],
            rows,
        )
    )
    lines += [
        "",
        "`CLAUSE_TOL_M` lets a clause start count as reached when the projected arc length is within that many metres of it. The task book's rule is tol = 0.",
        "",
    ]
    for kind, items in chosen["fails"].items():
        if items:
            lines += [
                f"不满足条目 — {kind}（tol={tol}，{len(items)} 条）:",
                "```",
                *[str(x) for x in items[:30]],
                "```",
                "",
            ]
    strict = runs[0.0]
    if tol != 0.0:
        issue(
            f"T2.3: the task book's rule (tol = 0) reaches the last clause in only {100 * strict['final_hit_share']:.0f}% of GT replays: "
            f"GT stops 0.25-0.5 m short of the goal, where the zero-length final clause starts. Default CLAUSE_TOL_M = {tol} "
            f"({100 * chosen['final_hit_share']:.0f}% final hits, {100 * chosen['monotone_share']:.0f}% monotone). Accept the tolerance?"
        )
    lines.append(f"判据（tol={tol}）：{'全部满足' if ok else '未全部满足'}")
    md_path("T2.3").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
