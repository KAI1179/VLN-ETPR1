"""T2.5: false-positive rates of contradiction rules R1 / R5 / R6 on GT replays (val_unseen).

Any trigger on a GT trajectory counts as a false positive. Env: OUT, CE_ORIG; RULE_SPLIT (default
val_unseen); CLAUSE_TOL_M (0.5), OFFTRACK_M (3.0). Needs oracle_progress_<split>.json and
clause_budget_train.json. Writes $OUT/rule_calibration_val_unseen.json and md/T2.5.md.
Exit 0 when every rule has a setting with episode FP rate < 5 %, else 1.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from common import issue
from common import (
    CLAUSE_TYPES,
    ClauseTrack,
    ce_gt,
    dump_json,
    env_float,
    hab_xz,
    load_json,
    load_oracle,
    md_path,
    md_table,
    out_dir,
)
from segments import replay

R5_BANDS = ((0.5, 2.0), (0.4, 2.5), (0.3, 3.0))
R6_GRID = ((0.5, 5), (0.5, 10), (1.0, 5), (1.0, 10))  # (radius m, gap steps)
# tightest first: a larger radius and a shorter gap trigger more; (1.0, 10) vs (0.5, 5) ordered by radius
R6_TIGHTNESS = ((1.0, 5), (1.0, 10), (0.5, 5), (0.5, 10))
FP_TARGET = 0.05


def r6_flags(
    locs: np.ndarray, step_clause_at_loc: np.ndarray, radius: float, gap: int
) -> np.ndarray:
    """flag[i]: location i is within `radius` of a location j <= i - gap visited in the same clause visit."""
    n = len(locs)
    flags = np.zeros(n, dtype=bool)
    entry = 0
    for i in range(n):
        if i > 0 and step_clause_at_loc[i] != step_clause_at_loc[i - 1]:
            entry = i
        hi = i - gap
        if hi < entry:
            continue
        d = np.linalg.norm(locs[entry : hi + 1] - locs[i], axis=1)
        flags[i] = bool((d <= radius).any())
    return flags


def main() -> int:
    split = os.environ.get("RULE_SPLIT", "val_unseen")
    tol, offtrack_m = env_float("CLAUSE_TOL_M", 0.5), env_float("OFFTRACK_M", 3.0)
    oracle, gt = load_oracle(split), ce_gt(split)
    budget = load_json(out_dir() / "clause_budget_train.json")
    med = budget["r5_median_by_n_clauses"]
    levels = list(budget["r1_budget"])  # P90 / P95 / P99

    r1_ep = {lv: 0 for lv in levels}
    r1_steps = {lv: 0 for lv in levels}
    r1_by_type = {lv: defaultdict(int) for lv in levels}
    r5_ep = {b: 0 for b in R5_BANDS}
    r5_side = {b: {"short": 0, "long": 0} for b in R5_BANDS}
    r6_ep = {g: 0 for g in R6_GRID}
    r6_steps = {g: 0 for g in R6_GRID}
    n_ep = n_steps = n_locs = offtrack_eps = 0
    offtrack_ids: list[dict[str, Any]] = []
    for ep in oracle["episodes"]:
        g = gt.get(str(ep["episode_id"]))
        if g is None:
            continue
        n_ep += 1
        track = ClauseTrack(ep, tol_m=tol, offtrack_m=offtrack_m)
        locs = hab_xz(g["locations"])
        r = replay(track, locs)
        if any(h.offtrack for h in r.hits):
            offtrack_eps += 1
            offtrack_ids.append({
                "episode_id": ep["episode_id"],
                "max_dist_m": round(max(h.dist for h in r.hits), 3),
            })
        n_steps += len(r.step_len)
        n_locs += len(locs)
        types = [ep["clauses"][c]["type"] for c in r.step_clause]
        for lv in levels:
            b = (
                np.array([budget["r1_budget"][lv][t] for t in types])
                if types
                else np.zeros(0)
            )
            trig = r.used_after > b
            r1_steps[lv] += int(trig.sum())
            if trig.any():
                r1_ep[lv] += 1
                for t in {types[i] for i in np.nonzero(trig)[0]}:
                    r1_by_type[lv][t] += 1
        walked = float(r.step_len.sum())
        m = med.get(str(len(ep["clauses"])), {}).get(
            "median_ref_arclen_m", float("nan")
        )
        for a, bb in R5_BANDS:
            if walked < a * m:
                r5_ep[(a, bb)] += 1
                r5_side[(a, bb)]["short"] += 1
            elif walked > bb * m:
                r5_ep[(a, bb)] += 1
                r5_side[(a, bb)]["long"] += 1
        clause_at_loc = np.array([h.idx for h in r.hits], dtype=int)
        for rad, gap in R6_GRID:
            f = r6_flags(locs, clause_at_loc, rad, gap)
            r6_steps[(rad, gap)] += int(f.sum())
            r6_ep[(rad, gap)] += bool(f.any())

    def rate(x: int, n: int) -> float:
        return round(x / n, 4) if n else float("nan")

    r1_table: dict[str, dict[str, Any]] = {
        lv: {
            "budget": budget["r1_budget"][lv],
            "episode_fp": rate(r1_ep[lv], n_ep),
            "step_rate": rate(r1_steps[lv], n_steps),
            "episodes_by_type": dict(r1_by_type[lv]),
        }
        for lv in levels
    }
    res: dict[str, Any] = {
        "split": split,
        "clause_tol_m": tol,
        "n_episodes": n_ep,
        "n_steps": n_steps,
        "offtrack_episodes": offtrack_eps,
        "offtrack_episode_ids": offtrack_ids,
        "R1": r1_table,
        "R2": "not applicable on GT replays (no satisfy() calls)",
        "R5": {
            f"a={a},b={b}": {
                "episode_fp": rate(r5_ep[(a, b)], n_ep),
                "short": r5_side[(a, b)]["short"],
                "long": r5_side[(a, b)]["long"],
                "step_rate": "n/a (end-of-episode rule)",
            }
            for a, b in R5_BANDS
        },
        "R6": {
            f"radius={rad},gap={gap}": {
                "episode_fp": rate(r6_ep[(rad, gap)], n_ep),
                "step_rate": rate(r6_steps[(rad, gap)], n_locs),
            }
            for rad, gap in R6_GRID
        },
    }
    # v0 default: the tightest setting whose episode FP rate is < 5 %, tightness in the declared order
    tightness = {
        "R1": levels,  # P90 is tighter than P95, P99
        "R5": [f"a={a},b={b}" for a, b in R5_BANDS],
        "R6": [f"radius={rad},gap={gap}" for rad, gap in R6_TIGHTNESS],
    }
    chosen, ok = {}, True
    for rule, order in tightness.items():
        passing = [k for k in order if res[rule][k]["episode_fp"] < FP_TARGET]
        chosen[rule] = passing[0] if passing else None
        ok &= bool(passing)
        if not passing:
            issue(
                f"T2.5: no {rule} setting reaches episode FP < 5 %; the rule needs new parameters before v0"
            )
    res["v0_defaults"] = chosen
    dump_json(out_dir() / "rule_calibration_val_unseen.json", res)

    lines = [
        "## T2.5 误报率表与选定的 v0 参数",
        "",
        f"split `{split}`，{n_ep} 条 GT 轨迹，{n_steps} 个前进步；GT 上任何触发都算误报。CLAUSE_TOL_M={tol}；offtrack（>{offtrack_m} m）episode：{offtrack_eps}。",
        "",
    ]
    r1_rows = [
        [
            lv,
            ", ".join(f"{t} {res['R1'][lv]['budget'][t]}" for t in CLAUSE_TYPES),
            f"{100 * res['R1'][lv]['episode_fp']:.2f}%",
            f"{100 * res['R1'][lv]['step_rate']:.3f}%",
            dict(res["R1"][lv]["episodes_by_type"]),
        ]
        for lv in levels
    ]
    lines += [
        "**R1 预算超支**（预算 = train 上该类型 `length_m` 的分位数，m）",
        "",
        md_table(
            [
                "档",
                "budget[type]",
                "episode 误报率",
                "按步触发率",
                "触发 episode 按类型",
            ],
            r1_rows,
        ),
        "",
    ]
    lines += (
        [f"offtrack episode 明细（GT 上即触发 O-偏离 v0）：{offtrack_ids}", ""]
        if offtrack_ids
        else []
    )
    lines += ["**R2 顺序违反**：GT 回放中不适用（没有 satisfy 调用），跳过。", ""]
    r5_rows = [
        [k, f"{100 * v['episode_fp']:.2f}%", v["short"], v["long"]]
        for k, v in res["R5"].items()
    ]
    lines += [
        "**R5 停止距离异常**（总水平路程 vs train 同子句数 ref 弧长中位数）",
        "",
        md_table(["(a, b)", "episode 误报率", "过短", "过长"], r5_rows),
        "",
    ]
    r6_rows = [
        [k, f"{100 * v['episode_fp']:.2f}%", f"{100 * v['step_rate']:.3f}%"]
        for k, v in res["R6"].items()
    ]
    lines += [
        "**R6 回环**（同一子句内回到 ≥gap 步之前位置的 radius 以内）",
        "",
        md_table(["参数", "episode 误报率", "按位置触发率"], r6_rows),
        "",
    ]
    lines += [
        "**v0 默认值**（episode 误报率 < 5% 的档中最紧的一档；紧度顺序 R1: P90 > P95 > P99，R5: (0.5,2.0) > (0.4,2.5) > (0.3,3.0)，R6: (1.0,5) > (1.0,10) > (0.5,5) > (0.5,10)）：",
        "",
        md_table(
            ["规则", "v0 默认"], [[k, v or "无满足档"] for k, v in chosen.items()]
        ),
        "",
    ]
    lines.append(
        f"判据（每条规则至少一档 < 5%）：{'满足' if ok else '不满足'}。R3 / R4 需要视觉验证器，本任务不做。"
    )
    md_path("T2.5").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
