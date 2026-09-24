"""T2.4: clause budgets on train (parameters of contradiction rules R1 and R5).

Env: OUT, CE_ORIG; BUDGET_SPLIT (default train); CLAUSE_TOL_M (default 0.5), OFFTRACK_M (3.0).
Needs $OUT/oracle_progress_<split>.json (T2.2) and the split's GT. Writes $OUT/clause_budget_train.json and md/T2.4.md.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

from common import (
    CLAUSE_TYPES,
    ClauseTrack,
    ce_gt,
    dump_json,
    env_float,
    hab_xz,
    load_oracle,
    md_path,
    md_table,
    out_dir,
    percentiles,
)
from segments import replay

QS = (50, 90, 95, 99)
MIN_GROUP = 20  # smaller clause-count groups fall back to the nearest group with enough episodes


def main() -> int:
    split = os.environ.get("BUDGET_SPLIT", "train")
    tol, offtrack_m = env_float("CLAUSE_TOL_M", 0.5), env_float("OFFTRACK_M", 3.0)
    oracle, gt = load_oracle(split), ce_gt(split)
    length = defaultdict(list)  # type -> oracle length_m per clause
    fwd = defaultdict(list)  # type -> GT forward steps spent in the clause
    walked = defaultdict(list)  # type -> GT horizontal metres walked in the clause
    by_n = defaultdict(lambda: {"geodesic": [], "ref_arclen": [], "gt_walked": []})
    n_gt = 0
    for ep in oracle["episodes"]:
        for c in ep["clauses"]:
            length[c["type"]].append(c["length_m"])
        n = len(ep["clauses"])
        by_n[n]["geodesic"].append(ep["geodesic_distance"])
        by_n[n]["ref_arclen"].append(ep["cum_arclen_m"][-1])
        g = gt.get(str(ep["episode_id"]))
        if g is None:
            continue
        n_gt += 1
        track = ClauseTrack(ep, tol_m=tol, offtrack_m=offtrack_m)
        locs = hab_xz(g["locations"])
        r = replay(track, locs)
        by_n[n]["gt_walked"].append(float(r.step_len.sum()))
        for ci, c in enumerate(ep["clauses"]):
            mask = r.step_clause == ci
            fwd[c["type"]].append(int(mask.sum()))
            walked[c["type"]].append(float(r.step_len[mask].sum()))
    tables = {
        "length_m": {t: percentiles(length[t], QS) for t in CLAUSE_TYPES},
        "gt_forward_steps": {t: percentiles(fwd[t], QS) for t in CLAUSE_TYPES},
        "gt_walked_m": {t: percentiles(walked[t], QS) for t in CLAUSE_TYPES},
    }
    groups = {}
    for n, d in sorted(by_n.items()):
        groups[str(n)] = {
            k: {
                "n": len(v),
                "median": round(float(np.median(v)), 4),
                "P5": round(float(np.percentile(v, 5)), 4),
                "P95": round(float(np.percentile(v, 95)), 4),
            }
            if v
            else {"n": 0}
            for k, v in d.items()
        }
    enough = [int(n) for n, g in groups.items() if g["ref_arclen"]["n"] >= MIN_GROUP]
    median_ref = {}
    for n in range(1, max([int(x) for x in groups] + [1]) + 1):
        src = min(enough, key=lambda m: (abs(m - n), m)) if enough else None
        median_ref[str(n)] = {
            "median_ref_arclen_m": groups[str(src)]["ref_arclen"]["median"]
            if src
            else float("nan"),
            "from_group": src,
        }
    budget = {
        f"P{q}": {t: tables["length_m"][t][f"P{q}"] for t in CLAUSE_TYPES}
        for q in (90, 95, 99)
    }
    out = {
        "split": split,
        "clause_tol_m": tol,
        "n_episodes": len(oracle["episodes"]),
        "n_with_gt": n_gt,
        **tables,
        "totals_by_n_clauses": groups,
        "r1_budget": budget,
        "r5_median_by_n_clauses": median_ref,
        "min_group": MIN_GROUP,
    }
    dump_json(out_dir() / "clause_budget_train.json", out)

    def rows(tab: dict) -> list[list]:
        return [[t, tab[t]["n"], *[tab[t][f"P{q}"] for q in QS]] for t in CLAUSE_TYPES]

    hdr = ["type", "n", *[f"P{q}" for q in QS]]
    lines = [
        "## T2.4 预算表",
        "",
        f"split `{split}`，{len(oracle['episodes'])} 个 oracle episode，其中 {n_gt} 个有 GT；子句归属用 T2.3 单调版本，CLAUSE_TOL_M={tol}。",
        "",
    ]
    lines += [
        "**子句长度 `length_m`（oracle，参考路径水平弧长，m）** — R1 预算来源",
        "",
        md_table(hdr, rows(tables["length_m"])),
        "",
    ]
    lines += [
        "**GT 前进步数（每步 0.25 m，转向不计）**",
        "",
        md_table(hdr, rows(tables["gt_forward_steps"])),
        "",
    ]
    lines += [
        "**GT 在子句内实际走过的水平路程（m）** — 供参考：GT 沿 navmesh 走，比参考路径折线长",
        "",
        md_table(hdr, rows(tables["gt_walked_m"])),
        "",
    ]
    trows = [
        [
            n,
            g["ref_arclen"]["n"],
            *[
                f"{g[k]['median']} / {g[k]['P5']} / {g[k]['P95']}" if g[k]["n"] else "-"
                for k in ("geodesic", "ref_arclen", "gt_walked")
            ],
        ]
        for n, g in groups.items()
    ]
    lines += [
        "**总路径长度按子句数分组（median / P5 / P95，m）**",
        "",
        md_table(
            [
                "子句数",
                "n",
                "geodesic_distance",
                "ref_path 水平弧长",
                "GT 实走水平路程",
            ],
            trows,
        ),
        "",
    ]
    lines.append(
        f"R5 使用 `ref_path 水平弧长` 的组中位数；样本少于 {MIN_GROUP} 的组退到最近的足量组。"
    )
    md_path("T2.4").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
