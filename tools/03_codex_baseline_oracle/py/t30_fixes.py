"""T3.0: the task-2 fixes, re-run of T2.2 / T2.3 / T2.4 / T2.5 with the new settings, rules_v0.json.

(1) clause-start tolerance: CLAUSE_TOL_MODE=final_only (0 m for intermediate clauses, CLAUSE_TOL_M for a
    final zero-length clause only); (2) turn keyword rule TURN_RULE=v1 (adds left | right); (3) offtrack GT
    episode ids; (4) R6 = same clause, within radius of a position >= gap FORWARD steps earlier;
    (5) R5 long side only (> 2.0x and > 2.5x the train median).

Inputs: task 2's $OUT/oracle_progress_{rand100,val_unseen,train}.json(.gz) (the clause geometry is
unchanged, only the types are re-derived from the clause text), the splits' GT and episode files.
Outputs ($OUT3): oracle_progress_<split>.json (re-typed), t30_replay_rand100.json, clause_budget_train.json,
rule_calibration_<split>.json, rules_v0.json, md/T3.0.md.
Exit 0 when the rand100 replay meets 100 % / 100 % / 0 and every rule has a setting with episode FP < 5 %
on the calibration split; 1 when a criterion fails; 2 when the calibration split (RULE_SPLIT, default
val_unseen) has no GT here (rules_v0.json is then provisional, calibrated on rand100).
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import oracles as O
import t3common as t3
from common import CLAUSE_TYPES, ClauseTrack, classify, hab_xz, percentiles
from segments import replay
from t23_replay import replay_split
from t25_rules import r6_flags

SPLITS = ("rand100", "val_unseen", "train")
R1_LEVELS = ("P90", "P95", "P99")
R1_CHOSEN = "P99"
R5_FACTORS = (2.0, 2.5)
R6_GRID = ((0.5, 5), (0.5, 10), (1.0, 5), (1.0, 10))  # (radius m, gap forward steps)
R6_CHOSEN = (1.0, 10)
FP_TARGET = 0.05
QS = (50, 90, 95, 99)
MIN_GROUP = 20  # as task 2 T2.4


# ── (2) re-typing = T2.2 rerun ─────────────────────────────────────────────


def retype(split: str) -> tuple[dict[str, Any], Counter, Counter, Counter] | None:
    data = t3.load_json_or_gz(t3.env_path("OUT") / f"oracle_progress_{split}.json")
    if data is None:
        return None
    before, after, moved = Counter(), Counter(), Counter()
    rule = t3.turn_rule()
    for ep in data["episodes"]:
        for c in ep["clauses"]:
            old = c["type"]
            new = classify(c["text"], turn_rule=rule)
            before[old] += 1
            after[new] += 1
            if old != new:
                moved[f"{old}->{new}"] += 1
            c["type"] = new
    data["turn_rule"] = rule
    data["retyped_from"] = str(t3.env_path("OUT"))
    t3.dump_json(t3.out3_dir() / f"oracle_progress_{split}.json", data)
    return data, before, after, moved


# ── (1) T2.3 rerun ─────────────────────────────────────────────────────────


def part_replay(oracle: dict, gt: dict) -> dict[str, dict]:
    tol, mode, off = t3.clause_tol_m(), t3.clause_tol_mode(), t3.offtrack_m()
    variants = [("all", 0.0), ("all", tol), ("final_only", tol), (mode, tol)]
    runs: dict[str, dict] = {}
    for m, tv in dict.fromkeys(variants):
        runs[f"{m}/{tv}"] = replay_split(oracle, gt, tv, off, tol_mode=m)
    return runs


# ── T2.4 rerun ─────────────────────────────────────────────────────────────


def part_budget(oracle: dict, gt: dict | None) -> dict[str, Any]:
    tol, mode, off = t3.clause_tol_m(), t3.clause_tol_mode(), t3.offtrack_m()
    length: dict[str, list[float]] = defaultdict(list)
    fwd: dict[str, list[float]] = defaultdict(list)
    walked: dict[str, list[float]] = defaultdict(list)
    by_n: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"geodesic": [], "ref_arclen": [], "gt_walked": []}
    )
    n_gt = 0
    for ep in oracle["episodes"]:
        for c in ep["clauses"]:
            length[c["type"]].append(c["length_m"])
        n = len(ep["clauses"])
        by_n[n]["geodesic"].append(ep["geodesic_distance"])
        by_n[n]["ref_arclen"].append(ep["cum_arclen_m"][-1])
        g = gt.get(str(ep["episode_id"])) if gt else None
        if g is None:
            continue
        n_gt += 1
        r = replay(
            ClauseTrack(ep, tol_m=tol, offtrack_m=off, tol_mode=mode),
            hab_xz(g["locations"]),
        )
        by_n[n]["gt_walked"].append(float(r.step_len.sum()))
        for ci, c in enumerate(ep["clauses"]):
            mask = r.step_clause == ci
            fwd[c["type"]].append(float(mask.sum()))
            walked[c["type"]].append(float(r.step_len[mask].sum()))
    tables = {
        "length_m": {t: percentiles(length[t], QS) for t in CLAUSE_TYPES},
        "gt_forward_steps": {t: percentiles(fwd[t], QS) for t in CLAUSE_TYPES},
        "gt_walked_m": {t: percentiles(walked[t], QS) for t in CLAUSE_TYPES},
    }
    groups: dict[str, dict[str, Any]] = {}
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
    r1 = {
        f"P{q}": {t: tables["length_m"][t][f"P{q}"] for t in CLAUSE_TYPES}
        for q in (90, 95, 99)
    }
    r1_steps = (
        {
            f"P{q}": {t: tables["gt_forward_steps"][t][f"P{q}"] for t in CLAUSE_TYPES}
            for q in (90, 95, 99)
        }
        if n_gt
        else None
    )
    return {
        "split": oracle["split"],
        "clause_tol_m": tol,
        "clause_tol_mode": mode,
        "turn_rule": oracle.get("turn_rule"),
        "n_episodes": len(oracle["episodes"]),
        "n_with_gt": n_gt,
        **tables,
        "totals_by_n_clauses": groups,
        "r1_budget": r1,
        "r1_budget_steps": r1_steps,
        "r5_median_by_n_clauses": median_ref,
        "min_group": MIN_GROUP,
    }


# ── T2.5 rerun ─────────────────────────────────────────────────────────────


def part_rules(
    oracle: dict, gt: dict, eps: dict | None, budget: dict, split: str
) -> dict[str, Any]:
    tol, mode, off = t3.clause_tol_m(), t3.clause_tol_mode(), t3.offtrack_m()
    med = budget["r5_median_by_n_clauses"]
    steps_budget = budget.get("r1_budget_steps")
    r1_ep = {lv: {"m": 0, "steps": 0} for lv in R1_LEVELS}
    r1_st = {lv: {"m": 0, "steps": 0} for lv in R1_LEVELS}
    r1_by_type = {lv: Counter() for lv in R1_LEVELS}
    r5_ep = {f: 0 for f in R5_FACTORS}
    r6_ep = {g: 0 for g in R6_GRID}
    r6_locs = {g: 0 for g in R6_GRID}
    r6_old_mismatch = 0
    n_ep = n_steps = n_locs = 0
    offtrack_ids: list[dict[str, Any]] = []
    for ep in oracle["episodes"]:
        g = gt.get(str(ep["episode_id"]))
        if g is None:
            continue
        n_ep += 1
        track = ClauseTrack(ep, tol_m=tol, offtrack_m=off, tol_mode=mode)
        locs = hab_xz(g["locations"])
        r = replay(track, locs)
        if any(h.offtrack for h in r.hits):
            offtrack_ids.append({
                "episode_id": ep["episode_id"],
                "max_dist_m": round(max(h.dist for h in r.hits), 3),
                "first_step": int(next(i for i, h in enumerate(r.hits) if h.offtrack)),
            })
        n_steps += len(r.step_len)
        n_locs += len(locs)
        types = [ep["clauses"][c]["type"] for c in r.step_clause]
        # steps used inside the clause after each step (same reset convention as used_after)
        used_steps = np.zeros(len(r.step_len), dtype=int)
        acc, cur = 0, -1
        for i, c in enumerate(r.step_clause):
            if c != cur:
                acc, cur = 0, int(c)
            acc += 1
            used_steps[i] = acc
        for lv in R1_LEVELS:
            b = (
                np.array([budget["r1_budget"][lv][t] for t in types])
                if types
                else np.zeros(0)
            )
            trig = r.used_after > b
            r1_st[lv]["m"] += int(trig.sum())
            if trig.any():
                r1_ep[lv]["m"] += 1
                for t in {types[i] for i in np.nonzero(trig)[0]}:
                    r1_by_type[lv][t] += 1
            if steps_budget is not None:
                bs = (
                    np.array([steps_budget[lv][t] for t in types])
                    if types
                    else np.zeros(0)
                )
                ts = used_steps > bs
                r1_st[lv]["steps"] += int(ts.sum())
                r1_ep[lv]["steps"] += bool(ts.any())
        walked = float(r.step_len.sum())
        m = med.get(str(len(ep["clauses"])), {}).get(
            "median_ref_arclen_m", float("nan")
        )
        for f in R5_FACTORS:
            r5_ep[f] += walked > f * m
        clause_at_loc = np.array([h.idx for h in r.hits], dtype=int)
        e = (eps or {}).get(str(ep["episode_id"]))
        steps = O.gt_steps(
            g["locations"], g["actions"], e["start_rotation"] if e else None
        )
        for rad, gap in R6_GRID:
            det = O.LoopDetector(rad, gap)
            flags = np.zeros(len(locs), dtype=bool)
            for s in steps:
                f = det.step(s.xz, int(clause_at_loc[s.loc_index]), s.forward)
                if s.forward:
                    flags[s.loc_index] = f
            r6_locs[(rad, gap)] += int(flags.sum())
            r6_ep[(rad, gap)] += bool(flags.any())
            r6_old_mismatch += int(
                (flags != r6_flags(locs, clause_at_loc, rad, gap)).sum()
            )

    def rate(x: int, n: int) -> float:
        return round(x / n, 4) if n else float("nan")

    return {
        "split": split,
        "clause_tol_m": tol,
        "clause_tol_mode": mode,
        "turn_rule": oracle.get("turn_rule"),
        "offtrack_m": off,
        "n_episodes": n_ep,
        "n_steps": n_steps,
        "offtrack_episodes": len(offtrack_ids),
        "offtrack_episode_ids": offtrack_ids,
        "R1": {
            lv: {
                "budget_m": budget["r1_budget"][lv],
                "budget_steps": steps_budget[lv] if steps_budget else None,
                "episode_fp_m": rate(r1_ep[lv]["m"], n_ep),
                "step_rate_m": rate(r1_st[lv]["m"], n_steps),
                "episode_fp_steps": rate(r1_ep[lv]["steps"], n_ep)
                if steps_budget
                else None,
                "step_rate_steps": rate(r1_st[lv]["steps"], n_steps)
                if steps_budget
                else None,
                "episodes_by_type": dict(r1_by_type[lv]),
            }
            for lv in R1_LEVELS
        },
        "R5": {
            f"long>{f}x": {"episode_fp": rate(r5_ep[f], n_ep), "long": r5_ep[f]}
            for f in R5_FACTORS
        },
        "R6": {
            f"radius={rad},gap={gap}": {
                "episode_fp": rate(r6_ep[(rad, gap)], n_ep),
                "loc_rate": rate(r6_locs[(rad, gap)], n_locs),
            }
            for rad, gap in R6_GRID
        },
        "r6_new_vs_task2_flag_mismatches": r6_old_mismatch,
    }


def rules_v0(budget: dict, cal: dict, provisional: bool) -> dict[str, Any]:
    r6_key = f"radius={R6_CHOSEN[0]},gap={R6_CHOSEN[1]}"
    return {
        "version": "v0",
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "R1": {
            "level": R1_CHOSEN,
            "type": budget["r1_budget"][R1_CHOSEN],
            "steps": budget["r1_budget_steps"][R1_CHOSEN]
            if budget.get("r1_budget_steps")
            else None,
            "all_levels": {
                "type": budget["r1_budget"],
                "steps": budget.get("r1_budget_steps"),
            },
            "charge": "a forward step is charged to the clause the agent was in when it started",
        },
        "R5": {
            "side": "long_only",
            "long_factor": R5_FACTORS[0],
            "long_factors": list(R5_FACTORS),
            "median_by_nclauses": {
                n: v["median_ref_arclen_m"]
                for n, v in budget["r5_median_by_n_clauses"].items()
            },
            "median_source": "train ref_path horizontal arc length, grouped by clause count",
        },
        "R6": {
            "radius_m": R6_CHOSEN[0],
            "gap_forward_steps": R6_CHOSEN[1],
            "definition": "same clause; position after a forward step within radius_m of a position "
            "recorded >= gap_forward_steps forward steps earlier; turn actions do not count",
        },
        "offtrack_m": cal["offtrack_m"],
        "clause_tol": {"mode": cal["clause_tol_mode"], "m": cal["clause_tol_m"]},
        "turn_rule": cal["turn_rule"],
        "provenance": {
            "written_by": "tools/03_codex_baseline_oracle/py/t30_fixes.py",
            "budget_split": budget["split"],
            "budget_n_with_gt": budget["n_with_gt"],
            "calibration_split": cal["split"],
            "calibration_n_episodes": cal["n_episodes"],
            "provisional": provisional,
            "fp_episode": {
                "R1": cal["R1"][R1_CHOSEN]["episode_fp_m"],
                "R5": {k: v["episode_fp"] for k, v in cal["R5"].items()},
                "R6": cal["R6"][r6_key]["episode_fp"],
            },
            "task2_out": str(t3.env_path("OUT")),
        },
    }


# ── report ─────────────────────────────────────────────────────────────────


def type_rows(before: Counter, after: Counter) -> list[list[Any]]:
    tb, ta = sum(before.values()) or 1, sum(after.values()) or 1
    return [
        [
            t,
            before[t],
            f"{100 * before[t] / tb:.1f}%",
            after[t],
            f"{100 * after[t] / ta:.1f}%",
        ]
        for t in CLAUSE_TYPES
    ]


def main() -> int:
    t3.md3_path("T3.0")
    t3.reset_issues3("T3.0")
    tol, mode, rule = t3.clause_tol_m(), t3.clause_tol_mode(), t3.turn_rule()
    rule_split = os.environ.get("RULE_SPLIT", "val_unseen")
    lines = [
        "## T3.0 修正：单调率 / 终点命中率 / 类型分布 / 预算表 / 误报率 / offtrack episode 列表",
        "",
        (
            f"设置：CLAUSE_TOL_MODE=`{mode}`（中间子句 0 m，末尾零长度子句 {tol} m），TURN_RULE=`{rule}`"
            f"（turn 触发词加入 left | right，排除条件不变），OFFTRACK_M={t3.offtrack_m()}；"
            f"R6 按前进步计（转向不计），R5 只保留过长侧。任务 #2 输出目录 `{t3.env_path('OUT')}`。"
        ),
        "",
    ]
    ok = True
    # (2) T2.2 rerun: types
    typed: dict[str, dict] = {}
    lines += ["**类型分布（修正前 → 修正后，T2.2 重跑）**", ""]
    for split in SPLITS:
        res = retype(split)
        if res is None:
            lines += [f"- {split}: 任务 #2 的 oracle_progress 缺失，跳过", ""]
            continue
        data, before, after, moved = res
        typed[split] = data
        lines += [
            f"{split}（{data['n_episodes']} episodes）：",
            "",
            t3.md_table(
                ["type", "before n", "before %", "after n", "after %"],
                type_rows(before, after),
            ),
            "",
            f"改动：{dict(moved) or '无'}",
            "",
        ]
    if "rand100" not in typed:
        t3.write_md("T3.0", [*lines, "SKIP：rand100 oracle 缺失，无法继续。"])
        return 2
    # (1) T2.3 rerun
    gt_r100 = t3.load_gt("rand100")
    if gt_r100 is None:
        t3.write_md("T3.0", [*lines, "SKIP：rand100 GT 不可读，无法继续。"])
        return 2
    runs = part_replay(typed["rand100"], gt_r100)
    t3.dump_json(t3.out3_dir() / "t30_replay_rand100.json", runs)
    chosen_key = f"{mode}/{tol}"
    chosen = runs[chosen_key]
    replay_ok = (
        chosen["monotone_share"] == 1.0
        and chosen["final_hit_share"] == 1.0
        and chosen["offtrack_episodes"] == 0
    )
    ok &= replay_ok
    lines += [
        "**rand100 GT 回放（T2.3 重跑）**：目标 单调率 100% / 终点命中 100% / offtrack 0",
        "",
        t3.md_table(
            [
                "mode / tol",
                "n",
                "raw idx 单调",
                "final idx = last",
                "offtrack episodes",
                "max dist P95 (m)",
            ],
            [
                [
                    f"{k}{' (chosen)' if k == chosen_key else ''}",
                    r["n"],
                    t3.pct(r["monotone_share"]),
                    t3.pct(r["final_hit_share"]),
                    r["offtrack_episodes"],
                    f"{r['max_dist_p95']:.3f}",
                ]
                for k, r in runs.items()
            ],
        ),
        "",
    ]
    for kind, items in chosen["fails"].items():
        if items:
            lines += [
                f"不满足条目 — {kind}（{chosen_key}）:",
                "```",
                *[str(x) for x in items[:20]],
                "```",
                "",
            ]
    lines.append(f"判据 1（{chosen_key}）：{'满足' if replay_ok else '不满足'}")
    lines.append("")
    # T2.4 rerun
    budget_split = os.environ.get("BUDGET_SPLIT", "train")
    if budget_split not in typed:
        t3.issue3(
            f"T3.0: no oracle_progress_{budget_split}; budgets and rules_v0.json not produced"
        )
        t3.write_md(
            "T3.0", [*lines, f"SKIP：{budget_split} oracle 缺失，无法计算预算。"]
        )
        return 2
    gt_train = t3.load_gt(budget_split)
    budget = part_budget(typed[budget_split], gt_train)
    t3.dump_json(t3.out3_dir() / "clause_budget_train.json", budget)
    hdr = ["type", "n", *[f"P{q}" for q in QS]]

    def rows(tab: dict) -> list[list[Any]]:
        return [[t, tab[t]["n"], *[tab[t][f"P{q}"] for q in QS]] for t in CLAUSE_TYPES]

    lines += [
        f"**预算表（T2.4 重跑，split `{budget_split}`，{budget['n_episodes']} episodes，{budget['n_with_gt']} 个有 GT）**",
        "",
        "子句长度 `length_m`（m）— R1 距离预算：",
        "",
        t3.md_table(hdr, rows(budget["length_m"])),
        "",
    ]
    if budget["n_with_gt"]:
        lines += [
            "GT 前进步数（转向不计）— R1 步数预算：",
            "",
            t3.md_table(hdr, rows(budget["gt_forward_steps"])),
            "",
        ]
    else:
        lines += [
            f"GT 前进步数：{budget_split} GT 不可读，步数预算未生成（rules_v0.json 中 R1.steps = null）。",
            "",
        ]
        t3.issue3(
            f"T3.0: {budget_split} GT unreadable here; R1 step budgets missing from rules_v0.json"
        )
    lines += [
        "R5 中位数（ref 弧长，按子句数）："
        + ", ".join(
            f"{n}: {v['median_ref_arclen_m']}"
            for n, v in budget["r5_median_by_n_clauses"].items()
        ),
        "",
    ]
    # T2.5 rerun
    cal_splits = [s for s in dict.fromkeys([rule_split, "rand100"]) if s in typed]
    cals: dict[str, dict] = {}
    for split in cal_splits:
        gt = t3.load_gt(split)
        if gt is None:
            lines += [f"**误报率（{split}）**：GT 不可读，跳过。", ""]
            continue
        cal = part_rules(typed[split], gt, t3.load_episodes(split), budget, split)
        cals[split] = cal
        t3.dump_json(t3.out3_dir() / f"rule_calibration_{split}.json", cal)
        lines += [
            f"**误报率表（T2.5 重跑，split `{split}`，{cal['n_episodes']} 条 GT，{cal['n_steps']} 前进步）**",
            "",
            "R1 预算超支（m = 距离预算；steps = 前进步预算）：",
            "",
            t3.md_table(
                [
                    "档",
                    "budget_m[type]",
                    "episode 误报 (m)",
                    "按步 (m)",
                    "episode 误报 (steps)",
                    "触发 episode 按类型",
                ],
                [
                    [
                        lv,
                        ", ".join(f"{t} {v['budget_m'][t]}" for t in CLAUSE_TYPES),
                        t3.pct(v["episode_fp_m"]),
                        t3.pct(v["step_rate_m"]),
                        t3.pct(v["episode_fp_steps"]),
                        dict(v["episodes_by_type"]),
                    ]
                    for lv, v in cal["R1"].items()
                ],
            ),
            "",
            "R5 停止路程过长（总水平路程 > factor × train 同子句数 ref 弧长中位数）：",
            "",
            t3.md_table(
                ["档", "episode 误报", "过长 n"],
                [[k, t3.pct(v["episode_fp"]), v["long"]] for k, v in cal["R5"].items()],
            ),
            "",
            "R6 回环（新口径：同一子句内，前进步后的位置进入 ≥gap 个前进步之前位置的 radius 内；转向不计步）：",
            "",
            t3.md_table(
                ["参数", "episode 误报", "按位置触发率"],
                [
                    [k, t3.pct(v["episode_fp"]), t3.pct(v["loc_rate"])]
                    for k, v in cal["R6"].items()
                ],
            ),
            "",
            (
                f"新口径与任务 #2 `r6_flags` 在 GT 上逐位置比较：{cal['r6_new_vs_task2_flag_mismatches']} 处不一致"
                "（GT 的 locations 本来就每前进步一点，差别只在 harness 侧含转向动作的轨迹上）。"
            ),
            "",
            f"offtrack（> {cal['offtrack_m']} m）GT episode：{cal['offtrack_episodes']} 条 → "
            + (
                str(cal["offtrack_episode_ids"])
                if cal["offtrack_episode_ids"]
                else "无"
            ),
            "",
        ]
    if rule_split not in cals:
        provisional = True
        if "rand100" in cals:
            cal = cals["rand100"]
            lines += [
                f"**rules_v0.json 为临时版**：校准 split `{rule_split}` 的 GT 不在本机，误报率来自 rand100（100 条）。",
                "",
            ]
            t3.issue3(
                f"T3.0: {rule_split} GT unreadable here; rules_v0.json calibrated on rand100 only (provisional)"
            )
        else:
            t3.write_md(
                "T3.0", [*lines, "SKIP：没有任何可用的 GT split，rules_v0.json 未写。"]
            )
            return 2
    else:
        provisional = False
        cal = cals[rule_split]
    r6_key = f"radius={R6_CHOSEN[0]},gap={R6_CHOSEN[1]}"
    fp_ok = {
        "R1": cal["R1"][R1_CHOSEN]["episode_fp_m"] < FP_TARGET,
        "R5": cal["R5"][f"long>{R5_FACTORS[0]}x"]["episode_fp"] < FP_TARGET,
        "R6": cal["R6"][r6_key]["episode_fp"] < FP_TARGET,
    }
    for rule_name, good in fp_ok.items():
        if not good:
            t3.issue3(
                f"T3.0: {rule_name} at the task-book setting has episode FP >= 5 % on {cal['split']}"
            )
    ok &= all(fp_ok.values())
    rules = rules_v0(budget, cal, provisional)
    old = t3.load_rules_v0()
    if old and "commit_oracle" in old:
        rules["commit_oracle"] = old[
            "commit_oracle"
        ]  # T3.6 may have run before a T3.0 rerun
    t3.save_rules_v0(rules)
    lines += [
        (
            f"**rules_v0.json**（`{t3.out3_dir() / 'rules_v0.json'}`）：R1 = {R1_CHOSEN}，R5 过长侧 factor {R5_FACTORS}，"
            f"R6 = {R6_CHOSEN[0]} m / {R6_CHOSEN[1]} 前进步，offtrack {cal['offtrack_m']} m，clause_tol {mode}/{tol}，turn_rule {rule}；"
            f"校准 split `{cal['split']}`{'（临时）' if provisional else ''}。"
            f"选定档的 episode 误报：R1 {t3.pct(cal['R1'][R1_CHOSEN]['episode_fp_m'])}，"
            f"R5 {t3.pct(cal['R5'][f'long>{R5_FACTORS[0]}x']['episode_fp'])}，R6 {t3.pct(cal['R6'][r6_key]['episode_fp'])}。"
        ),
        "",
        f"判据 2（选定档 episode 误报 < 5%）：{'满足' if all(fp_ok.values()) else '不满足 ' + str(fp_ok)}",
        "",
        f"结论：{'PASS' if ok and not provisional else ('SKIP（临时校准）' if provisional else 'FAIL')}",
    ]
    t3.write_md("T3.0", lines)
    if provisional:
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
