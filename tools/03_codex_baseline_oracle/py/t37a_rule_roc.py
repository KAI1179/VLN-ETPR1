"""T3.7(a): contradiction rules replayed on REAL agent trajectories (T3.3 export) — true-positive check.

usage: t37a_rule_roc.py <traj_dir>   (files ep<episode_id>.json from t33_export_traj.py)
Per rule (R1, R5, R6, offtrack, commit-oracle, any): episode-level TPR (failed episodes with >= 1 trigger), FPR
(successful episodes with >= 1 trigger), first-trigger step, delay after the first off-route step (> offtrack_m
from the reference polyline, the proxy for t*). Parameter sweeps: R1 P90/P95/P99, R6 radius {0.5,1.0} x gap {5,10}.
Writes $OUT3/t37a_rule_roc.json and md/T3.7a.md. Exit 0; 2 (SKIP) when no trajectories; small n is stated.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
import oracles as O
from t3common import load_oracle3, load_rules_v0, md3_path, out3_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_oracles" / "py"))
from common import ClauseTrack  # noqa: E402

RULES = ("R1", "R5", "R6", "offtrack", "commit", "any")


def make_rules(
    rules: dict[str, Any],
    r1_level: str,
    r6_radius: float,
    r6_gap: int,
    ep: dict[str, Any],
) -> tuple[Any, Any, Any]:
    budget = (
        rules["R1"]["type"]
        if r1_level == rules["R1"].get("level", "P99")
        else rules["R1"]["all_levels"][r1_level]["type"]
    )
    r1 = O.BudgetRule(budget_m=budget, budget_steps=None)
    r6 = O.LoopDetector(r6_radius, r6_gap)
    co = rules.get("commit_oracle") or {}
    commit = O.CommitOracle(
        np.asarray(ep["ref_path_xz"]),
        O.CommitParams(
            theta_deg=co.get("theta_deg", 45.0),
            advance_m=co.get("advance_m", 1.0),
            vertex_radius_m=co.get("vertex_radius_m", 2.0),
            turn_deg=co.get("turn_deg", 30.0),
            ontrack_m=co.get("ontrack_m", 1.0),
            sustained=co.get("sustained", True),
        ),
    )
    return r1, r6, commit


def replay(
    traj: dict[str, Any],
    ep: dict[str, Any],
    rules: dict[str, Any],
    r1_level: str,
    r6_radius: float,
    r6_gap: int,
) -> dict[str, Any]:
    tol, mode = rules["clause_tol"]["m"], rules["clause_tol"]["mode"]
    track = ClauseTrack(ep, tol_m=tol, offtrack_m=rules["offtrack_m"], tol_mode=mode)
    r1, r6, commit = make_rules(rules, r1_level, r6_radius, r6_gap, ep)
    first: dict[str, int | None] = dict.fromkeys(RULES)
    prev_idx, first_off, total = -1, None, 0.0
    prev_xz = None
    for i, s in enumerate(traj["steps"]):
        xz = np.asarray([s["x"], s["z"]])
        forward = int(s.get("action", 1)) == 1
        if prev_xz is not None:
            total += float(np.linalg.norm(xz - prev_xz))
        prev_xz = xz
        h = track.locate(xz, prev_idx, monotone=True)
        prev_idx = h.idx
        if h.offtrack and first_off is None:
            first_off = i
        flags = {
            "R1": r1.step(xz, h.idx, track.types[h.idx], forward),
            "R6": r6.step(xz, h.idx, forward),
            "offtrack": h.offtrack,
            "commit": commit.step(xz, float(s.get("yaw", 0.0))),
        }
        for r, f in flags.items():
            if f and first[r] is None:
                first[r] = i
    n_cl = str(len(ep["clauses"]))
    med = (rules["R5"].get("median_by_nclauses") or {}).get(n_cl)
    if med and total > rules["R5"]["long_factor"] * float(med):
        first["R5"] = len(traj["steps"]) - 1
    hits = [v for k, v in first.items() if k != "any" and v is not None]
    first["any"] = min(hits) if hits else None
    return {
        "first": first,
        "first_off": first_off,
        "n_steps": len(traj["steps"]),
        "path_m": total,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fails = [r for r in rows if not r["success"]]
    succ = [r for r in rows if r["success"]]
    out: dict[str, Any] = {
        "n": len(rows),
        "n_fail": len(fails),
        "n_success": len(succ),
        "rules": {},
    }
    for rule in RULES:
        tp = [r for r in fails if r["first"][rule] is not None]
        fp = [r for r in succ if r["first"][rule] is not None]
        delays = [
            r["first"][rule] - r["first_off"] for r in tp if r["first_off"] is not None
        ]
        out["rules"][rule] = {
            "tpr": len(tp) / len(fails) if fails else None,
            "fpr": len(fp) / len(succ) if succ else None,
            "n_tp": len(tp),
            "n_fp": len(fp),
            "delay_vs_first_offroute_steps": {
                "n": len(delays),
                "p50": st.median(delays) if delays else None,
                "p90": round(float(np.percentile(delays, 90)), 1) if delays else None,
            },
        }
    return out


def main(argv: list[str]) -> int:
    traj_dir = Path(argv[1]) if len(argv) > 1 else out3_dir() / "traj"
    md = md3_path("T3.7a")
    files = sorted(traj_dir.glob("ep*.json"))
    rules = load_rules_v0()
    oracle = load_oracle3("rand100")
    if not files or rules is None or oracle is None:
        md.write_text(
            "## T3.7(a) 规则 ROC\n\nSKIP：缺少轨迹（T3.3 未跑或基线未在 oracleES/none 上跑）、rules_v0.json 或 oracle。\n"
        )
        print("SKIP: trajectories / rules / oracle missing")
        return 2
    by_id = {int(e["episode_id"]): e for e in oracle["episodes"]}
    trajs = [json.loads(f.read_text()) for f in files]
    trajs = [t for t in trajs if int(t["episode_id"]) in by_id and t.get("steps")]
    configs = [
        (
            "v0",
            rules["R1"].get("level", "P99"),
            rules["R6"]["radius_m"],
            rules["R6"]["gap_forward_steps"],
        )
    ]
    for lvl in ("P90", "P95", "P99"):
        configs.append((
            f"R1={lvl}",
            lvl,
            rules["R6"]["radius_m"],
            rules["R6"]["gap_forward_steps"],
        ))
    for rad in (0.5, 1.0):
        for gap in (5, 10):
            configs.append((
                f"R6=({rad},{gap})",
                rules["R1"].get("level", "P99"),
                rad,
                gap,
            ))
    result: dict[str, Any] = {"n_traj": len(trajs), "configs": {}}
    for name, lvl, rad, gap in configs:
        if lvl != rules["R1"].get("level", "P99") and lvl not in (
            rules["R1"].get("all_levels") or {}
        ):
            continue
        rows = []
        for t in trajs:
            r = replay(t, by_id[int(t["episode_id"])], rules, lvl, rad, gap)
            rows.append({
                **r,
                "episode_id": t["episode_id"],
                "success": bool(t.get("success")),
            })
        result["configs"][name] = summarize(rows)
        if name == "v0":
            result["episodes"] = rows
    (out3_dir() / "t37a_rule_roc.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False)
    )
    v0 = result["configs"]["v0"]
    lines = [
        "## T3.7(a) 矛盾规则在真实轨迹上的真阳性",
        "",
        f"轨迹 {v0['n']} 条（失败 {v0['n_fail']}，成功 {v0['n_success']}）；样本小，只作方向判断。首个偏离步 = 到参考折线 > {rules['offtrack_m']} m 的首步（t* 代理）。",
        "",
        "| 规则（v0 参数） | TPR（失败集召回） | FPR（成功集误触发） | 触发延迟 vs 首个偏离步 P50 / P90（步） |",
        "|---|---|---|---|",
    ]
    for rule in RULES:
        s = v0["rules"][rule]
        d = s["delay_vs_first_offroute_steps"]

        def f(x: float | None) -> str:
            return "-" if x is None else f"{100 * x:.0f}%"

        lines.append(
            f"| {rule} | {f(s['tpr'])} ({s['n_tp']}/{v0['n_fail']}) | {f(s['fpr'])} ({s['n_fp']}/{v0['n_success']}) | {d['p50']} / {d['p90']} (n={d['n']}) |"
        )
    lines += [
        "",
        "参数扫描（TPR / FPR）：",
        "",
        "| 配置 | R1 | R6 | any |",
        "|---|---|---|---|",
    ]
    for name, s in result["configs"].items():
        if name == "v0":
            continue

        def g(r: str) -> str:
            return (
                "-"
                if s["rules"][r]["tpr"] is None
                else f"{100 * s['rules'][r]['tpr']:.0f}% / {100 * (s['rules'][r]['fpr'] or 0):.0f}%"
            )

        lines.append(f"| {name} | {g('R1')} | {g('R6')} | {g('any')} |")
    any_tpr = v0["rules"]["any"]["tpr"]
    verdict = (
        "无失败样本，无法判断"
        if any_tpr is None
        else (
            "任一规则召回 < 50%：任务 #4 前触发器改为验证器驱动（任务书 T3.7a 判据）"
            if any_tpr < 0.5
            else "召回 ≥ 50%，v0 触发器可进入 E-P"
        )
    )
    lines += ["", f"结论：{verdict}。逐集结果见 `t37a_rule_roc.json`。"]
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
