"""T3.6: O-commit (branch deviation) oracle, calibrated on GT replays.

Turning vertices = reference-path vertices whose direction change exceeds 30 deg. The oracle
(oracles.CommitOracle) triggers when the agent, after leaving the 2 m disc of a turning vertex, keeps a
heading more than theta away from the GT next-segment tangent while advancing >= d. Any trigger on a GT
trajectory is a false positive. Sweeps theta in {30, 45, 60} deg x d in {0.5, 1.0, 1.5} m on COMMIT_SPLIT
(default val_unseen; rand100 is always run as well) and writes the tightest setting with episode FP < 2 %
into $OUT3/rules_v0.json under "commit_oracle".

GT heading: the GT files carry no heading, but they carry the action sequence; the yaw is replayed exactly
from start_rotation and the 15 deg turn actions (oracles.gt_steps), which the report cross-checks against
the GT motion direction. Also lists *.house / *_semantic.ply under MP3D_DIR (default /data/xukai/mp3d).
Outputs: $OUT3/t36_commit_<split>.json, md/T3.6.md. Exit 0 PASS, 1 no setting < 2 %, 2 SKIP (COMMIT_SPLIT
has no GT here; rules_v0.json then carries a provisional rand100 setting).
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import oracles as O
import t3common as t3

THETAS = (30.0, 45.0, 60.0)
ADVANCES = (0.5, 1.0, 1.5)
GRID = tuple(
    itertools.product(THETAS, ADVANCES)
)  # tightness order: small theta, small d first
FP_TARGET = 0.02
BASE = O.CommitParams()  # vertex_radius_m 2.0, turn_deg 30, ontrack_m 1.0, sustained


def sweep(split: str) -> dict[str, Any] | None:
    oracle, gt, eps = t3.load_oracle3(split), t3.load_gt(split), t3.load_episodes(split)
    if oracle is None or gt is None:
        return None
    fp_ep = dict.fromkeys(GRID, 0)
    fp_vertex = dict.fromkeys(GRID, 0)
    examples: dict[tuple[float, float], list[dict[str, Any]]] = {g: [] for g in GRID}
    n_ep = n_turn = n_ep_turn = 0
    head_err: list[float] = []
    for ep in oracle["episodes"]:
        g = gt.get(str(ep["episode_id"]))
        if g is None:
            continue
        e = (eps or {}).get(str(ep["episode_id"]))
        steps = O.gt_steps(
            g["locations"], g["actions"], e["start_rotation"] if e else None
        )
        ref = np.asarray(ep["ref_path_xz"], dtype=float)
        turns = O.turning_vertices(ref, BASE.turn_deg)
        n_ep += 1
        n_turn += len(turns)
        n_ep_turn += bool(turns)
        for a, b in zip(steps[:-1], steps[1:]):
            if b.forward and float(np.linalg.norm(b.xz - a.xz)) > 0.05:
                head_err.append(O.angle_deg(O.forward_xz(b.yaw), b.xz - a.xz))
        if not turns:
            continue
        for th, d in GRID:
            co = O.CommitOracle(ref, O.CommitParams(theta_deg=th, advance_m=d))
            for s in steps:
                co.step(s.xz, s.yaw)
            if co.triggered:
                fp_ep[(th, d)] += 1
                fp_vertex[(th, d)] += len(co.triggers)
                if len(examples[(th, d)]) < 10:
                    examples[(th, d)].append({
                        "episode_id": ep["episode_id"],
                        "trigger_steps": co.triggers,
                        "vertices": co.trigger_vertices,
                        "events": co.events,
                    })
    return {
        "split": split,
        "n_episodes": n_ep,
        "n_episodes_with_turning_vertex": n_ep_turn,
        "n_turning_vertices": n_turn,
        "heading_vs_motion_deg": {
            "median": round(float(np.median(head_err)), 3) if head_err else None,
            "P99": round(float(np.percentile(head_err, 99)), 3) if head_err else None,
        },
        "grid": {
            f"theta={th},d={d}": {
                "theta_deg": th,
                "advance_m": d,
                "episode_fp": round(fp_ep[(th, d)] / n_ep, 4) if n_ep else float("nan"),
                "fp_episodes": fp_ep[(th, d)],
                "vertex_fp": round(fp_vertex[(th, d)] / n_turn, 4)
                if n_turn
                else float("nan"),
                "fp_vertices": fp_vertex[(th, d)],
                "examples": examples[(th, d)],
            }
            for th, d in GRID
        },
    }


def pick(res: dict[str, Any]) -> tuple[float, float] | None:
    for th, d in GRID:
        if res["grid"][f"theta={th},d={d}"]["episode_fp"] < FP_TARGET:
            return th, d
    return None


def mp3d_listing() -> str:
    root = Path(os.environ.get("MP3D_DIR", "/data/xukai/mp3d")).expanduser()
    if not root.is_dir():
        return f"MP3D_DIR `{root}` 不存在或不可读（本机没有 MP3D）。"
    house = list(root.glob("**/*.house"))
    sem = list(root.glob("**/*_semantic.ply"))
    return (
        f"MP3D_DIR `{root}`：`*.house` {len(house)} 个，`*_semantic.ply` {len(sem)} 个"
        f"（例：{[str(p.relative_to(root)) for p in (house + sem)[:3]]}）。"
    )


def main() -> int:
    t3.md3_path("T3.6")
    t3.reset_issues3("T3.6")
    cal_split = os.environ.get("COMMIT_SPLIT", "val_unseen")
    lines = [
        "## T3.6 O-承诺：参数扫描误报率表、选定参数、MP3D 语义标注有无",
        "",
        (
            f"拐点 = 参考路径相邻两段方向变化 > {BASE.turn_deg:g}°；触发 = 离开拐点 {BASE.vertex_radius_m:g} m 圆后，"
            f"朝向与 GT 下一段切向夹角 > θ 且（持续偏离中）前进 ≥ d；智能体投影到拐点下一个顶点之后且距路线 ≤ {BASE.ontrack_m:g} m 时窗口关闭。"
            "GT 朝向由 start_rotation 与 15° 转向动作精确回放（oracles.gt_steps）。实现：`oracles.CommitOracle`。"
        ),
        "",
    ]
    results: dict[str, dict[str, Any]] = {}
    for split in dict.fromkeys([cal_split, "rand100"]):
        res = sweep(split)
        if res is None:
            lines += [f"- {split}：oracle 或 GT 不可读，跳过。", ""]
            continue
        results[split] = res
        t3.dump_json(t3.out3_dir() / f"t36_commit_{split}.json", res)
        h = res["heading_vs_motion_deg"]
        lines += [
            (
                f"**{split}**：{res['n_episodes']} 条 GT，{res['n_episodes_with_turning_vertex']} 条含拐点，"
                f"拐点 {res['n_turning_vertices']} 个；回放朝向与 GT 实际运动方向夹角 median {h['median']}°，P99 {h['P99']}°。"
            ),
            "",
            t3.md_table(
                [
                    "θ (deg)",
                    "d (m)",
                    "episode 误报",
                    "误报 episode n",
                    "拐点误报率",
                    "误报拐点 n",
                ],
                [
                    [
                        v["theta_deg"],
                        v["advance_m"],
                        t3.pct(v["episode_fp"]),
                        v["fp_episodes"],
                        t3.pct(v["vertex_fp"]),
                        v["fp_vertices"],
                    ]
                    for v in res["grid"].values()
                ],
            ),
            "",
        ]
    if not results:
        t3.write_md("T3.6", [*lines, "SKIP：没有可用的 split。"])
        return 2
    provisional = cal_split not in results
    src = results.get(cal_split) or results["rand100"]
    choice = pick(src)
    if provisional:
        t3.issue3(
            f"T3.6: {cal_split} GT unreadable here; commit_oracle in rules_v0.json calibrated on rand100 only (provisional)"
        )
    if choice is None:
        t3.issue3(
            f"T3.6: no (theta, d) setting reaches episode FP < 2 % on {src['split']}"
        )
        lines += [f"没有一档在 `{src['split']}` 上误报 < 2%。", ""]
    else:
        th, d = choice
        v = src["grid"][f"theta={th},d={d}"]
        rules = t3.load_rules_v0() or {
            "version": "v0",
            "provenance": {"note": "created by t36 before T3.0"},
        }
        rules["commit_oracle"] = {
            "theta_deg": th,
            "advance_m": d,
            "vertex_radius_m": BASE.vertex_radius_m,
            "turn_deg": BASE.turn_deg,
            "ontrack_m": BASE.ontrack_m,
            "sustained": BASE.sustained,
            "fp_episode": v["episode_fp"],
            "fp_vertex": v["vertex_fp"],
            "calibration_split": src["split"],
            "provisional": provisional,
            "tightness_order": "ascending (theta, d); first with episode FP < 0.02",
        }
        t3.save_rules_v0(rules)
        lines += [
            (
                f"**选定**（误报 < 2% 的最紧一档，紧度按 (θ, d) 升序）：θ = {th:g}°，d = {d:g} m，"
                f"episode 误报 {t3.pct(v['episode_fp'])}，拐点误报 {t3.pct(v['vertex_fp'])}，split `{src['split']}`"
                f"{'（临时，val_unseen GT 不在本机）' if provisional else ''}；已写入 rules_v0.json `commit_oracle`。"
            ),
            "",
        ]
        if v["examples"]:
            lines += [
                "该档误报样例（episode_id, 触发步, 拐点, 事件）：",
                "```",
                *[
                    str({
                        k: x[k]
                        for k in ("episode_id", "trigger_steps", "vertices", "events")
                    })
                    for x in v["examples"][:5]
                ],
                "```",
                "",
            ]
    lines += [mp3d_listing(), ""]
    status = "SKIP（临时校准）" if provisional else ("PASS" if choice else "FAIL")
    lines.append(f"结论：{status}")
    t3.write_md("T3.6", lines)
    if provisional:
        return 2
    return 0 if choice else 1


if __name__ == "__main__":
    sys.exit(main())
