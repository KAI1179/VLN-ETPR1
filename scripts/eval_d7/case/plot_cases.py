#!/usr/bin/env python3
"""Render paired D7 evaluation trajectories on an MP3D connectivity diagram (CPU)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from tap import Tap

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")


class Args(Tap):
    base_dir: str = str(REPO / "data/logs/checkpoints/d7_case_base_val_unseen/trajectories")
    d_dir: str = str(REPO / "data/logs/checkpoints/d7_case_D_val_unseen/trajectories")
    selection: str = str(REPO / "reports/figs/d7_case/selection.json")
    output: str = str(REPO / "reports/figs/d7_case")
    divergence_m: float = 2.0
    candidate_match_m: float = 1.0
    route_match_m: float = 1.0


def horizontal(positions: list | np.ndarray) -> np.ndarray:
    return np.asarray(positions, dtype=float)[..., [0, 2]]


def route_distance(point: np.ndarray, route: np.ndarray) -> float:
    """Minimum distance to a reference path segment, in metres."""
    if len(route) == 1:
        return float(np.linalg.norm(point - route[0]))
    start = route[:-1]
    delta = route[1:] - start
    length_sq = np.sum(delta * delta, axis=1)
    fraction = np.divide(
        np.sum((point - start) * delta, axis=1), length_sq,
        out=np.zeros_like(length_sq), where=length_sq != 0,
    ).clip(0, 1)
    return float(np.linalg.norm(point - (start + fraction[:, None] * delta), axis=1).min())


def decision_pair(base_steps: list[dict], d_steps: list[dict], threshold: float) -> tuple[int, int] | None:
    """Align ordered decision positions, then locate the last shared pre-divergence pair.

    Dynamic time warping allows differing high-level step counts. The result is
    a spatial correspondence, not a claim that the internal graph states match.
    """
    a = np.asarray([s["position"] for s in base_steps], dtype=float)
    b = np.asarray([s["position"] for s in d_steps], dtype=float)
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    costs = np.full((len(a) + 1, len(b) + 1), np.inf)
    costs[0, 0] = 0
    predecessor = {}
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            options = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
            previous = min(options, key=lambda pair: costs[pair])
            costs[i, j] = distances[i - 1, j - 1] + costs[previous]
            predecessor[i, j] = previous
    alignment = []
    current = (len(a), len(b))
    while current != (0, 0):
        i, j = current
        alignment.append((i - 1, j - 1))
        current = predecessor[current]
    alignment.reverse()
    last_shared = None
    for i, j in alignment:
        if distances[i, j] > threshold:
            return last_shared
        last_shared = (i, j)
    return None


def candidate_rows(base: dict, d: dict, gt: np.ndarray, tolerance: float, route_tolerance: float) -> list[dict]:
    """Join candidates by position, never by rollout-local ghost identifiers."""
    left, right = base["candidates"], d["candidates"]
    matches = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            distance = float(np.linalg.norm(np.asarray(a["position"]) - b["position"]))
            if distance <= tolerance:
                matches.append((distance, i, j))
    used_left, used_right = set(), set()
    paired = []
    for _, i, j in sorted(matches):
        if i not in used_left and j not in used_right:
            paired.append((left[i], right[j]))
            used_left.add(i)
            used_right.add(j)
    paired += [(a, None) for i, a in enumerate(left) if i not in used_left]
    paired += [(None, b) for j, b in enumerate(right) if j not in used_right]
    rows = []
    for number, (a, b) in enumerate(paired, 1):
        candidate = a if a is not None else b
        rows.append({
            "label": f"C{number}", "position": candidate["position"],
            "base_id": a["id"] if a else None, "d_id": b["id"] if b else None,
            "base_probability": a["probability"] if a else None,
            "d_probability": b["probability"] if b else None,
            "base_selected": a is not None and a["id"] == base["selected_id"],
            "d_selected": b is not None and b["id"] == d["selected_id"],
            "near_gt_route": route_distance(np.asarray(candidate["position"]), gt) <= route_tolerance,
        })
    rows.append({
        "label": "STOP", "position": None, "base_id": None, "d_id": None,
        "base_probability": base["stop_probability"],
        "d_probability": d["stop_probability"],
        "base_selected": base["stop"], "d_selected": d["stop"],
        "near_gt_route": None,
    })
    return rows


def background(axis, scene: str, start_node: str, start_y: float) -> list[dict]:
    entries = json.loads((REPO / f"pretrain_src/datasets/R2R/connectivity/{scene}_connectivity.json").read_text())
    positions = np.asarray([[entry["pose"][3], -entry["pose"][7]] for entry in entries])
    for i, entry in enumerate(entries):
        if not entry["included"]:
            continue
        for j in range(i + 1, len(entries)):
            if entries[j]["included"] and entry["unobstructed"][j]:
                axis.plot(*positions[[i, j]].T, color="0.85", lw=0.8, zorder=0)
    included = np.asarray([entry["included"] for entry in entries])
    axis.scatter(*positions[included].T, s=5, c="0.8", zorder=0)
    camera_offset = next(entry["pose"][11] for entry in entries if entry["image_id"] == start_node) - start_y
    return [{
        "id": entry["image_id"],
        "position": [entry["pose"][3], entry["pose"][11] - camera_offset, -entry["pose"][7]],
        "degree": sum(edge and other["included"] for edge, other in zip(entry["unobstructed"], entries)),
    } for entry in entries if entry["included"]]


def probability(value: float | None, selected: bool) -> str:
    return "—" if value is None else f"{value:.3f}" + (" *" if selected else "")


def main() -> None:
    args = Args().parse_args()
    with gzip.open(REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz", "rt") as stream:
        episodes = {str(ep["episode_id"]): ep for ep in json.load(stream)["episodes"]}
    selection = json.loads(Path(args.selection).read_text())
    if isinstance(selection, dict):
        selected = selection["positive"] + selection["counterexamples"]
    else:
        selected = selection
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotations = {}
    annotation_path = REPO / "pretrain_src/datasets/R2R/annotations/pretrain_R2R_RxR/R2R_val_unseen_enc_xlmr.jsonl"
    for line in annotation_path.read_text().splitlines():
        item = json.loads(line)
        annotations[str(item["episode_id"])] = item
    index = ["# D7 决策案例", "", "底图为 MP3D 原始连通图简图，不是可通行区域栅格。无 FPV 渲染。", "", "决策步用三维轨迹位置 DTW 对齐；候选按三维位置匹配，不按 ghost ID。两个模型的内部图状态未强制相同，概率仅作各自状态的定性对照。GT 标记为候选距 GT 路径三维距离不超过 1 m，不等同于专家动作标签。", ""]
    for row in selected:
        key = str(row["episode_id"]) if isinstance(row, dict) else str(row)
        base = json.loads((Path(args.base_dir) / f"{key}.json").read_text())
        d = json.loads((Path(args.d_dir) / f"{key}.json").read_text())
        episode = episodes[key]
        gt_3d = np.asarray(episode["reference_path"], dtype=float)
        gt = horizontal(gt_3d)
        pair = decision_pair(base["steps"], d["steps"], args.divergence_m)
        fig, (axis, panel) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.6, 1]})
        scene = Path(episode["scene_id"]).stem
        graph = background(axis, scene, annotations[key]["path"][0], episode["start_position"][1])
        axis.plot(*gt.T, "--", color="0.35", lw=2, label="GT")
        for dump, color, name in ((base, "#d62728", "Baseline"), (d, "#1f77b4", "D")):
            positions = horizontal(dump.get("trajectory", [step["position"] for step in dump["steps"]]))
            axis.plot(*positions.T, color=color, lw=2, label=name)
        axis.scatter(*gt[0], marker="o", s=70, color="green", label="Start", zorder=5)
        goal = horizontal(episode["goals"][0]["position"])
        axis.scatter(*goal, marker="*", s=120, color="black", label="Goal", zorder=5)
        axis.add_patch(Circle(goal, 3, fill=False, color="black", linestyle=":"))
        rows = []
        junction = None
        subtitle = "No >2 m divergence found"
        if pair is not None:
            i, j = pair
            a, b = base["steps"][i], d["steps"][j]
            point = (horizontal(a["position"]) + horizontal(b["position"])) / 2
            axis.add_patch(Circle(point, 0.8, fill=False, color="purple", lw=2, zorder=6))
            rows = candidate_rows(a, b, gt_3d, args.candidate_match_m, args.route_match_m)
            nearest = min(graph, key=lambda node: float(np.linalg.norm(np.asarray(node["position"]) - np.asarray(a["position"]))))
            distance = float(np.linalg.norm(np.asarray(nearest["position"]) - np.asarray(a["position"])))
            junction = {"node": nearest["id"], "degree": nearest["degree"], "distance_m": distance, "confirmed": nearest["degree"] >= 3 and distance <= 1.0}
            for candidate in rows:
                if candidate["position"] is None:
                    continue
                xy = horizontal(candidate["position"])
                axis.scatter(*xy, s=25, color="purple")
                axis.annotate(candidate["label"], xy, fontsize=8)
            subtitle = f"Matched decision: baseline step {a['step']}, D step {b['step']}"
        panel.axis("off")
        if rows:
            table = panel.table(
                cellText=[[r["label"], probability(r["base_probability"], r["base_selected"]), probability(r["d_probability"], r["d_selected"]), "—" if r["near_gt_route"] is None else "yes" if r["near_gt_route"] else "no"] for r in rows],
                colLabels=["Candidate", "Base p", "D p", "Near GT"], loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
        panel.set_title(subtitle + "\n* = selected; unmatched candidate = —", fontsize=10)
        instruction = episode["instruction"]["instruction_text"]
        fig.suptitle(f"Episode {key} · {scene}\n" + textwrap.fill(instruction, 110), fontsize=11)
        axis.set_aspect("equal")
        axis.set_xlabel("Habitat x (m)")
        axis.set_ylabel("Habitat z (m)")
        axis.legend(fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{key}.{suffix}", dpi=180)
        plt.close(fig)
        (output / f"{key}_decision.json").write_text(json.dumps({"episode_id": key, "decision_pair": pair, "junction": junction, "candidates": rows}, indent=2) + "\n")
        qualifies = not base["final"]["oracle_success"] and bool(d["final"]["success"]) and junction is not None and junction["confirmed"]
        status = "正例条件重评确认" if qualifies else "不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例"
        index += [f"## Episode {key}", "", status, "", instruction, "", f"![{key}]({key}.png)", "", subtitle, "", f"岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{junction}", "", f"Baseline final: {base['final']}; D final: {d['final']}", "", "| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |", "|---|---|---|---:|---:|---|"]
        index += [f"| {r['label']} | {r['base_id']} | {r['d_id']} | {probability(r['base_probability'], r['base_selected'])} | {probability(r['d_probability'], r['d_selected'])} | {r['near_gt_route']} |" for r in rows]
        if pair is not None:
            index += ["", f"STOP 是否为强制停止：baseline={a['forced_stop']}，D={b['forced_stop']}。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。", ""]
        index += ["", "FPV：未渲染；仅 CPU 连通图绘图。", ""]
    (output / "index.md").write_text("\n".join(index) + "\n")
    print(output / "index.md")


if __name__ == "__main__":
    main()
