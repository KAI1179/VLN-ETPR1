#!/usr/bin/env python3
"""CPU-only D7 behavior tables; missing evaluations are explicitly pending."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
import sys

# The category extractor eagerly loads CLIP; keep that operation on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from tap import Tap  # noqa: E402

RUNS = REPO / "data/logs/checkpoints"
LABELS = ("成功", "曾到目标范围但失败", "从未到达")
METRICS = ("success", "oracle_success", "spl", "ndtw", "path_length", "steps_taken")
DIRECTION_PHRASES = (
    "turn around", "on your left", "on your right", "to the left", "to the right",
    "straight ahead", "left", "right", "straight", "forward", "backward", "upstairs", "downstairs",
)
DIRECTION_RE = re.compile(r"\b(?:" + "|".join(DIRECTION_PHRASES) + r")\b", re.I)


class Args(Tap):
    base_run: str = str(RUNS / "d7_base_val_unseen")
    b_run: str = str(RUNS / "d5_deploy_val_unseen")
    d_run: str = str(RUNS / "d7_kd_deploy_val_unseen")
    draws: int = 2000
    output: str = str(REPO / "reports/d7_behavior")


def load_run(directory: Path, expected: set[str]) -> dict:
    files = sorted((directory / "eval_results").glob("stats_ep_*_val_unseen_r*_w*.json"))
    if not files:
        return {"status": "pending", "directory": str(directory)}
    rows = {}
    for file in files:
        part = json.loads(file.read_text())
        if set(rows) & set(part):
            raise ValueError(f"Duplicate episode IDs in {file}")
        rows.update(part)
    if set(rows) != expected:
        raise ValueError(f"Incomplete/different episode set: {directory}, {len(rows)} vs {len(expected)}")
    return {"status": "complete", "files": [str(f) for f in files], "rows": rows}


def features() -> dict:
    from prior.grid_map._cognitive import extract_categories

    path = REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"
    with gzip.open(path, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    annotation_path = REPO / "pretrain_src/datasets/R2R/annotations/pretrain_R2R_RxR/R2R_val_unseen_enc_xlmr.jsonl"
    annotations = {}
    for line in annotation_path.read_text().splitlines():
        item = json.loads(line)
        if int(item["episode_id"]) >= 0:
            key = str(item["episode_id"])
            if key in annotations:
                raise ValueError(f"Duplicate original R2R annotation: {key}")
            annotations[key] = item
    graphs = {}
    result = {}
    for episode in episodes:
        key = str(episode["episode_id"])
        item = annotations[key]
        scene = Path(episode["scene_id"]).stem
        if scene != item["scan"]:
            raise ValueError(f"Scene mismatch: {key}")
        if scene not in graphs:
            entries = json.loads((REPO / f"pretrain_src/datasets/R2R/connectivity/{scene}_connectivity.json").read_text())
            graphs[scene] = {
                entry["image_id"]: {
                    "position": np.asarray(entry["pose"])[[3, 7, 11]],
                    "degree": sum(edge and neighbor["included"] for edge, neighbor in zip(entry["unobstructed"], entries)),
                } for entry in entries if entry["included"]
            }
        graph = graphs[scene]
        original_path = item["path"]
        length = sum(float(np.linalg.norm(graph[a]["position"] - graph[b]["position"])) for a, b in zip(original_path, original_path[1:]))
        instruction = episode["instruction"]["instruction_text"]
        objects, regions = extract_categories(instruction)
        result[key] = {
            "scene": scene, "landmarks": len(objects) + len(regions),
            "directions": len(DIRECTION_RE.findall(instruction)), "length": length,
            "junctions": sum(graph[node]["degree"] >= 3 for node in original_path),
        }
    if len(result) != 1839:
        raise ValueError(f"Expected 1839 English R2R episodes, got {len(result)}")
    return result


def paired(left: dict, right: dict, ids: list[str], feats: dict, draws: int) -> dict:
    a = np.asarray([[left[i][m] for m in METRICS] for i in ids], dtype=float)
    b = np.asarray([[right[i][m] for m in METRICS] for i in ids], dtype=float)
    delta = b - a
    groups = {}
    for index, key in enumerate(ids):
        groups.setdefault(feats[key]["scene"], []).append(index)
    scene_groups = [groups[scene] for scene in sorted(groups)]
    sums = np.asarray([delta[g].sum(axis=0) for g in scene_groups])
    counts = np.asarray([len(g) for g in scene_groups])
    samples = np.random.default_rng(0).integers(len(groups), size=(draws, len(groups)))
    bootstrap = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)[:, None]
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    scale = np.asarray([100, 100, 100, 100, 1, 1])
    def state(row: dict) -> int:
        return 0 if row["success"] else (1 if row["oracle_success"] else 2)
    transition = np.zeros((3, 3), dtype=int)
    for key in ids:
        transition[state(left[key]), state(right[key])] += 1
    strata = {}
    for label, value in (("landmarks", "地标数"), ("junctions", "岔路数")):
        for low, high, name in ((0, 1, "0–1"), (2, 3, "2–3"), (4, float("inf"), "≥4")):
            chosen = [i for i in ids if low <= feats[i][label] <= high]
            strata[f"{value}:{name}"] = chosen
    cuts = np.quantile([feats[i]["length"] for i in sorted(feats)], [1 / 3, 2 / 3])
    for number in range(3):
        strata[f"路径长度:T{number + 1}"] = [i for i in ids if np.searchsorted(cuts, feats[i]["length"], side="right") == number]
    output = {
        "n": len(ids), "scenes": len(groups), "transition": transition.tolist(),
        "metrics": {m: {"left": float(a[:, j].mean() * scale[j]), "right": float(b[:, j].mean() * scale[j]), "difference": float(delta[:, j].mean() * scale[j]), "ci": (intervals[:, j] * scale[j]).tolist()} for j, m in enumerate(METRICS)},
        "length_cutoffs_m": cuts.tolist(), "strata": {},
    }
    for name, subset in strata.items():
        if subset:
            indices = [ids.index(i) for i in subset]
            subgroups = [list(set(g) & set(indices)) for g in scene_groups]
            subgroups = [g for g in subgroups if g]
            ss = np.asarray([delta[g, 0].sum() for g in subgroups])
            nn = np.asarray([len(g) for g in subgroups])
            draws_idx = np.random.default_rng(0).integers(len(nn), size=(draws, len(nn)))
            ci = np.quantile(ss[draws_idx].sum(axis=1) / nn[draws_idx].sum(axis=1) * 100, [0.025, 0.975])
            output["strata"][name] = {"n": len(subset), "scenes": len(nn), "left_sr": float(a[indices, 0].mean() * 100), "right_sr": float(b[indices, 0].mean() * 100), "difference_pp": float(delta[indices, 0].mean() * 100), "ci": ci.tolist()}
        else:
            output["strata"][name] = {"n": 0, "status": "empty"}
    return output


def main() -> None:
    args = Args().parse_args()
    dataset = REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"
    with gzip.open(dataset, "rt") as stream:
        expected = {str(ep["episode_id"]) for ep in json.load(stream)["episodes"]}
    runs = {name: load_run(Path(path), expected) for name, path in (("baseline", args.base_run), ("B", args.b_run), ("D", args.d_run))}
    if any(run["status"] != "complete" for run in runs.values()):
        state = {k: {key: value for key, value in v.items() if key != "rows"} for k, v in runs.items()}
        Path(args.output + ".json").write_text(json.dumps({"status": "pending", "runs": state}, indent=2) + "\n")
        Path(args.output + ".md").write_text("# D7 T3 行为统计（CPU）\n\n【待评测】baseline或新的同协议D部署结果缺失，未计算配对转移表、分层差值或机制结论。未加载类别提取模型。\n\n```json\n" + json.dumps(state, indent=2) + "\n```\n\n历史D为control协议，不代替新的teleport部署评测；基线原目录的eval_results为空，不用汇总63.13反推逐episode数据。\n")
        print(args.output + ".md (pending missing evaluations)")
        return
    feats = features()
    ids = sorted(feats, key=int)
    if set(ids) != expected:
        raise ValueError("Feature episode IDs differ from evaluation set")
    result = {"definitions": {"landmarks": "extract_categories: unique object indices plus unique region indices; separate category spaces", "directions": list(DIRECTION_PHRASES), "junctions": "Original R2R path node occurrences of degree >=3; included unobstructed neighbors; endpoints included", "length": "Original R2R path length from 3D connectivity poses in metres", "bootstrap": f"Scene-cluster resampling {args.draws} draws, seed0; episode-weighted mean; percentile95%CI"}, "runs": {k: {key: value for key, value in v.items() if key != "rows"} for k, v in runs.items()}, "features": feats, "comparisons": {}}
    text = ["# D7 T3 行为统计（CPU）", "", "仅描述配对结果，不作机制推断。SR/OSR/SPL/nDTW及其差值按百分数/百分点；PL为米，steps为连续动作数。", "", "特征定义：" + json.dumps(result["definitions"], ensure_ascii=False), ""]
    for left, right in (("baseline", "B"), ("B", "D")):
        name = f"{left}→{right}"
        if runs[left]["status"] != "complete" or runs[right]["status"] != "complete":
            text += [f"## {name}", "", "【待评测】逐 episode 输入未齐全，未计算配对表。", ""]
            continue
        pair = paired(runs[left]["rows"], runs[right]["rows"], ids, feats, args.draws)
        result["comparisons"][name] = pair
        text += [f"## {name}（n={pair['n']}，场景={pair['scenes']}）", "", "| 起点类别→终点类别 | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|"]
        text += ["| " + label + " | " + " | ".join(map(str, row)) + " |" for label, row in zip(LABELS, pair["transition"])]
        text += ["", "| 指标 | 起点模型 | 终点模型 | 配对差 | 场景bootstrap95%CI |", "|---|---:|---:|---:|---|"]
        text += [f"| {metric} | {v['left']:.3f} | {v['right']:.3f} | {v['difference']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |" for metric, v in pair["metrics"].items()]
        text += ["", f"路径长度三分位切点：{pair['length_cutoffs_m']} m。", "", "| 分层 | n | 场景数 | 起点SR | 终点SR | 差值pp | 95%CI |", "|---|---:|---:|---:|---:|---:|---|"]
        for layer, v in pair["strata"].items():
            text.append(f"| {layer} | {v['n']} | {v['scenes']} | {v['left_sr']:.2f} | {v['right_sr']:.2f} | {v['difference_pp']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] |" if v["n"] else f"| {layer} | 0 | — | — | — | — | — |")
        text.append("")
    text += ["## 输入状态", "", "```json", json.dumps(result["runs"], ensure_ascii=False, indent=2), "```", "", "缺失输入保留为待评测；不以历史汇总百分数替代逐 episode 数据。D默认读取新的teleport部署重评，未使用历史control结果作同协议对照。"]
    Path(args.output + ".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    Path(args.output + ".md").write_text("\n".join(text) + "\n")
    print(args.output + ".md")


if __name__ == "__main__":
    main()
