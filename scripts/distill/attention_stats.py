#!/usr/bin/env python3
"""D9 CPU-only, prespecified attention summaries and uncomposited case assets."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.stats import spearmanr
from tap import Tap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MODELS = ("base", "try5", "kd")
KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"


class Args(Tap):
    dump_root: str = str(REPO / "data/logs/d9_dump")
    output: str = str(REPO / "reports/d9")
    cases_output: str = str(REPO / "reports/figs/d9_case")
    draws: int = 2000


def word_attention(row, field, index):
    """Sum pieces before normalising; retain special-token mass separately."""
    attn = row[field][index].astype(np.float64)
    mapping = row["word_index"]
    words = np.zeros((len(attn), len(row["words"])), dtype=np.float64)
    for piece, word in enumerate(mapping):
        if word >= 0:
            words[:, int(word)] += attn[:, piece]
    total = words.sum(1, keepdims=True)
    words = np.divide(words, total, out=np.full_like(words, np.nan), where=total > 0)
    # The declared main statistic averages heads first, then renormalises words.
    mean = np.zeros(words.shape[1], dtype=np.float64)
    for piece, word in enumerate(mapping):
        if word >= 0:
            mean[int(word)] += attn[:, piece].mean()
    mean = mean / mean.sum() if mean.sum() else np.full_like(mean, np.nan)
    special_field = "txt_special" if field == "txt_attn" else "stop_txt_special"
    specials = row[special_field][index].astype(np.float64)
    return mean, words, specials


def text_values(row, field, index):
    mean, heads, specials = word_attention(row, field, index)
    positions = np.arange(len(mean), dtype=np.float64) / max(1, len(mean))

    def metrics(weights):
        if not np.isfinite(weights).all():
            return [np.nan, np.nan, np.nan]
        positive = weights > 0
        return [np.sum(weights * positions), -np.sum(weights[positive] * np.log(weights[positive])),
                np.sort(weights)[-3:].sum()]

    values = metrics(mean)
    spread = np.std([metrics(head) for head in heads], axis=0)
    output = dict(zip(("position", "entropy", "top3"), values))
    output.update({name + "_head_std": value for name, value in zip(("position", "entropy", "top3"), spread)})
    output.update(dict(zip(("cls", "sep", "padding"), specials.mean(0))))
    output["special"] = specials.mean(0).sum()
    output["flag"] = float(output["special"] > .8)
    return output


def point_distance(points, path):
    """Horizontal distance to a polyline, without treating it as isolated nodes."""
    path = np.asarray(path, dtype=np.float64)[:, [0, 2]]
    distance = np.full(len(points), np.inf)
    if len(path) == 1:
        return np.linalg.norm(points-path[0], axis=1)
    for a, b in zip(path[:-1], path[1:]):
        delta = b-a
        t = (points-a) @ delta / (delta @ delta) if delta @ delta else np.zeros(len(points))
        nearest = a + np.clip(t, 0, 1)[:, None] * delta
        distance = np.minimum(distance, np.linalg.norm(points-nearest, axis=1))
    return distance


def geometry(episode, row):
    origin = np.asarray(row["map_origin_xz"], dtype=np.float64)
    cell = (np.indices((100, 100)).transpose(1, 2, 0).reshape(-1, 2) + .5) * .5 + origin
    centers = (np.indices((10, 10)).transpose(1, 2, 0).reshape(-1, 2) + .5) * 5 + origin
    return cell, centers, point_distance(cell, episode["reference_path"]) <= 1.5


def map_values(row, index, geom, field):
    cell, centers, route = geom
    position = row["pos"][index][[0, 2]]
    agent = np.linalg.norm(cell-position, axis=1) <= 1.5
    chosen = str(row["policy_argmax_id"][index])
    ids = json.loads(str(row["candidate_ids"][index]))
    coords = json.loads(str(row["candidate_positions"][index]))
    # STOP is a non-spatial query, including when the policy chooses another node.
    candidate = None if field == "stop_map" or chosen == "-1" else np.asarray(coords[ids.index(chosen)])[[0, 2]]
    disc = np.zeros(len(cell), dtype=bool) if candidate is None else np.linalg.norm(cell-candidate, axis=1) <= 1.5

    def coverage(mask):
        return mask.reshape(10, 10, 10, 10).mean((1, 3)).reshape(100)

    probabilities = row[field][index].astype(np.float64)
    weights = {"agent": coverage(agent), "candidate": coverage(disc), "route": coverage(route),
               "other": coverage(~(agent | disc | route)),
               "patch_distance": np.linalg.norm(centers-position, axis=1)}
    output = {}
    for name, weight in weights.items():
        values = probabilities[:, :100] @ weight
        if name == "patch_distance":
            mass = probabilities[:, :100].sum(1)
            values = np.divide(values, mass, out=np.full_like(values, np.nan), where=mass > 0)
            total = probabilities[:, :100].mean(0)
            output[name] = float(total @ weight / total.sum()) if total.sum() else np.nan
        else:
            output[name] = float(values.mean())
        output[name + "_head_std"] = float(values.std())
    if candidate is None:
        output["candidate"] = output["candidate_head_std"] = np.nan
    output["metadata"] = float(probabilities[:, 100].mean())
    output["metadata_head_std"] = float(probabilities[:, 100].std())
    return output


def load_rows(directory):
    rows = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            row = {key: data[key].copy() for key in data.files}
        eid = str(row["episode_id"].item())
        if eid in rows:
            raise ValueError(f"Duplicate episode: {directory}/{eid}")
        rows[eid] = row
    if not rows:
        raise ValueError(f"No dumps: {directory}")
    return rows


def geodesic_labels(episodes, shared):
    """Exactly D8's geodesic definition, CPU PathFinder only (no Simulator)."""
    import habitat_sim

    finders, output = {}, {}
    for eid, row in shared.items():
        episode = episodes[eid]
        scene = Path(episode["scene_id"]).stem
        if scene not in finders:
            finder = habitat_sim.PathFinder()
            navmesh = REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
            if not finder.load_nav_mesh(str(navmesh)):
                raise ValueError(f"Cannot load {navmesh}")
            finders[scene] = finder
        finder = finders[scene]

        def distance(position):
            query = habitat_sim.MultiGoalShortestPath()
            query.requested_start = np.asarray(position, dtype=np.float32)
            query.requested_ends = np.asarray([goal["position"] for goal in episode["goals"]], dtype=np.float32)
            finder.find_path(query)
            return float(query.geodesic_distance)

        start = distance(episode["start_position"])
        if not np.isfinite(start) or start <= 0:
            raise ValueError(f"Invalid start geodesic: {eid}/{start}")
        for index, step in enumerate(row["step"]):
            dist = distance(row["pos"][index])
            if not np.isfinite(dist):
                raise ValueError(f"Nonfinite S geodesic: {eid}/{step}")
            p = 1-dist/start
            output[eid, int(step)] = (p, dist, min(4, int(max(0, p)*5)))
    return output


def frame_data(rows, episodes, labels, has_map):
    chunks = []
    for eid in sorted(rows, key=int):
        row = rows[eid]
        geom = geometry(episodes[eid], row) if has_map else None
        for index, step in enumerate(row["step"]):
            proximity, distance, label = labels[eid, int(step)]
            frame = {"eid": eid, "step": int(step), "scene": str(row["scene_id"].item()),
                     "proximity": proximity, "distance": distance, "bin": label,
                     "stop_prob": float(row["stop_prob"][index])}
            frame.update(text_values(row, "txt_attn", index))
            frame.update({"stop_"+k: v for k, v in text_values(row, "stop_txt", index).items()})
            if has_map:
                frame.update({"map_"+k: v for k, v in map_values(row, index, geom, "map_attn").items()})
                frame.update({"stop_map_"+k: v for k, v in map_values(row, index, geom, "stop_map").items()})
            chunks.append(frame)
    return {key: np.asarray([frame[key] for frame in chunks]) for key in chunks[0]}


def scene_ci(values, scenes, draws):
    """Frame-weighted means; resample whole scenes including their frame counts."""
    keep = np.isfinite(values)
    values, scenes = values[keep], scenes[keep]
    groups = [values[scenes == scene] for scene in np.unique(scenes)]
    if not groups:
        return {"n": 0, "value": None, "ci": None}
    sums = np.array([x.sum() for x in groups])
    sizes = np.array([len(x) for x in groups])
    sample = np.random.default_rng(0).integers(0, len(groups), (draws, len(groups)))
    boot = sums[sample].sum(1) / sizes[sample].sum(1)
    return {"n": int(len(values)), "value": float(values.mean()), "ci": np.quantile(boot, [.025, .975]).tolist()}


def scene_trends(data, mask):
    output = {}
    for scene in np.unique(data["scene"]):
        keep = mask & (data["scene"] == scene) & np.isfinite(data["position"])
        x, y = data["proximity"][keep], data["position"][keep]
        valid = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
        output[str(scene)] = {"n": int(len(x)), "rho": float(spearmanr(x, y).correlation) if valid else None,
                              "slope": float(np.polyfit(x, y, 1)[0]) if valid else None,
                              "special": float(data["special"][keep].mean()) if len(x) else None}
    return output


def trend_difference(left, right, key, draws):
    scenes = [scene for scene in left if left[scene][key] is not None and right[scene][key] is not None]
    return scene_ci(np.array([left[s][key]-right[s][key] for s in scenes]), np.asarray(scenes), draws)


def group_difference(data, metric, high, low, draws):
    high, low = high & np.isfinite(data[metric]), low & np.isfinite(data[metric])
    scenes = np.unique(data["scene"])
    hs = np.array([data[metric][high & (data["scene"] == s)].sum() for s in scenes])
    ls = np.array([data[metric][low & (data["scene"] == s)].sum() for s in scenes])
    hn = np.array([(high & (data["scene"] == s)).sum() for s in scenes])
    ln = np.array([(low & (data["scene"] == s)).sum() for s in scenes])
    sizes = {str(s): {"high": int(a), "low": int(b)} for s, a, b in zip(scenes, hn, ln)}
    if not hn.sum() or not ln.sum():
        return {"high_n": int(hn.sum()), "low_n": int(ln.sum()), "value": None, "ci": None, "per_scene": sizes}
    sample = np.random.default_rng(0).integers(0, len(scenes), (draws, len(scenes)))
    valid = (hn[sample].sum(1) > 0) & (ln[sample].sum(1) > 0)
    sample = sample[valid]
    boot = hs[sample].sum(1)/hn[sample].sum(1)-ls[sample].sum(1)/ln[sample].sum(1)
    return {"high_n": int(hn.sum()), "low_n": int(ln.sum()), "value": float(hs.sum()/hn.sum()-ls.sum()/ln.sum()),
            "ci": np.quantile(boot, [.025, .975]).tolist() if len(boot) else None,
            "valid_bootstrap_draws": int(len(boot)), "per_scene": sizes,
            "high_special": float(data["stop_special"][high].mean()),
            "low_special": float(data["stop_special"][low].mean())}


def aggregate(data, draws):
    result = {"text": {}, "A1": {}, "paired_text": {}, "paired_A1": {}, "A3": {}, "paired_A3": {}, "A4": {}}
    text_metrics = ("position", "entropy", "top3", "special", "cls", "sep", "padding", "flag",
                    "position_head_std", "entropy_head_std", "top3_head_std")
    for model, row in data.items():
        result["text"][model] = {}
        for name, mask in (("unflagged", row["flag"] == 0), ("flagged", row["flag"] == 1),
                           ("all", np.ones(len(row["step"]), dtype=bool))):
            result["text"][model][name] = {metric: scene_ci(row[metric][mask], row["scene"][mask], draws) for metric in text_metrics}
        result["A1"][model] = scene_trends(row, row["flag"] == 0)
        high = (row["distance"] > 3) & (row["stop_prob"] > .5)
        low = (row["distance"] > 3) & (row["stop_prob"] < .1)
        result["A4"][model] = {}
        for metric in ["stop_position", "stop_entropy", "stop_top3", "stop_special", "stop_flag"] + [k for k in row if k.startswith("stop_map_") and not k.endswith("head_std")]:
            if metric in ("stop_position", "stop_entropy", "stop_top3"):
                for flag in (0, 1):
                    mask = row["stop_flag"] == flag
                    result["A4"][model][metric+f"_flag{flag}"] = group_difference(row, metric, high & mask, low & mask, draws)
            elif metric != "stop_map_candidate":
                result["A4"][model][metric] = group_difference(row, metric, high, low, draws)
        if model != "base":
            result["A3"][model] = {}
            for label in ("all", 0, 1, 2, 3, 4):
                mask = np.ones(len(row["step"]), dtype=bool) if label == "all" else row["bin"] == label
                result["A3"][model][str(label)] = {key: scene_ci(row[key][mask], row["scene"][mask], draws) for key in row if key.startswith("map_")}
                result["A3"][model][str(label)]["text_special"] = scene_ci(row["special"][mask], row["scene"][mask], draws)
    for left, right in (("kd", "base"), ("try5", "base"), ("kd", "try5")):
        a, b = data[left], data[right]
        common = (a["flag"] == 0) & (b["flag"] == 0)
        key = f"{left}-{right}"
        result["paired_text"][key] = {}
        for metric in text_metrics:
            mask = np.ones(len(common), dtype=bool) if metric in ("special", "cls", "sep", "padding", "flag") else common
            result["paired_text"][key][metric] = scene_ci((a[metric]-b[metric])[mask], a["scene"][mask], draws)
        ta, tb = scene_trends(a, common), scene_trends(b, common)
        result["paired_A1"][key] = {"common_frames": int(common.sum()), "left_scenes": ta, "right_scenes": tb,
                                             "special": scene_ci((a["special"]-b["special"])[common], a["scene"][common], draws),
                                             **{metric: trend_difference(ta, tb, metric, draws) for metric in ("rho", "slope")}}
    a, b = data["kd"], data["try5"]
    for label in ("all", 0, 1, 2, 3, 4):
        mask = np.ones(len(a["step"]), dtype=bool) if label == "all" else a["bin"] == label
        result["paired_A3"][str(label)] = {key: scene_ci((a[key]-b[key])[mask], a["scene"][mask], draws) for key in a if key.startswith("map_")}
        result["paired_A3"][str(label)]["text_special"] = scene_ci((a["special"]-b["special"])[mask], a["scene"][mask], draws)
    return result


def export_cases(root, episodes, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets, missing = [], []

    def serializable(value):
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serializable(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    for eid, wanted in (("302", (0, 3, 4)), ("1813", (0, 7, 8))):
        output = destination / eid
        output.mkdir(parents=True, exist_ok=True)
        aligned = {}
        episode = episodes[eid]
        scene = Path(episode["scene_id"]).stem
        cache = REPO / "data/llm_navigation" / KEY / "r2r/val_unseen/cognitive_maps/raster" / scene / f"R2R_val_unseen_{eid}.npz"
        with np.load(cache) as item:
            grid = item["grid"].copy()
        manifest = REPO / "reports/figs/d7_case2" / eid / "manifest.json"
        crop = json.loads(manifest.read_text()).get("topdown_crop") if manifest.exists() else None
        for model in MODELS:
            with np.load(root / model / "O" / f"{eid}.npz", allow_pickle=False) as item:
                row = {key: item[key].copy() for key in item.files}
            for step in dict.fromkeys((*wanted, int(row["step"][-1]))):
                indices = np.flatnonzero(row["step"] == step)
                if not len(indices):
                    missing.append({"episode_id": eid, "model": model, "requested_step": step,
                                    "last_recorded_step": int(row["step"][-1]), "reason": "step absent from recorded own rollout"})
                    continue
                index = int(indices[0])
                text = {"words": row["words"].tolist(), "step": step, "model": model,
                        "policy_argmax_id": str(row["policy_argmax_id"][index]), "executed_id": str(row["executed_id"][index])}
                for field in ("txt_attn", "stop_txt"):
                    mean, heads, specials = word_attention(row, field, index)
                    text[field] = {"mean": mean.tolist(), "per_head": heads.tolist(), "special_per_head": specials.tolist()}
                text_path = output / f"attn_txt_{model}_step{step}.json"
                text = serializable(text)
                text_path.write_text(json.dumps(text, indent=2, allow_nan=False)+"\n")
                assets.append(str(text_path))
                aligned.setdefault(step, {})[model] = text
                if model == "base":
                    continue
                prefix = output / f"attn_map_{model}_step{step}"
                np.savez_compressed(str(prefix)+".npz", selected=row["map_attn"][index, :, :100].mean(0).reshape(10, 10),
                                    stop=row["stop_map"][index, :, :100].mean(0).reshape(10, 10),
                                    selected_per_head=row["map_attn"][index], stop_per_head=row["stop_map"][index],
                                    origin_xz=row["map_origin_xz"], meters_per_cell=.5, patch_size_cells=10,
                                    registration="assumed registration", agent_pos=row["pos"][index], heading=row["heading"][index],
                                    policy_argmax_id=row["policy_argmax_id"][index], executed_id=row["executed_id"][index],
                                    metadata_mass=row["map_attn"][index, :, 100].mean(), stop_metadata_mass=row["stop_map"][index, :, 100].mean(),
                                    candidate_ids=row["candidate_ids"][index], candidate_positions=row["candidate_positions"][index],
                                    gt_path=np.asarray(episode["reference_path"]), topdown_crop_json=json.dumps(crop))
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(grid[17] > 0, cmap="gray", origin="upper", interpolation="nearest", extent=(-.5, 9.5, 9.5, -.5))
                ax.imshow(row["map_attn"][index, :, :100].mean(0).reshape(10, 10), cmap="magma", alpha=.75, interpolation="nearest", origin="upper")
                ax.set_axis_off()
                fig.savefig(str(prefix)+"_raw.png", dpi=150, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                assets += [str(prefix)+".npz", str(prefix)+"_raw.png"]
        for step, models in aligned.items():
            target = output / f"attn_txt_aligned_step{step}.json"
            target.write_text(json.dumps({"episode_id": eid, "models": models}, indent=2, allow_nan=False)+"\n")
            assets.append(str(target))
    return assets, missing


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz", "rt") as stream:
        episodes = {str(e["episode_id"]): e for e in json.load(stream)["episodes"]}
    root = Path(args.dump_root)
    dumps = {model: load_rows(root / model / "S") for model in MODELS}
    if not set(dumps["base"]) == set(dumps["try5"]) == set(dumps["kd"]):
        raise ValueError("S episode sets differ")
    for eid, row in dumps["base"].items():
        for model in ("try5", "kd"):
            for key in ("step", "pos", "candidate_ids", "candidate_positions"):
                if not np.array_equal(row[key], dumps[model][eid][key]):
                    raise ValueError(f"S alignment differs: {eid}/{model}/{key}")
    labels = geodesic_labels(episodes, dumps["base"])
    data = {model: frame_data(rows, episodes, labels, model != "base") for model, rows in dumps.items()}
    result = {"definitions": {"dump_root": str(root.resolve()), "draws": args.draws, "seed": 0, "cpu_workers": 1,
                               "script_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                               "heads": {model: {"graph_query_text": int(next(iter(rows.values()))["txt_attn"].shape[1]),
                                                 "graph_map_attention": int(next(iter(rows.values()))["map_attn"].shape[1]) if model != "base" else None}
                                         for model, rows in dumps.items()},
                               "A4": "unpaired high-minus-low scene-cluster bootstrap; STOP query has no candidate; other uses agent union route in both groups",
                               "registration": "assumed registration; upstream map attention before graph-text interaction",
                               "patch_distance": "mean distance conditional on spatial-token mass (metadata excluded)",
                               "special_flag": ">0.8; paired words use neither-flagged common frames",
                               "head_spread": "NA if any head has zero word mass after float16 storage; summary n counts defined frames",
                               "undefined": "NA, never zero"},
              "statistics": aggregate(data, args.draws)}
    # Keep raw frame values so every aggregate is reproducible without GPU dumps.
    for model, row in data.items():
        np.savez_compressed(output / f"frames_{model}.npz", **row)
    result["case_assets"], result["missing_case_frames"] = export_cases(root, episodes, Path(args.cases_output))
    (output / "stats.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    from scripts.eval_d9.attention_report import write_report
    write_report(result, output)
    print(f"D9 complete: {output / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
