#!/usr/bin/env python3
"""CPU-only D10 two-arm summaries, reusing the D9 attention definitions."""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from tap import Tap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.distill.attention_stats import (
    frame_data, geodesic_labels, geometry, group_difference, load_rows,
    scene_ci, scene_trends, trend_difference,
)
from scripts.eval_d9.attention_report import estimate, number, table

ARMS = ("Z", "M")
REGIONS = ("agent", "candidate", "route", "other")
TEXT = ("position", "entropy", "top3", "special", "cls", "sep", "padding", "flag",
        "position_head_std", "entropy_head_std", "top3_head_std")
SPECIAL = ("special", "cls", "sep", "padding", "flag")
BINS = ("all", "0", "1", "2", "3", "4")


class Args(Tap):
    zero_dir: str = str(REPO / "data/logs/d10_dump/zero/S")
    map_dir: str = str(REPO / "data/logs/d9_reference_layout/try5/S")
    checks: str = str(REPO / "reports/d10/checks.json")
    output: str = str(REPO / "reports/d10")
    draws: int = 2000


def conditional_lift(probabilities, coverage):
    """Mean-head spatial mass, conditional on spatial tokens, divided by area."""
    spatial = np.asarray(probabilities, dtype=np.float64)[:, :100].mean(0)
    area = float(np.mean(coverage))
    mass = float(spatial.sum())
    return float(spatial @ coverage / mass / area) if area > 0 and mass > 0 else np.nan


def add_lifts(data, rows, episodes):
    """Use D9's raster-level union and fractional patch coverage unchanged."""
    values = {region: [] for region in REGIONS}
    for eid in sorted(rows, key=int):
        row = rows[eid]
        cell, _, route = geometry(episodes[eid], row)
        for index, _ in enumerate(row["step"]):
            agent = np.linalg.norm(cell-row["pos"][index][[0, 2]], axis=1) <= 1.5
            chosen = str(row["policy_argmax_id"][index])
            disc = np.zeros(len(cell), dtype=bool)
            if chosen != "-1":
                ids = json.loads(str(row["candidate_ids"][index]))
                positions = json.loads(str(row["candidate_positions"][index]))
                target = np.asarray(positions[ids.index(chosen)])[[0, 2]]
                disc = np.linalg.norm(cell-target, axis=1) <= 1.5
            masks = {"agent": agent, "candidate": disc, "route": route,
                     "other": ~(agent | disc | route)}
            for region, mask in masks.items():
                coverage = mask.reshape(10, 10, 10, 10).mean((1, 3)).reshape(100)
                values[region].append(conditional_lift(row["map_attn"][index], coverage))
    for region, value in values.items():
        data["lift_"+region] = np.asarray(value)


def bin_mask(row, label):
    return np.ones(len(row["step"]), dtype=bool) if label == "all" else row["bin"] == int(label)


def trend_summary(scenes, metric, draws):
    selected = [scene for scene, item in scenes.items() if item[metric] is not None]
    return scene_ci(np.asarray([scenes[s][metric] for s in selected]), np.asarray(selected), draws)


def summarize(data, draws):
    z, m = data["Z"], data["M"]
    common = (z["flag"] == 0) & (m["flag"] == 0)
    result = {"text": {}, "A1": {}, "A3": {}, "lift": {}, "A4": {},
              "paired_text": {}, "paired_A1": {}, "paired_A3": {}, "paired_lift": {}}
    for arm, row in data.items():
        result["text"][arm] = {}
        for name, keep in (("all", np.ones(len(common), dtype=bool)),
                           ("unflagged", row["flag"] == 0), ("flagged", row["flag"] == 1)):
            result["text"][arm][name] = {
                metric: scene_ci(row[metric][keep], row["scene"][keep], draws) for metric in TEXT}
        result["A1"][arm] = scene_trends(row, row["flag"] == 0)
        result["A3"][arm], result["lift"][arm] = {}, {}
        for label in BINS:
            keep = bin_mask(row, label)
            result["A3"][arm][label] = {
                key: scene_ci(row[key][keep], row["scene"][keep], draws)
                for key in row if key.startswith("map_")}
            result["A3"][arm][label]["text_special"] = scene_ci(
                row["special"][keep], row["scene"][keep], draws)
            result["lift"][arm][label] = {
                region: scene_ci(row["lift_"+region][keep], row["scene"][keep], draws)
                for region in REGIONS}
        high = (row["distance"] > 3) & (row["stop_prob"] > .5)
        low = (row["distance"] > 3) & (row["stop_prob"] < .1)
        result["A4"][arm] = {}
        metrics = ["stop_position", "stop_entropy", "stop_top3", "stop_special", "stop_flag"]
        metrics += [key for key in row if key.startswith("stop_map_")
                    and not key.endswith("head_std") and key != "stop_map_candidate"]
        for metric in metrics:
            if metric in ("stop_position", "stop_entropy", "stop_top3"):
                for flag in (0, 1):
                    keep = row["stop_flag"] == flag
                    result["A4"][arm][metric+f"_flag{flag}"] = group_difference(
                        row, metric, high & keep, low & keep, draws)
            else:
                result["A4"][arm][metric] = group_difference(row, metric, high, low, draws)
    for metric in TEXT:
        keep = np.ones(len(common), dtype=bool) if metric in SPECIAL else common
        result["paired_text"][metric] = scene_ci(
            (m[metric]-z[metric])[keep], m["scene"][keep], draws)
    tm, tz = scene_trends(m, common), scene_trends(z, common)
    result["paired_A1"] = {"common_frames": int(common.sum()), "M_scenes": tm, "Z_scenes": tz,
                           **{key: trend_difference(tm, tz, key, draws) for key in ("rho", "slope")}}
    for label in BINS:
        keep = bin_mask(m, label)
        result["paired_A3"][label] = {
            key: scene_ci((m[key]-z[key])[keep], m["scene"][keep], draws)
            for key in m if key.startswith("map_")}
        result["paired_A3"][label]["text_special"] = scene_ci(
            (m["special"]-z["special"])[keep], m["scene"][keep], draws)
        result["paired_lift"][label] = {
            region: scene_ci((m["lift_"+region]-z["lift_"+region])[keep], m["scene"][keep], draws)
            for region in REGIONS}
    result["zero_lift_check"] = {
        region: {"defined_frames": int(np.isfinite(z["lift_"+region]).sum()),
                 "max_abs_from_one": float(np.max(np.abs(z["lift_"+region][np.isfinite(z["lift_"+region])]-1)))
                 if np.isfinite(z["lift_"+region]).any() else None}
        for region in REGIONS}
    return result


def figure_data(stats, draws):
    """Exactly the five requested panels, with the reported cluster intervals."""
    return {
        "special_share": {arm: {key: stats["text"][arm]["all"][key]
                                for key in SPECIAL} for arm in ARMS},
        "rho_position_proximity": {
            arm: {"per_scene": stats["paired_A1"][arm+"_scenes"],
                  "mean_scene_rho": trend_summary(stats["paired_A1"][arm+"_scenes"], "rho", draws)}
            for arm in ARMS},
        "STOP_row_shift": stats["A4"],
        "route_band_by_bin": {arm: {label: stats["A3"][arm][label]["map_route"]
                                    for label in BINS} for arm in ARMS},
        "route_lift_by_bin": {arm: {label: stats["lift"][arm][label]["route"]
                                    for label in BINS} for arm in ARMS},
        "paired_M_minus_Z": {"special_share": {key: stats["paired_text"][key] for key in SPECIAL},
                             "rho_position_proximity": stats["paired_A1"]["rho"],
                             "route_band_by_bin": {b: stats["paired_A3"][b]["map_route"] for b in BINS},
                             "route_lift_by_bin": {b: stats["paired_lift"][b]["route"] for b in BINS}},
    }


def write_report(result, output):
    stats = result["statistics"]
    lines = ["# D10 attention statistics", "", "Raw tables precede paired contrasts. NA is undefined, never zero.",
             "Attention is descriptive, upstream of graph–text interaction; map registration is assumed.",
             "Single training run per arm. The user waived the pretraining-initialization identity blocker; see README identity audit.",
             "", "## Raw A1–A2: instruction attention", ""]
    rows = [[arm, subset, key, item["n"], *estimate(item)]
            for arm, subsets in stats["text"].items() for subset, metrics in subsets.items()
            for key, item in metrics.items()]
    table(lines, ["Arm", "Special subset", "Metric", "Frames", "Mean", "Scene CI"], rows)
    rows = [[arm, scene, item["n"], number(item["rho"]), number(item["slope"]), number(item["special"])]
            for arm, scenes in stats["A1"].items() for scene, item in scenes.items()]
    table(lines, ["Arm", "Scene", "Unflagged frames", "Rho", "Slope", "Special mass"], rows)
    for key, title in (("A3", "Raw A3: map attention"), ("lift", "Raw conditional-spatial lift")):
        lines += ["## "+title, "", "Regions overlap; candidate is NA for STOP. Metadata is reported separately.", ""]
        rows = [[arm, label, metric, item["n"], *estimate(item)]
                for arm, bins in stats[key].items() for label, metrics in bins.items()
                for metric, item in metrics.items()]
        table(lines, ["Arm", "Proximity bin", "Metric", "Defined frames", "Mean", "Scene CI"], rows)
    lines += ["## Raw A4: STOP-row high-minus-low groups", "",
              "Actual distance >3m; STOP probability >0.5 versus <0.1. Between-group scene bootstrap, not paired frames.",
              "STOP queries have no candidate; other uses agent ∪ route. Flagged word statistics stay separate.", ""]
    rows = [[arm, key, item["high_n"], item["low_n"], *estimate(item),
             number(item.get("high_special")), number(item.get("low_special"))]
            for arm, metrics in stats["A4"].items() for key, item in metrics.items()]
    table(lines, ["Arm", "Metric", "High n", "Low n", "High − low", "Scene CI", "High special", "Low special"], rows)
    rows = [[arm, key, scene, counts["high"], counts["low"]]
            for arm, metrics in stats["A4"].items() for key, item in metrics.items()
            for scene, counts in item["per_scene"].items()]
    table(lines, ["Arm", "Metric", "Scene", "High n", "Low n"], rows)
    lines += ["## Paired M − Z", "",
              "Word contrasts use common neither-flagged frames; candidate contrasts require both non-STOP and finite values.", ""]
    table(lines, ["Text metric", "Common frames", "Difference", "Scene CI"],
          [[key, item["n"], *estimate(item)] for key, item in stats["paired_text"].items()])
    table(lines, ["Trend", "Common frames", "Defined scenes", "Difference", "Scene CI"],
          [[key, stats["paired_A1"]["common_frames"], stats["paired_A1"][key]["n"],
            *estimate(stats["paired_A1"][key])] for key in ("rho", "slope")])
    for key, title in (("paired_A3", "Paired A3"), ("paired_lift", "Paired conditional-spatial lift")):
        lines += ["### "+title, ""]
        table(lines, ["Bin", "Metric", "Common defined frames", "Difference", "Scene CI"],
              [[label, metric, item["n"], *estimate(item)] for label, metrics in stats[key].items()
               for metric, item in metrics.items()])
    lines += ["## Z lift check", "", "Uniformity is checked, not assumed. Nonuniform Z remains an empirical reference.", "",
              "```json", json.dumps(stats["zero_lift_check"], indent=2), "```", "",
              "## Definitions", "", "```json", json.dumps(result["definitions"], indent=2), "```", ""]
    (output / "stats.md").write_text("\n".join(lines))
    start, end = "<!-- D10 RESULTS START -->", "<!-- D10 RESULTS END -->"
    readme_path = output / "README.md"
    original = readme_path.read_text() if readme_path.exists() else "# D10 attention analysis\n"
    reading = [start, "## D10 results", "", "See [stats.md](stats.md), [stats.json](stats.json), and [figure_data.json](figure_data.json).", "",
               "User-authorized continuation despite the recorded pretraining-initialization difference. Single run per arm; attention comparisons are descriptive.",
               "This section does not remove or supersede the identity-audit evidence above.", ""]
    for metric in ("entropy", "top3"):
        item = stats["paired_text"][metric]
        detected = item["ci"] is not None and (item["ci"][0] > 0 or item["ci"][1] < 0)
        reading.append(f"- M − Z {metric}: {number(item['value'])}; CI {item['ci']}; "
                       + ("difference detected within this pair." if detected else "no difference detected."))
    item = stats["paired_lift"]["all"]["route"]
    reading += [f"- Route-band lift M − Z: {number(item['value'])}; CI {item['ci']}."]
    reading += ["- M's route-band attention exceeds the empirical Z reference." if item["ci"] is not None and item["ci"][0] > 0
                else "- No higher route-band lift detected for M relative to Z.", "", end]
    block = "\n".join(reading)
    if start in original:
        before, tail = original.split(start, 1)
        _, after = tail.split(end, 1)
        original = before+block+after
    else:
        original = original.rstrip()+"\n\n"+block+"\n"
    readme_path.write_text(original)


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    checks = json.loads(Path(args.checks).read_text())
    if checks["passed"] is not True:
        raise ValueError("D10 Section 3.3 checks have not passed; run checks.py first")
    dumps = {"Z": load_rows(Path(args.zero_dir)), "M": load_rows(Path(args.map_dir))}
    if dumps["Z"].keys() != dumps["M"].keys():
        raise ValueError("D10 S episode sets differ")
    for eid, row in dumps["Z"].items():
        for key in ("step", "pos", "candidate_ids", "candidate_positions"):
            if not np.array_equal(row[key], dumps["M"][eid][key]):
                raise ValueError(f"D10 S alignment differs: {eid}/{key}")
    count = sum(len(row["step"]) for row in dumps["Z"].values())
    if len(dumps["Z"]) != 1833 or count != 13992:
        raise ValueError(f"Incomplete D10 S data: {len(dumps['Z'])} episodes/{count} frames")
    with gzip.open(REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz", "rt") as stream:
        episodes = {str(e["episode_id"]): e for e in json.load(stream)["episodes"]}
    labels = geodesic_labels(episodes, dumps["M"])
    data = {}
    for arm, rows in dumps.items():
        data[arm] = frame_data(rows, episodes, labels, True)
        add_lifts(data[arm], rows, episodes)
    result = {"definitions": {"zero_dir": str(Path(args.zero_dir).resolve()),
                               "map_dir": str(Path(args.map_dir).resolve()), "draws": args.draws,
                               "seed": 0, "cpu_workers": 1, "checks": checks,
                               "identity_waiver": "User permitted continuation despite pretraining initialization difference; Z training start pose confirmed zero.",
                               "lift": "region raw mass / spatial mass / fractional spatial area; mass or area zero is NA",
                               "A4": "Between-group high-minus-low, distance>3m, STOP>.5 versus STOP<.1; no candidate region",
                               "paired_words": "Common frames where neither model has special mass>.8",
                               "registration": "assumed; upstream map attention, globally mixed patch tokens",
                               "metadata": "Reported separately, not interpreted as a map-content effect",
                               "head_spread": "D9 head standard deviations retained; undefined head-word mass is NA",
                               "undefined": "NA, never zero"},
              "statistics": summarize(data, args.draws)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for arm, row in data.items():
        np.savez_compressed(output / f"frames_{arm}.npz", **row)
    (output / "stats.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output / "figure_data.json").write_text(json.dumps(figure_data(result["statistics"], args.draws), indent=2, allow_nan=False)+"\n")
    write_report(result, output)
    print(f"D10 CPU statistics complete: {output / 'stats.md'}", flush=True)


if __name__ == "__main__":
    main()
