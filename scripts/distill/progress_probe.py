#!/usr/bin/env python3
"""CPU-only goal-proximity labels and scene-held-out linear probes for D8."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tap import Tap
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
MODELS = ("base", "try5", "kd")
MISSING = {"308", "349", "455", "804", "1318", "1331"}


class Args(Tap):
    dump_root: str = str(REPO / "data/logs/d8_dump")
    output: str = str(REPO / "reports/d8")
    permutations: int = 20
    draws: int = 2000
    threads: int = 4
    check_shared_only: bool = False

    def configure(self):
        self.add_argument("--check-shared-only", action="store_true", dest="check_shared_only", default=False, required=False)


def load_dumps(root, episodes, sources=("S", "O")):
    result = {}
    for model in MODELS:
        for source in sources:
            directory = root / model / source
            rows = {}
            for path in sorted(directory.glob("*.npz")):
                with np.load(path, allow_pickle=False) as item:
                    row = {key: item[key].copy() for key in item.files}
                eid = str(row["episode_id"].item())
                if eid in rows:
                    raise ValueError(f"Duplicate episode {eid}: {directory}")
                count = len(row["step"])
                if row["feat"].shape != (count, 1536) or row["pos"].shape != (count, 3):
                    raise ValueError(f"Bad feature/position shape: {path}")
                if not np.isfinite(row["feat"]).all():
                    raise ValueError(f"Nonfinite features: {path}")
                rows[eid] = row
            expected = set(episodes) - (MISSING if source == "S" or model != "base" else set())
            if set(rows) != expected:
                raise ValueError(f"Incomplete {directory}: missing={sorted(expected-set(rows))}, unexpected={sorted(set(rows)-expected)}")
            result[model, source] = rows
    return result


def check_shared(dumps):
    for eid in sorted(dumps["try5", "S"], key=int):
        base = dumps["base", "S"][eid]
        for model in ("try5", "kd"):
            row = dumps[model, "S"][eid]
            for key in ("step", "pos", "candidate_count", "candidate_ids", "candidate_positions", "scene_id"):
                if not np.array_equal(base[key], row[key]):
                    raise ValueError(f"Shared states disagree: episode={eid}, model={model}, field={key}")


def polyline_distance(position, path):
    points = np.asarray(path, dtype=np.float64)
    a, b = points[:-1], points[1:]
    delta = b - a
    denom = (delta * delta).sum(1)
    t = np.divide(((position - a) * delta).sum(1), denom, out=np.zeros(len(a)), where=denom > 0)
    nearest = a + np.clip(t, 0, 1)[:, None] * delta
    return float(np.linalg.norm(position - nearest, axis=1).min())


def labelled(dumps, episodes):
    import habitat_sim

    finders = {}
    distances = {}
    output, accounting = {}, {}
    for (model, source), rows in dumps.items():
        chunks = {key: [] for key in ("x", "y", "scene", "eid", "step", "prob", "distance", "proximity", "on_gt")}
        counts = {"before": 0, "infinite_dropped": 0, "negative_clipped": 0, "off_gt": 0, "after": 0}
        for eid, row in rows.items():
            episode = episodes[eid]
            scene = Path(episode["scene_id"]).stem
            if scene not in finders:
                finder = habitat_sim.PathFinder()
                navmesh = REPO / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
                if not finder.load_nav_mesh(str(navmesh)):
                    raise ValueError(f"Cannot load navmesh: {navmesh}")
                finders[scene] = finder
            finder = finders[scene]

            def geo(position):
                query = habitat_sim.MultiGoalShortestPath()
                query.requested_start = np.asarray(position, dtype=np.float32)
                query.requested_ends = np.asarray([g["position"] for g in episode["goals"]], dtype=np.float32)
                finder.find_path(query)
                return float(query.geodesic_distance)

            if eid not in distances:
                distances[eid] = geo(episode["start_position"])
            start = distances[eid]
            if not np.isfinite(start) or start <= 0:
                raise ValueError(f"Invalid start geodesic: episode={eid}, distance={start}")
            for index, position in enumerate(row["pos"]):
                counts["before"] += 1
                distance = geo(position)
                if not np.isfinite(distance):
                    counts["infinite_dropped"] += 1
                    continue
                progress = 1 - distance / start
                counts["negative_clipped"] += int(progress < 0)
                on_gt = polyline_distance(position, episode["reference_path"]) <= 1
                counts["off_gt"] += int(not on_gt)
                if source == "O" and not on_gt:
                    continue
                values = {"x": row["feat"][index], "y": min(4, int(max(0, progress) * 5)), "scene": scene,
                          "eid": eid, "step": row["step"][index], "prob": row["stop_prob"][index],
                          "distance": distance, "proximity": progress, "on_gt": on_gt}
                for key, value in values.items():
                    chunks[key].append(value)
                counts["after"] += 1
        output[model, source] = {key: np.asarray(values) for key, values in chunks.items()}
        accounting[f"{model}/{source}"] = counts
    # Main comparison uses the exact shared frame intersection, excluding U in all models.
    base = output["base", "S"]
    keep = ~np.isin(base["eid"], list(MISSING))
    output["base", "S"] = {key: value[keep] for key, value in base.items()}
    # File enumeration is lexical; align all shared frames to episode/step order.
    for model in MODELS:
        data = output[model, "S"]
        order = np.lexsort((data["step"], data["eid"].astype(int)))
        output[model, "S"] = {key: value[order] for key, value in data.items()}
    for model in ("try5", "kd"):
        for key in ("eid", "step", "y", "scene"):
            if not np.array_equal(output["base", "S"][key], output[model, "S"][key]):
                raise ValueError(f"Labelled shared frames disagree: {model}/{key}")
    return output, accounting


def readout(x, y, scene):
    pred = np.empty_like(y)
    majority = np.empty_like(y)
    folds = []
    for held in np.unique(scene):
        test = scene == held
        train = ~test
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=1000, multi_class="multinomial"))
        estimator.fit(x[train].astype(np.float64), y[train])
        pred[test] = estimator.predict(x[test].astype(np.float64))
        majority[test] = np.bincount(y[train], minlength=5).argmax()
        folds.append({"scene": str(held), "n": int(test.sum()),
                      "accuracy": float(accuracy_score(y[test], pred[test])),
                      "balanced_accuracy": float(balanced_accuracy_score(y[test], pred[test]))})
    return pred, majority, folds


def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "spearman": float(spearmanr(y, pred).correlation) if np.unique(pred).size > 1 else None,
            "confusion": confusion_matrix(y, pred, labels=np.arange(5)).tolist(),
            "bins": np.bincount(y, minlength=5).tolist()}


def paired_ci(y, left, right, scene, draws):
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(scene == s) for s in np.unique(scene)]
    values = []
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(balanced_accuracy_score(y[indices], left[indices]) - balanced_accuracy_score(y[indices], right[indices]))
    return {"difference": float(balanced_accuracy_score(y, left)-balanced_accuracy_score(y, right)),
            "ci": np.quantile(values, [0.025, 0.975]).tolist()}


def curves(data, draws):
    result = []
    rng = np.random.default_rng(0)
    for label in range(5):
        indices = np.flatnonzero((data["y"] == label) & (data["distance"] > 3))
        groups = [indices[data["eid"][indices] == eid] for eid in np.unique(data["eid"][indices])]
        if not groups:
            result.append({"bin": label, "n": 0, "mean": None, "ci": None})
            continue
        sums = np.array([data["prob"][g].sum() for g in groups])
        sizes = np.array([len(g) for g in groups])
        values = []
        for _ in range(draws):
            sample = rng.integers(0, len(groups), len(groups))
            values.append(sums[sample].sum()/sizes[sample].sum())
        result.append({"bin": label, "n": len(indices), "mean": float(data["prob"][indices].mean()),
                       "ci": np.quantile(values, [0.025, 0.975]).tolist()})
    return result


def main():
    args = Args().parse_args()
    with threadpool_limits(limits=args.threads):
        execute(args)


def execute(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz", "rt") as stream:
        episodes = {str(e["episode_id"]): e for e in json.load(stream)["episodes"]}
    dumps = load_dumps(Path(args.dump_root), episodes, sources=("S",) if args.check_shared_only else ("S", "O"))
    check_shared(dumps)
    shared = {"episodes": len(dumps["base", "S"]),
              "frames": sum(len(row["step"]) for row in dumps["base", "S"].values()),
              "fields_checked": ["step", "pos", "candidate_count", "candidate_ids", "candidate_positions", "scene_id"],
              "status": "all frames identical", "episodes_detail": {}}
    for eid, row in dumps["base", "S"].items():
        shared["episodes_detail"][eid] = {"frames": len(row["step"]), "first_position": row["pos"][0].tolist(),
                                         "last_position": row["pos"][-1].tolist()}
    (output / "shared_validation.json").write_text(json.dumps(shared, indent=2)+"\n")
    if args.check_shared_only:
        print(f"Shared validation passed: {shared['episodes']} episodes, {shared['frames']} frames", flush=True)
        return
    datasets, accounting = labelled(dumps, episodes)
    results = {"definitions": {"proximity": "1-geodesic(position,goal)/geodesic(start,goal)", "bins": 5,
                              "inputs": "base no map; try5/kd WITH MAP; control; 4 rank", "step_control": "t,t/15",
                              "main": "shared S; all-frame metadata checked; LOSO; balanced accuracy",
                              "O_filter": "distance to GT polyline <=1m", "curve_filter": "geodesic distance>3m",
                              "S_expert": "existing SPL/geodesic expert replay, possibly shortcuts; off-GT S frames retained as diagnostic, not filtered"},
               "accounting": accounting, "probes": {}, "paired": {}, "curves": {}}
    for source in ("S", "O"):
        predictions = {}
        for model in MODELS:
            data = datasets[model, source]
            print(f"Fitting {model}/{source}: {len(data['y'])} frames, {len(np.unique(data['scene']))} scenes", flush=True)
            pred, majority, folds = readout(data["x"], data["y"], data["scene"])
            step = data["step"].astype(float)
            step_pred, _, step_folds = readout(np.column_stack([step, step/15]), data["y"], data["scene"])
            row = {"model": metrics(data["y"], pred), "step_only": metrics(data["y"], step_pred),
                   "majority": metrics(data["y"], majority), "permutations": [], "folds": folds, "step_folds": step_folds}
            rng = np.random.default_rng(0)
            for draw in range(args.permutations):
                labels = data["y"].copy()
                for scene in np.unique(data["scene"]):
                    indices = np.flatnonzero(data["scene"] == scene)
                    labels[indices] = rng.permutation(labels[indices])
                null_pred, _, _ = readout(data["x"], labels, data["scene"])
                row["permutations"].append(metrics(labels, null_pred))
                print(f"{source}/{model} permutation {draw+1}/{args.permutations}", flush=True)
            results["probes"][f"{model}/{source}"] = row
            results["paired"][f"{model}-step/{source}"] = paired_ci(data["y"], pred, step_pred, data["scene"], args.draws)
            results["curves"][f"{model}/{source}"] = curves(data, args.draws)
            predictions[model] = pred
            (output / "progress_probe_partial.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
            print(f"Completed {model}/{source}: BA={row['model']['balanced_accuracy']:.4f}", flush=True)
        if source == "S":
            for left, right in (("kd", "base"), ("kd", "try5"), ("try5", "base")):
                data = datasets[left, source]
                results["paired"][f"{left}-{right}/S"] = paired_ci(data["y"], predictions[left], predictions[right], data["scene"], args.draws)
    (output / "progress_probe.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["# D8 goal-proximity probe", "", "Baseline: no map. Try5/KD: with map. All control, four ranks.", "",
             "|Model/source|Accuracy|Balanced accuracy|Spearman|Step BA|Majority BA|", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results["probes"].items():
        m = row["model"]
        lines.append(f"|{name}|{m['accuracy']:.4f}|{m['balanced_accuracy']:.4f}|{m['spearman']}|{row['step_only']['balanced_accuracy']:.4f}|{row['majority']['balanced_accuracy']:.4f}|")
    lines += ["", "|Paired BA difference|Difference|95% scene-bootstrap CI|", "|---|---:|---|"]
    for name, row in results["paired"].items():
        lines.append(f"|{name}|{row['difference']:.4f}|{row['ci']}|")
    lines += ["", "## Frame accounting", "", "```json", json.dumps(accounting, indent=2), "```",
              "", "All confusion matrices, label histograms and permutation draws are in progress_probe.json."]
    lines += ["", "## Held-out scene metrics", "", "|Model/source|Scene|Frames|Accuracy|Balanced accuracy|", "|---|---|---:|---:|---:|"]
    for name, row in results["probes"].items():
        for fold in row["folds"]:
            lines.append(f"|{name}|{fold['scene']}|{fold['n']}|{fold['accuracy']:.4f}|{fold['balanced_accuracy']:.4f}|")
    lines += ["", "## Permutation leakage control", "", "|Model/source|Draws|Accuracy mean ± SD|Balanced accuracy mean ± SD|", "|---|---:|---|---|"]
    for name, row in results["probes"].items():
        permutations = row["permutations"]
        if permutations:
            acc = np.array([p["accuracy"] for p in permutations])
            ba = np.array([p["balanced_accuracy"] for p in permutations])
            lines.append(f"|{name}|{len(permutations)}|{acc.mean():.4f} ± {acc.std():.4f}|{ba.mean():.4f} ± {ba.std():.4f}|")
    lines += ["", "## Per-bin confusion matrices", "", "Rows: true bin; columns: predicted bin."]
    for name, row in results["probes"].items():
        lines += ["", f"### {name}", "", "|True bin|0|1|2|3|4|", "|---|---:|---:|---:|---:|---:|"]
        for label, values in enumerate(row["model"]["confusion"]):
            lines.append("|"+"|".join(map(str, [label]+values))+"|")
    (output / "progress_probe.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    curve_lines = ["# STOP probability by goal proximity", "", "Frames restricted to geodesic distance >3m. O additionally on-GT <=1m.", "", "|Model/source|Bin|Frames|Mean|95% episode-bootstrap CI|", "|---|---:|---:|---:|---|"]
    for axis, source in zip(axes, ("S", "O")):
        for model in MODELS:
            rows = results["curves"][f"{model}/{source}"]
            valid = [r for r in rows if r["mean"] is not None]
            axis.plot([r["bin"] for r in valid], [r["mean"] for r in valid], label=model, marker="o")
            axis.fill_between([r["bin"] for r in valid], [r["ci"][0] for r in valid], [r["ci"][1] for r in valid], alpha=0.15)
            for row in rows:
                curve_lines.append(f"|{model}/{source}|{row['bin']}|{row['n']}|{row['mean']}|{row['ci']}|")
        axis.set(title=source, xlabel="Proximity bin", ylabel="STOP probability")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "stop_prob_by_progress.png", dpi=180)
    plt.close(figure)
    (output / "stop_prob_by_progress.md").write_text("\n".join(curve_lines)+"\n")
    (output / "README.md").write_text("# D8\n\n[Linear probe](progress_probe.md) · [STOP curves](stop_prob_by_progress.md)\n\nPrimary comparison: shared states, balanced accuracy. Positive paired CI lower bound is required for an improvement claim. These are experimental representation readouts, not instruction-stage labels.\n")


if __name__ == "__main__":
    main()
