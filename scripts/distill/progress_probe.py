#!/usr/bin/env python3
"""CPU-only goal-proximity labels and scene-held-out linear probes for D8."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from tap import Tap
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.eval_d8.probe_fit import readout
from scripts.eval_d8.probe_report import write_final_report, write_source_report
MODELS = ("base", "try5", "kd")
MISSING = {"308", "349", "455", "804", "1318", "1331"}


class Args(Tap):
    dump_root: str = str(REPO / "data/logs/d8_dump")
    output: str = str(REPO / "reports/d8")
    permutations: int = 20
    draws: int = 2000
    threads: int = 1
    workers: int = 11
    resume: bool = False
    check_shared_only: bool = False

    def configure(self):
        self.add_argument("--resume", action="store_true", default=False, required=False)
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


def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "spearman": float(spearmanr(y, pred).correlation) if np.unique(pred).size > 1 else None,
            "confusion": confusion_matrix(y, pred, labels=np.arange(5)).tolist(),
            "bins": np.bincount(y, minlength=5).tolist()}


def paired_ci(y, left, right, scene, draws):
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(scene == s) for s in np.unique(scene)]
    values, accuracies = [], []
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(balanced_accuracy_score(y[indices], left[indices]) - balanced_accuracy_score(y[indices], right[indices]))
        accuracies.append(accuracy_score(y[indices], left[indices])-accuracy_score(y[indices], right[indices]))
    return {"balanced_accuracy": {
                "difference": float(balanced_accuracy_score(y, left)-balanced_accuracy_score(y, right)),
                "ci": np.quantile(values, [0.025, 0.975]).tolist()},
            "accuracy": {"difference": float(accuracy_score(y, left)-accuracy_score(y, right)),
                         "ci": np.quantile(accuracies, [0.025, 0.975]).tolist()}}


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
                              "S_expert": "existing SPL/geodesic expert replay, possibly shortcuts; off-GT S frames retained as diagnostic, not filtered",
                              "max_iter": 5000, "C": 1, "solver": "lbfgs",
                              "bootstrap": "2000 scene-cluster draws; fixed out-of-fold predictions; no refitting",
                              "workers": args.workers, "worker_blas_threads": 1,
                              "permutation_draws": args.permutations,
                              "script_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "accounting": accounting, "probes": {}, "paired": {}, "curves": {}}
    execution = {key: results["definitions"][key] for key in
                 ("script_commit", "script_sha256", "workers", "worker_blas_threads")}
    if args.resume:
        results = json.loads((output / "progress_probe_partial.json").read_text())
        definitions = results["definitions"]
        history = definitions.setdefault("execution_history", [
            {key: definitions[key] for key in execution}
        ])
        history.append(dict(execution, stage="resume"))
        definitions["workers"] = args.workers
        definitions["resume_script_commit"] = execution["script_commit"]
        definitions["resume_script_sha256"] = execution["script_sha256"]
        if definitions["permutation_draws"] != args.permutations:
            raise ValueError("Resume permutation count differs from the saved run")
        if results["accounting"] != accounting:
            raise ValueError("Resume label accounting differs from the saved run")
    else:
        results["definitions"]["execution_history"] = [dict(execution, stage="initial")]
    for source in ("S", "O"):
        source_dir = output / ("probe_s" if source == "S" else "probe_o")
        source_dir.mkdir(parents=True, exist_ok=True)
        predictions = {}
        for model in MODELS:
            data = datasets[model, source]
            if f"{model}/{source}" in results["probes"]:
                with np.load(source_dir / f"{model}_oof.npz", allow_pickle=False) as saved:
                    for key in ("y", "eid", "step", "scene"):
                        if not np.array_equal(saved[key], data[key]):
                            raise ValueError(f"Resume OOF {model}/{source}: {key} differs")
                    predictions[model] = saved["pred"].copy()
                print(f"Reusing completed {model}/{source}", flush=True)
                continue
            print(f"Fitting {model}/{source}: {len(data['y'])} frames, {len(np.unique(data['scene']))} scenes", flush=True)
            pred, majority, folds = readout(data["x"], data["y"], data["scene"], args.workers, f"{model}/{source}/feature")
            step = data["step"].astype(float)
            step_pred, _, step_folds = readout(np.column_stack([step, step/15]), data["y"], data["scene"], args.workers, f"{model}/{source}/step")
            row = {"model": metrics(data["y"], pred), "step_only": metrics(data["y"], step_pred),
                   "majority": metrics(data["y"], majority), "permutations": [], "folds": folds, "step_folds": step_folds,
                   "converged": all(f["converged"] for f in folds),
                   "step_converged": all(f["converged"] for f in step_folds)}
            results["probes"][f"{model}/{source}"] = row
            results["paired"][f"{model}-step/{source}"] = paired_ci(data["y"], pred, step_pred, data["scene"], args.draws)
            results["paired"][f"{model}-step/{source}"]["provisional"] = not (row["converged"] and row["step_converged"])
            results["curves"][f"{model}/{source}"] = curves(data, args.draws)
            predictions[model] = pred
            np.savez(source_dir / f"{model}_oof.npz", pred=pred, step_pred=step_pred, majority=majority,
                     y=data["y"], scene=data["scene"], eid=data["eid"], step=data["step"])
            (output / "progress_probe_partial.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
            print(f"Completed {model}/{source}: BA={row['model']['balanced_accuracy']:.4f}", flush=True)
        if source == "S":
            for left, right in (("kd", "base"), ("kd", "try5"), ("try5", "base")):
                if f"{left}-{right}/S" in results["paired"]:
                    continue
                data = datasets[left, source]
                results["paired"][f"{left}-{right}/S"] = paired_ci(data["y"], predictions[left], predictions[right], data["scene"], args.draws)
                results["paired"][f"{left}-{right}/S"]["provisional"] = not (
                    results["probes"][f"{left}/S"]["converged"] and results["probes"][f"{right}/S"]["converged"])
        # Save the primary comparisons and curves before the expensive null fits.
        (source_dir / "progress_probe.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        (output / "progress_probe_partial.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        write_source_report(results, source_dir, source)
        for model in MODELS:
            data = datasets[model, source]
            row = results["probes"][f"{model}/{source}"]
            completed_draws = len(row["permutations"])
            if completed_draws > args.permutations:
                raise ValueError(f"Too many saved permutations for {model}/{source}")
            rng = np.random.default_rng(0)
            for draw in range(args.permutations):
                labels = data["y"].copy()
                for scene in np.unique(data["scene"]):
                    indices = np.flatnonzero(data["scene"] == scene)
                    labels[indices] = rng.permutation(labels[indices])
                if draw < completed_draws:
                    continue
                null_pred, _, null_folds = readout(data["x"], labels, data["scene"], args.workers,
                                                f"{model}/{source}/permutation{draw+1}")
                null_row = metrics(labels, null_pred)
                null_row.update(folds=null_folds, converged=all(f["converged"] for f in null_folds))
                row["permutations"].append(null_row)
                (source_dir / "progress_probe.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
                (output / "progress_probe_partial.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
                write_source_report(results, source_dir, source)
                print(f"{source}/{model} permutation {draw+1}/{args.permutations}", flush=True)
    (output / "progress_probe.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    write_final_report(results, output)


if __name__ == "__main__":
    main()
