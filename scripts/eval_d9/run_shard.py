"""Replay one original D8 rank with its four-environment batch layout."""

import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_rank(model, source, directory, metrics_directory):
    import numpy as np

    from scripts.eval_d9.verify import METRICS, MISSING, check_dump, read_metrics

    rank = int(os.environ["D9_REFERENCE_RANK"])
    with gzip.open(ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz", "rt") as stream:
        dataset_ids = {str(episode["episode_id"]) for episode in json.load(stream)["episodes"]}
    with gzip.open(ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen_gt.json.gz", "rt") as stream:
        ordered_ids = list(json.load(stream))
    expected = set(ordered_ids[rank::4]) & dataset_ids
    if source == "S" or model != "base":
        expected -= MISSING
    directory = Path(directory)
    reference = ROOT / "data/logs/d8_dump" / model / "S" if source == "S" else None
    with tempfile.TemporaryDirectory(prefix="d9-rank-check-") as temporary:
        view = Path(temporary) / "dump"
        view.mkdir()
        ids_path = Path(temporary) / "ids.json"
        ids_path.write_text(json.dumps(sorted(expected)))
        for eid in expected:
            path = directory / f"{eid}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            (view / path.name).symlink_to(path.resolve())
            if reference is not None:
                with np.load(path, allow_pickle=False) as actual, np.load(reference / path.name, allow_pickle=False) as original:
                    if not np.array_equal(actual["heading"], original["heading"]):
                        raise ValueError(f"{eid}: D8 heading mismatch")
        check_dump(SimpleNamespace(directory=str(view), episodes_json=str(ids_path),
                                   source=source, model=model,
                                   reference_dump=str(reference) if reference else ""))
        result = json.loads((view / ".complete.json").read_text())
        result.update(logical_rank=rank, reference_world_size=4, num_environments=4)
        (directory / f".rank{rank}.complete.json").write_text(json.dumps(result, indent=2) + "\n")
    if source == "O":
        metric_ids = set(ordered_ids[rank::4]) & dataset_ids
        actual = read_metrics(metrics_directory, len(metric_ids))
        reference_path = ROOT / f"data/logs/checkpoints/d8_{model}_O_control_val_unseen/eval_results"
        paths = list(reference_path.glob(f"stats_ep_*_val_unseen_r{rank}_w4.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one D8 metrics file: {reference_path}, rank {rank}")
        original = json.loads(paths[0].read_text())
        if set(actual) != metric_ids or set(original) != metric_ids:
            raise ValueError(f"{model}/O/rank{rank}: metrics episode sets differ")
        changes = [dict(episode=eid, field=key, reference=original[eid][key], candidate=actual[eid][key])
                   for eid in sorted(metric_ids) for key in METRICS
                   if original[eid][key] != actual[eid][key]]
        comparison = dict(model=model, source=source, logical_rank=rank,
                          physical_world_size=1, reference_world_size=4,
                          episodes=len(actual), fields=METRICS, changes=changes,
                          reference=str(paths[0]), candidate=str(metrics_directory),
                          comparison_only=True)
        output = ROOT / f"reports/d9/{model}_own_d8_reference_rank{rank}.json"
        output.write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"D8 own comparison: {len(changes)} changed metric values; {output}", flush=True)


def main():
    rank = int(os.environ["D9_REFERENCE_RANK"])
    if rank not in range(4):
        raise ValueError(f"Invalid original D8 rank: {rank}")
    if sys.argv[1:2] == ["--verify-rank"]:
        verify_rank(*sys.argv[2:])
        return
    import run
    from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer

    original = BaseVLNCETrainer.collect_val_traj

    def collect_reference_rank(trainer):
        original(trainer)
        return list(trainer.gt_data)[rank::4]

    BaseVLNCETrainer.collect_val_traj = collect_reference_rank
    run.main()


if __name__ == "__main__":
    main()
