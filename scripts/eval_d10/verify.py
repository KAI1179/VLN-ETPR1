"""CPU-only D10 shared-frame alignment and hook equivalence checks."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval_d9.verify import METRICS, check_dump, read_metrics


class Args(Tap):
    mode: str
    directory: str = ""
    reference_dump: str = ""
    partial: bool = False
    reference_metrics: str = ""
    candidate_metrics: str = ""
    output: str = ""


def alignment(directory, reference_directory, partial=False):
    directory, reference_directory = Path(directory), Path(reference_directory)
    paths = sorted(directory.glob("*.npz"))
    actual = {path.stem for path in paths}
    expected = {path.stem for path in reference_directory.glob("*.npz")}
    if not actual or not actual <= expected or (not partial and actual != expected):
        raise ValueError(f"Shared episode sets differ: {len(actual)} Z, {len(expected)} M")
    # Reuse D9's full schema/probability validation, but put its completion
    # marker in a temporary view, never in D9 or the input dump directory.
    with tempfile.TemporaryDirectory(prefix="d10-schema-") as temporary:
        view = Path(temporary) / "dump"
        view.mkdir()
        ids_path = Path(temporary) / "ids.json"
        ids_path.write_text(json.dumps(sorted(actual)))
        for path in paths:
            (view / path.name).symlink_to(path.resolve())
        check_dump(SimpleNamespace(
            directory=str(view), episodes_json=str(ids_path), source="S",
            model="try5", reference_dump="",
        ))
    frames = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as zero, np.load(
            reference_directory / path.name, allow_pickle=False
        ) as mapped:
            for key in ("step", "pos", "heading", "candidate_ids",
                        "candidate_positions", "executed_id", "map_start_position",
                        "map_origin_xz", "meters_per_cell"):
                if not np.array_equal(zero[key], mapped[key]):
                    raise ValueError(f"{path.stem}: Z/M mismatch in {key}")
            frames += len(zero["step"])
    if not partial and (len(actual), frames) != (1833, 13992):
        raise ValueError(f"Expected 1833 episodes/13992 frames, got {len(actual)}/{frames}")
    return {
        "episodes": len(actual), "frames": frames, "partial": partial,
        "directory": str(directory.resolve()),
        "reference_dump": str(reference_directory.resolve()), "aligned": True,
    }


def smoke(reference_directory, candidate_directory):
    reference = read_metrics(reference_directory, 4)
    candidate = read_metrics(candidate_directory, 4)
    if reference.keys() != candidate.keys():
        raise ValueError("Hook-off/on completed episode IDs differ")
    changes = [
        {"episode": eid, "field": field,
         "reference": reference[eid][field], "candidate": candidate[eid][field]}
        for eid in sorted(reference) for field in METRICS
        if reference[eid][field] != candidate[eid][field]
    ]
    return {
        "episodes": 4, "episode_ids": sorted(reference), "fields": METRICS,
        "changes": changes, "reference_metrics": str(Path(reference_directory).resolve()),
        "candidate_metrics": str(Path(candidate_directory).resolve()),
    }


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    if args.mode == "alignment":
        result = alignment(args.directory, args.reference_dump, args.partial)
    elif args.mode == "smoke":
        result = smoke(args.reference_metrics, args.candidate_metrics)
    else:
        raise ValueError(f"Unknown verification mode: {args.mode}")
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="")
    if result.get("changes"):
        raise ValueError(f"{len(result['changes'])} hook-on/off metric mismatches")


if __name__ == "__main__":
    main()
