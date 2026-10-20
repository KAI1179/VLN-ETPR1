#!/usr/bin/env python3
"""CPU-only pre-launch check for the RxR-CE KD student (no simulator, no GPU).

Prints, for the chosen language setting, how many training episodes have a
student LLM map, whether every one of them also has a teacher GT map, the
raster formats of each map source, and the GT teacher checkpoint's embedded
config.  With --require any problem that would crash or silently change the
run is an error.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python scripts/distill/check_rxr_kd.py --setting en
"""
import gzip
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from tap import Tap

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cache_report,
    llm_navigation_cognitive_map_raster_path,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path


class Args(Tap):
    setting: str  # en | all
    require: bool = False


def _env(name):
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is not set (run through scripts/distill/run_rxr_kd.sh)")
    return value


def _npz_summary(path):
    data = np.load(path, allow_pickle=True)
    return {k: list(data[k].shape) for k in data.files}


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    problems = []
    cfg = get_config(f"{root}/run_rxr/iter_train.yaml,{root}/run_rxr/kd_{args.setting}.yaml")
    languages = list(cfg.TASK_CONFIG.DATASET.LANGUAGES)
    assert cfg.MODEL.task_type == "rxr" and list(cfg.TASK_CONFIG.DATASET.ROLES) == ["guide"]
    suffix = os.environ.get("DATASET_SUFFIX", "_90")
    if not (root / "data/wp_pred/check_cwp_bestdist_hfov63").is_file():
        problems.append("missing data/wp_pred/check_cwp_bestdist_hfov63")

    source = root / f"data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide{suffix}.json.gz"
    with gzip.open(source, "rt") as stream:
        episodes = [
            e for e in json.load(stream)["episodes"]
            if "*" in languages or e["instruction"]["language"] in languages
        ]
    lang_of = {str(e["episode_id"]): e["instruction"]["language"] for e in episodes}
    scene_of = {str(e["episode_id"]): e["scene_id"] for e in episodes}

    cache_dir = os.environ.get("LLM_CACHE_DIR") or None
    key = _env("LLM_CACHE_KEY")
    report = llm_navigation_cache_report(
        "rxr", "train", require_boxes=False, cache_dir=cache_dir, model_key=key,
        languages=languages,
    )
    available = [e for e in report.available_episode_ids if e in lang_of]
    missing_llm = [e for e in lang_of if e not in set(available)]

    namespace = _env("GT_TEACHER_NAMESPACE")
    missing_gt = [
        e for e in available
        if not cognitive_map_cache_path(scene_of[e], f"RxR_train_{e}", namespace=namespace).is_file()
    ]

    formats = {}
    by_lang = {}
    for e in available:
        by_lang.setdefault(lang_of[e], e)
    for lang, e in sorted(by_lang.items()):
        llm = llm_navigation_cognitive_map_raster_path(
            scene_of[e], f"RxR_train_{e}", "rxr", "train", cache_dir=cache_dir, model_key=key)
        formats[f"llm/{lang}"] = _npz_summary(llm.resolve())
        gt = cognitive_map_cache_path(scene_of[e], f"RxR_train_{e}", namespace=namespace)
        if gt.is_file():
            formats[f"gt/{lang}"] = _npz_summary(gt)
    llm_grids = {str(v.get("grid")) for k, v in formats.items() if k.startswith("llm/")}
    if len(llm_grids) > 1:
        problems.append(f"LLM raster grid shapes differ across languages: {llm_grids}")

    print(json.dumps({
        "setting": args.setting,
        "languages": languages,
        "train_file": str(source),
        "train_episodes_by_language": dict(Counter(lang_of.values())),
        "llm_cache": f"{cache_dir or '<default>'}/{key}",
        "llm_available_by_language": dict(Counter(lang_of[e] for e in available)),
        "llm_missing_by_language": dict(Counter(lang_of[e] for e in missing_llm)),
        "teacher_namespace": namespace,
        "teacher_gt_missing_for_available_students": len(missing_gt),
        "teacher_gt_missing_first": missing_gt[:20],
        "raster_formats": formats,
    }, indent=2))
    if not available:
        problems.append("no training episode has an LLM navigation cache")
    if missing_gt:
        problems.append(f"{len(missing_gt)} student episodes have no teacher GT map")

    import torch

    for name in ("PRETRAINED_CKPT", "GT_TEACHER_CKPT"):
        if not Path(_env(name)).is_file():
            problems.append(f"{name} not found: {os.environ[name]}")
    teacher_path = Path(os.environ["GT_TEACHER_CKPT"])
    if teacher_path.is_file():
        ckpt = torch.load(teacher_path, map_location="cpu")
        saved = ckpt.get("config")
        model = getattr(saved, "MODEL", None)
        map_cfg = getattr(model, "MAP_ENCODER", None)
        dataset = getattr(getattr(saved, "TASK_CONFIG", None), "DATASET", None)
        info = {
            "policy_name": getattr(model, "policy_name", None),
            "task_type": getattr(model, "task_type", None),
            "elevation_axis": getattr(model, "elevation_axis", None),
            "map_architecture": getattr(map_cfg, "architecture", None),
            "map_source": getattr(map_cfg, "source", None),
            "cache_namespace": getattr(map_cfg, "cache_namespace", None),
            "coordinate_fusion": getattr(map_cfg, "coordinate_fusion", None),
            "dataset_languages": getattr(dataset, "LANGUAGES", None),
            "dataset_suffix": getattr(dataset, "SUFFIX", None),
            "iteration": ckpt.get("iteration"),
            "state_keys": len(ckpt["state_dict"]),
            "map_keys": sum(".map_encoder." in k or ".graph_map_attention." in k
                            for k in ckpt["state_dict"]),
        }
        print("teacher_ckpt", json.dumps(info, indent=2, default=str))
        if info["cache_namespace"] not in (None, namespace):
            problems.append(f"teacher cache_namespace {info['cache_namespace']!r} != {namespace!r}")
        if info["map_architecture"] not in (None, "try5"):
            problems.append(f"teacher map architecture {info['map_architecture']!r} is not try5")
        if info["coordinate_fusion"]:
            problems.append("teacher was trained with coordinate_fusion True")
        if info["elevation_axis"] not in (None, "y"):
            problems.append(f"teacher elevation_axis {info['elevation_axis']!r} (student uses y)")
        if not info["map_keys"]:
            problems.append("teacher state_dict has no map_encoder / graph_map_attention weights")

    for problem in problems:
        print("PROBLEM:", problem)
    if args.require and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
