"""CPU-only provenance and cache audit; stop before Q1 T1 metrics."""

from __future__ import annotations

from collections import Counter
import gzip
import faulthandler
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
OUT = ROOT / "reports/map_quality"
SOURCE = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
EFFECTIVE = SOURCE + "__missing-all-zero"
CACHE = ROOT / "outputs/kd27000_missing_zero_cache"
GT = "gt.legacy.r1p5.direction5.v1"


def cpu_imports():
    """Load original helper modules without trainer-registration side effects."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    from no_tensorflow import configure_no_tensorflow

    configure_no_tensorflow()
    for name in (
        "vlnce_baselines", "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt", "vlnce_baselines.models.etp_llm",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules[name] = package
    evaluation = importlib.import_module("vlnce_baselines.models.etp_llm.llm_grid_eval")
    navigation = importlib.import_module("vlnce_baselines.models.etp_llm.navigation")
    maps = importlib.import_module("vlnce_baselines.models.etp_prior_gt.map_utils")
    return evaluation, navigation, maps


def main():
    faulthandler.dump_traceback_later(60)
    evaluation, navigation, maps = cpu_imports()
    faulthandler.cancel_dump_traceback_later()
    from prior import R2R_DIR
    from vlnce_baselines.models.etp_llm.llm_grid_train import LLMGridValidationError
    from vlnce_baselines.models.etp_llm.llm_grid_train import _LLMGridJSONError

    OUT.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for split, expected in (("val_unseen", 1839), ("val_seen", 778)):
        dataset = R2R_DIR / split / f"{split}.json.gz"
        with gzip.open(dataset, "rt", encoding="utf-8") as stream:
            episodes = json.load(stream)["episodes"]
        assert len(episodes) == expected
        split_dir = navigation.llm_navigation_split_dir("R2R", split, model_key=SOURCE)
        manifest_path = split_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        counts = Counter()
        missing_ids = []
        with (OUT / f"t0_{split}_episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode in episodes:
                identifier = f"R2R_{split}_{episode['episode_id']}"
                args = (episode["scene_id"], identifier, "R2R", split)
                source_path = navigation.llm_navigation_cognitive_map_raster_path(*args, model_key=SOURCE)
                prediction = navigation.llm_navigation_prediction_path(*args, model_key=SOURCE)
                status_path = navigation.llm_navigation_status_path(*args, model_key=SOURCE)
                effective_path = navigation.llm_navigation_cognitive_map_raster_path(
                    *args, cache_dir=CACHE, model_key=EFFECTIVE,
                )
                effective_status_path = navigation.llm_navigation_status_path(
                    *args, cache_dir=CACHE, model_key=EFFECTIVE,
                )
                gt_path = maps.cognitive_map_cache_path(episode["scene_id"], identifier, namespace=GT)
                status = json.loads(status_path.read_text()) if status_path.is_file() else None
                effective_status = json.loads(effective_status_path.read_text()) if effective_status_path.is_file() else {}
                json_valid = schema_valid = None
                error = None
                if prediction.is_file():
                    try:
                        evaluation.parse_grid_text(prediction.read_text(encoding="utf-8"), shape=(37, 50, 50))
                    except LLMGridValidationError as exception:
                        json_valid = not isinstance(exception, _LLMGridJSONError)
                        schema_valid = False
                        error = str(exception)
                    else:
                        json_valid = schema_valid = True
                reason = None
                if not source_path.is_file():
                    missing_ids.append(str(episode["episode_id"]))
                    if json_valid is False:
                        reason = "json_parse_failed"
                    elif schema_valid is False:
                        reason = "schema_invalid"
                    elif not prediction.is_file() and status is None:
                        reason = "not_generated_no_text_or_status"
                    else:
                        reason = "unconfirmed_write_or_artifact_missing"
                    counts[reason] += 1
                counts["episodes"] += 1
                counts["source_available"] += source_path.is_file()
                counts["source_missing"] += not source_path.is_file()
                counts["raw_text_available"] += prediction.is_file()
                counts["json_valid"] += json_valid is True
                counts["schema_valid"] += schema_valid is True
                counts["effective_available"] += effective_path.is_file()
                counts["effective_zero_filled"] += bool(effective_status.get("zero_filled"))
                counts["gt_available"] += gt_path.is_file()
                record = {
                    "episode_id": str(episode["episode_id"]), "scene_id": episode["scene_id"],
                    "source_cache_path": str(source_path), "source_cache_available": source_path.is_file(),
                    "prediction_path": str(prediction), "raw_text_available": prediction.is_file(),
                    "json_valid": json_valid, "schema_valid": schema_valid,
                    "parse_error": error, "missing_reason": reason, "generation_status": status,
                    "effective_cache_path": str(effective_path), "effective_available": effective_path.is_file(),
                    "effective_zero_filled": bool(effective_status.get("zero_filled")),
                    "gt_cache_path": str(gt_path), "gt_available": gt_path.is_file(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = dict(counts)
        for model_key, cache_dir, expected_count in (
            (SOURCE, None, counts["source_available"]),
            (EFFECTIVE, CACHE, counts["effective_available"]),
        ):
            report = navigation.llm_navigation_cache_report(
                "R2R", split, [str(ep["episode_id"]) for ep in episodes],
                require_boxes=False, cache_dir=cache_dir, model_key=model_key,
            )
            assert report.available_count == expected_count
        local_adapter = Path("/data/xukai/etp-r1-snapshot/ETP-R1") / manifest["model_name_or_path"]
        summary.update({
            "dataset_path": str(dataset), "manifest_path": str(manifest_path),
            "manifest": manifest, "source_key": SOURCE, "effective_key": EFFECTIVE,
            "effective_cache_dir": str(CACHE), "teacher_gt_namespace": GT,
            "missing_episode_ids": sorted(missing_ids, key=int),
            "json_valid_fraction_all_episodes": counts["json_valid"] / expected,
            "schema_valid_fraction_all_episodes": counts["schema_valid"] / expected,
            "json_valid_fraction_available_text": counts["json_valid"] / counts["raw_text_available"],
            "schema_valid_fraction_available_text": counts["schema_valid"] / counts["raw_text_available"],
            "generation_decoding": "do_sample=False in llm_boxes_train._generation_kwargs; not recorded in manifest",
            "located_lora_copy": str(local_adapter),
            "located_lora_copy_exists": local_adapter.is_dir(),
        })
        all_summary[split] = summary
        lines = [
            f"# Predicted map quality: R2R {split}", "", "## T0: scope and provenance", "",
            "| Item | Value |", "|---|---|",
            f"| Navigation checkpoint | {ROOT}/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth |",
            f"| Training config | {ROOT}/data/logs/checkpoints/kd_dual/config.yaml |",
            f"| Final evaluation | {ROOT}/data/logs/checkpoints/kd_full27000_control_val_unseen_missing_zero/eval_results/stats_ckpt_27000_val_unseen.json |",
            "| Final val_unseen SR / SPL (%) | 67.48232735182165 / 56.63107734961897 |",
            f"| Source LLM cache key | {SOURCE} |",
            f"| Actual final-evaluation key | {EFFECTIVE} |",
            f"| Actual final-evaluation cache root | {CACHE} |",
            f"| Teacher GT namespace | {GT} |",
            f"| Located LoRA copy (historical manifest path resolved under /data checkout) | {local_adapter} |",
            f"| Generation max input / new tokens | {manifest['max_input_length']} / {manifest['max_new_tokens']} |",
            f"| Generation scale | {manifest['scale']} |",
            "| Decoding code | do_sample=False; no explicit temperature/top_p/num_beams; manifest does not record these |",
            f"| Generation manifest | {manifest_path} |", "",
            "The manifest records a historical relative model path. The located copy is not a checksum-verified identification of the original generation weights. Decoding settings above are supported by code, not logged in the manifest.", "",
            f"Evidence: {ROOT}/scripts/distill/eval_kd27000_missing_zero.sh; {ROOT}/scripts/eval_d7/d7d_eval.sh; {ROOT}/scripts/distill/make_missing_zero_eval_cache.py; {ROOT}/data/logs/checkpoints/kd_dual/config.yaml; {ROOT}/data/logs/checkpoints/kd_dual/launch_info.txt.", "",
            "| Cache / validity item | Count | Denominator | Fraction |", "|---|---:|---:|---:|",
        ]
        for key in ("source_available", "source_missing", "raw_text_available", "json_valid", "schema_valid", "effective_available", "effective_zero_filled", "gt_available"):
            lines.append(f"| {key} | {counts[key]} | {expected} | {counts[key] / expected:.10f} |")
        lines += ["", "Validity uses the original parser exported by llm_grid_eval at shape (37,50,50); fractions above use all expected episodes. No prediction, training or navigation inference is run.", "",
                  "| Missing source-cache reason | Count |", "|---|---:|"]
        for reason in ("json_parse_failed", "schema_invalid", "not_generated_no_text_or_status", "unconfirmed_write_or_artifact_missing"):
            lines.append(f"| {reason} | {counts[reason]} |")
        lines += ["", "Missing means the source raster file is absent. Missing text alone is not proof that generation was never attempted; retained status records and parser errors are recorded per episode.", "",
                  "Missing episode IDs: " + ", ".join(summary["missing_episode_ids"]), "",
                  "## T1–T4", "", "Not executed: paused after T0 as requested.", ""]
        (OUT / f"map_quality_{split}.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "t0_summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
