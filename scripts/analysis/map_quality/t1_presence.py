"""Q1 T1: CPU-only category presence, GT mention flags and seeded donors."""

from __future__ import annotations

import importlib.util
import json
import sys

from t0_audit import CACHE, EFFECTIVE, GT, OUT, ROOT, cpu_imports


def load_donor_module():
    path = ROOT / "scripts/distill/make_raster_donor_cache.py"
    spec = importlib.util.spec_from_file_location("q1_original_donor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def aggregate(rows, evaluation):
    macro = evaluation._aggregate_metrics(rows)
    result = {}
    for group in ("region", "object"):
        for partition in ("all", "mentioned", "unmentioned"):
            stem = f"{'' if partition == 'all' else partition + '_'}{group}_category"
            tp, predicted, target = (
                sum(row[f"{stem}_{suffix}"] for row in rows)
                for suffix in ("true_positive_count", "predicted_count", "target_count")
            )
            result[f"{group}_{partition}"] = {
                "episodes": len(rows), "tp": tp, "predicted": predicted, "target": target,
                "micro": {
                    "precision": evaluation._safe_div(tp, predicted),
                    "recall": evaluation._safe_div(tp, target),
                    "f1": evaluation._safe_div(2 * tp, predicted + target),
                },
                "macro": {metric: macro[f"{stem}_{metric}"] for metric in ("precision", "recall", "f1")},
            }
    return result


def main():
    evaluation, navigation, maps = cpu_imports()
    import numpy as np
    from prior.llm_grid_samples import downsample_grid
    from vlnce_baselines.models.etp_llm.llm_grid_train import _load_grid_mentions

    donor_module = load_donor_module()
    summary = {}
    for split, expected in (("val_unseen", 1839), ("val_seen", 778)):
        episodes = donor_module._load_episodes(split)
        complete = [ep for ep in episodes if navigation.llm_navigation_cache_complete(
            ep.scene_id, ep.cache_id, "R2R", split, cache_dir=CACHE, model_key=EFFECTIVE,
        )]
        assert len(episodes) == len(complete) == expected
        donors = donor_module.assign_donors(complete, "cross_scene", seed=0)
        grids = {}
        zero_filled = set()
        for ep in complete:
            path = navigation.llm_navigation_cognitive_map_raster_path(
                ep.scene_id, ep.cache_id, "R2R", split, cache_dir=CACHE, model_key=EFFECTIVE,
            )
            tensors = maps.cognitive_map_file_to_tensors(path, metadata_schema="direction5")
            grids[ep.episode_id] = downsample_grid(tensors["grid"].numpy(), 2) > 0
            assert grids[ep.episode_id].shape == (37, 50, 50)
            status_path = navigation.llm_navigation_status_path(
                ep.scene_id, ep.cache_id, "R2R", split, cache_dir=CACHE, model_key=EFFECTIVE,
            )
            if json.loads(status_path.read_text()).get("zero_filled"):
                zero_filled.add(ep.episode_id)
        records = {name: [] for name in ("LLM", "cross_scene_donor")}
        assignment_rows = []
        with (OUT / f"t1_{split}_episodes.jsonl").open("w", encoding="utf-8") as stream:
            for ep in episodes:
                gt_path = maps.cognitive_map_cache_path(ep.scene_id, ep.cache_id, namespace=GT)
                gt_tensors = maps.cached_cognitive_map_to_tensors(
                    ep.scene_id, ep.cache_id, namespace=GT, metadata_schema="direction5",
                )
                target = downsample_grid(gt_tensors["grid"].numpy(), 2) > 0
                objects, regions = _load_grid_mentions(gt_path)
                donor = donors[ep.episode_id]
                assert donor_module._scene_of(ep) != donor_module._scene_of(donor)
                assignment_rows.append({
                    "episode_id": ep.episode_id, "scene_id": ep.scene_id,
                    "donor_episode_id": donor.episode_id, "donor_scene_id": donor.scene_id,
                    "target_zero_filled": ep.episode_id in zero_filled,
                    "donor_zero_filled": donor.episode_id in zero_filled,
                })
                for condition, pred in (
                    ("LLM", grids[ep.episode_id]),
                    ("cross_scene_donor", grids[donor.episode_id]),
                ):
                    metrics = evaluation._category_presence_metrics(pred, target, objects, regions)
                    records[condition].append(metrics)
                    stream.write(json.dumps({
                        "episode_id": ep.episode_id, "scene_id": ep.scene_id,
                        "condition": condition,
                        "donor_episode_id": donor.episode_id if condition != "LLM" else None,
                        "gt_mentioned_objects": sorted(objects), "gt_mentioned_regions": sorted(regions),
                        "source_zero_filled": ep.episode_id in zero_filled,
                        "predicted_present_categories": np.flatnonzero(pred.any(axis=(1, 2))).tolist(),
                        "gt_present_categories": np.flatnonzero(target.any(axis=(1, 2))).tolist(),
                        "metrics": metrics,
                    }, ensure_ascii=False) + "\n")
        with (OUT / f"donor_assignment_{split}_seed0.jsonl").open("w", encoding="utf-8") as stream:
            for row in assignment_rows:
                stream.write(json.dumps(row) + "\n")
        summary[split] = {
            "episodes": expected, "cache_key": EFFECTIVE, "cache_dir": str(CACHE),
            "gt_namespace": GT, "donor_seed": 0, "donor_pool_size": len(complete),
            "zero_filled_targets": len(zero_filled),
            "zero_filled_donor_assignments": sum(row["donor_zero_filled"] for row in assignment_rows),
            "conditions": {name: aggregate(rows, evaluation) for name, rows in records.items()},
        }
        report = OUT / f"map_quality_{split}.md"
        prefix = report.read_text().split("## T1–T4", 1)[0]
        # Re-running replaces this stage, rather than duplicating tables.
        prefix = prefix.split("<!-- Q1_T0B_BEGIN -->", 1)[0]
        lines = [prefix.rstrip(), "", "<!-- Q1_T0B_BEGIN -->", "## T0b: teacher and distillation evidence", "",
                 (OUT / "t0b_teacher_distillation.md").read_text().replace(
                     "T1–T4 remain paused.", "This T0b snapshot was recorded before T1."
                 ), "<!-- Q1_T0B_END -->", "",
                 "## T1: category presence independent of position", "",
                 "All values below are fractions in [0,1]. Each cell lists P / R / F1.", "",
                 "| Condition | Category type | GT mention partition | Micro P / R / F1 | Macro P / R / F1 |",
                 "|---|---|---|---|---|"]
        for condition, groups in summary[split]["conditions"].items():
            for group, values in groups.items():
                category, partition = group.split("_", 1)
                columns = [" / ".join(f"{values[method][metric]:.10f}" for metric in ("precision", "recall", "f1"))
                           for method in ("micro", "macro")]
                lines.append(f"| {condition} | {category} | {partition} | {columns[0]} | {columns[1]} |")
        lines += ["",
                  "Definition: original _category_presence_metrics on GT and predicted 37×50×50 grids after original 2×2 max-pooling and >0 binarization; micro pools episode-category TP/predicted/target counts, macro averages per-episode P/R/F1, and zero denominators return 0 (episodes with empty partitions remain in macro).", "",
                  "Mention partitions use only the target GT boxes payload's existing mentioned flags via original _load_grid_mentions, not instruction re-extraction or donor flags.", "",
                  f"Donor definition: original make_raster_donor_cache.assign_donors(cross_scene, seed=0), with replacement, on the complete actual final-evaluation cache ({len(complete)} episodes). Only the raster changes; target GT and target metadata/mentions stay fixed. This uses {EFFECTIVE}, not the earlier source-only donor cache.", "",
                  f"Zero-filled target count: {len(zero_filled)}; zero-filled donor selections: {summary[split]['zero_filled_donor_assignments']}. Both are included in denominators; no episode is dropped.", "",
                  f"Per-episode values: {OUT}/t1_{split}_episodes.jsonl", "",
                  f"Donor assignments reused by later stages: {OUT}/donor_assignment_{split}_seed0.jsonl", "",
                  "## T2–T4", "", "Not executed: paused after T1 as requested.", ""]
        report.write_text("\n".join(lines), encoding="utf-8")
        print(split, json.dumps(summary[split], ensure_ascii=False), flush=True)
    (OUT / "t1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
