"""Q1 T2–T4: full-split CPU metrics using unchanged project helpers."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import sys

from t0_audit import CACHE, EFFECTIVE, GT, OUT, ROOT, cpu_imports

CONDITIONS = ("LLM", "LLM_xz_swapped", "cross_scene_donor", "GT_legacy")


def load_frame_module():
    path = ROOT / "scripts/distill/check_map_frame.py"
    spec = importlib.util.spec_from_file_location("q1_original_frame", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    evaluation, navigation, maps = cpu_imports()
    import numpy as np
    import torch
    from prior import R2R_DIR
    from prior.constants import OBJECT_CATEGORIES
    from prior.llm_grid_samples import downsample_grid

    frame = load_frame_module()
    summaries = {}
    for split, expected in (("val_unseen", 1839), ("val_seen", 778)):
        with gzip.open(R2R_DIR / split / f"{split}.json.gz", "rt") as stream:
            episodes = json.load(stream)["episodes"]
        dense_path = R2R_DIR / split / f"{split}_gt.json.gz"
        with gzip.open(dense_path, "rt") as stream:
            dense_gt = json.load(stream)
        donor_path = OUT / f"donor_assignment_{split}_seed0.jsonl"
        assignments = [json.loads(line) for line in donor_path.read_text().splitlines()]
        donors = {row["episode_id"]: row for row in assignments}
        assert len(episodes) == len(donors) == expected
        # Keep compact bool grids, not a model or GPU tensors, for donor reuse.
        predicted = {}
        for ep in episodes:
            eid = str(ep["episode_id"])
            tensors = navigation.llm_cached_cognitive_map_to_tensors(
                ep["scene_id"], f"R2R_{split}_{eid}", "R2R", split,
                cache_dir=CACHE, model_key=EFFECTIVE, metadata_schema="direction5",
            )
            predicted[eid] = {key: value.cpu() for key, value in tensors.items()}
            predicted[eid]["grid"] = tensors["grid"] > 0
        t2_rows = {name: [] for name in CONDITIONS}
        t3_rows = []
        t4_rows = {}
        paths = [OUT / f"t{stage}_{split}_episodes.jsonl" for stage in (2, 3, 4)]
        with paths[0].open("w") as file2, paths[1].open("w") as file3, paths[2].open("w") as file4:
            for index, ep in enumerate(episodes):
                eid = str(ep["episode_id"])
                gt = maps.cached_cognitive_map_to_tensors(
                    ep["scene_id"], f"R2R_{split}_{eid}", namespace=GT, metadata_schema="direction5",
                )
                gt_grid = gt["grid"] > 0
                own = predicted[eid]
                donor_id = donors[eid]["donor_episode_id"]
                assert donors[eid]["scene_id"] != donors[eid]["donor_scene_id"]
                donor_grid = predicted[donor_id]["grid"]
                dense = np.asarray(dense_gt[eid]["locations"], dtype=np.float64)
                reference = np.asarray(ep["reference_path"], dtype=np.float64)
                assert dense.ndim == reference.ndim == 2 and dense.shape[1] == reference.shape[1] == 3
                assert len(dense) and len(reference) and np.isfinite(dense).all() and np.isfinite(reference).all()
                for condition, grid, metadata in (
                    ("LLM", own["grid"], own), ("LLM_xz_swapped", own["grid"], own),
                    ("cross_scene_donor", donor_grid, own), ("GT_legacy", gt_grid, gt),
                ):
                    origin = maps.map_origin_xz(ep["start_position"], metadata["start_position"], 1.0)
                    # Reuse the exact original region-hit implementation for
                    # all-channel union by putting that union in region slot 27.
                    all_proxy = grid.new_zeros((OBJECT_CATEGORIES + 1, 100, 100))
                    all_proxy[OBJECT_CATEGORIES] = grid.any(dim=0)
                    row = {"episode_id": eid, "scene_id": ep["scene_id"], "condition": condition,
                           "donor_episode_id": donor_id if condition == "cross_scene_donor" else None,
                           "origin_xz": origin.tolist(), "dense_point_count": len(dense),
                           "reference_point_count": len(reference)}
                    for label, points in (("dense", dense), ("reference", reference)):
                        cells = maps.world_to_map_cells(points, origin)
                        if condition == "LLM_xz_swapped":
                            cells = cells.flip(-1)
                        row[f"{label}_region_hit"] = frame._region_hit_rate(grid, cells)
                        row[f"{label}_any_channel_hit"] = frame._region_hit_rate(all_proxy, cells)
                        cell_index = torch.floor(cells).long()
                        row[f"{label}_inside_count"] = int(((cell_index >= 0) & (cell_index < maps.SIZE)).all(dim=1).sum())
                    row["region_cells_0p5m"] = int(grid[OBJECT_CATEGORIES:].any(dim=0).sum())
                    row["region_cells_1m"] = int(downsample_grid(grid.numpy(), 2)[OBJECT_CATEGORIES:].any(axis=0).sum())
                    t2_rows[condition].append(row)
                    file2.write(json.dumps(row) + "\n")
                pred_vectors = own["map_trajectory_metadata"].numpy()
                gt_vectors = gt["map_trajectory_metadata"].numpy()
                pn = np.linalg.norm(pred_vectors, axis=1) > 0
                gn = np.linalg.norm(gt_vectors, axis=1) > 0
                common = pn & gn
                pair_cosines = [None] * 5
                for slot in np.flatnonzero(common):
                    pair_cosines[int(slot)] = evaluation._direction_vector_metrics(
                        pred_vectors[slot:slot + 1], gt_vectors[slot:slot + 1]
                    )["direction_vector_cosine"]
                mean_cosine = evaluation._direction_vector_metrics(
                    pred_vectors[common], gt_vectors[common]
                )["direction_vector_cosine"] if common.any() else None
                row3 = {"episode_id": eid, "predicted_nonzero": int(pn.sum()), "gt_nonzero": int(gn.sum()),
                        "count_difference_pred_minus_gt": int(pn.sum()) - int(gn.sum()),
                        "count_equal": bool(pn.sum() == gn.sum()), "common_nonzero_slots": int(common.sum()),
                        "position_aligned_cosines": pair_cosines, "episode_mean_cosine": mean_cosine,
                        "d1_cosine": pair_cosines[0], "predicted_vectors": pred_vectors.tolist(), "gt_vectors": gt_vectors.tolist()}
                t3_rows.append(row3)
                file3.write(json.dumps(row3) + "\n")
                for resolution, factor in (("1m_50x50", 2), ("5m_10x10", 10)):
                    def pool(grid):
                        if factor == 2:
                            return downsample_grid(grid.numpy(), 2)
                        return torch.nn.functional.max_pool2d(grid.float(), factor, factor).numpy() > 0
                    target = pool(gt_grid)
                    for condition, grid in (("LLM", own["grid"]), ("cross_scene_donor", donor_grid)):
                        pred = pool(grid)
                        intersections = (pred & target).sum(axis=(1, 2))
                        unions = (pred | target).sum(axis=(1, 2))
                        channel_ious = [float(i / u) if u else None for i, u in zip(intersections, unions)]
                        row4 = {"episode_id": eid, "condition": condition, "resolution": resolution,
                                "channel_intersections": intersections.tolist(), "channel_unions": unions.tolist(),
                                "channel_ious": channel_ious}
                        for group, selection in (("object", slice(0, OBJECT_CATEGORIES)), ("region", slice(OBJECT_CATEGORIES, None))):
                            values = channel_ious[selection]
                            valid_values = [value for value in values if value is not None]
                            row4[f"{group}_mean_iou"] = float(np.mean(valid_values)) if valid_values else 0.0
                            row4[f"{group}_nonempty_union_channels"] = len(valid_values)
                            row4[f"{group}_fixed_channel_zero_empty_iou"] = sum(valid_values) / len(values)
                        t4_rows.setdefault((resolution, condition), []).append(row4)
                        file4.write(json.dumps(row4) + "\n")
                if (index + 1) % 300 == 0:
                    print(f"{split}: {index + 1}/{expected} episodes T2–T4", flush=True)
        summary2 = {}
        for condition, rows in t2_rows.items():
            item = {key: float(np.mean([r[key] for r in rows])) for key in (
                "dense_region_hit", "dense_any_channel_hit", "reference_region_hit", "reference_any_channel_hit",
                "region_cells_0p5m", "region_cells_1m",
            )}
            for key in ("region_cells_0p5m", "region_cells_1m"):
                item[key + "_median"] = float(np.median([r[key] for r in rows]))
            for trajectory in ("dense", "reference"):
                count = sum(r[f"{trajectory}_point_count"] for r in rows)
                item[f"{trajectory}_total_points"] = count
                for label in ("region_hit", "any_channel_hit"):
                    item[f"{trajectory}_{label}_point_pooled"] = sum(r[f"{trajectory}_{label}"] * r[f"{trajectory}_point_count"] for r in rows) / count
            summary2[condition] = item
        common_count = sum(r["common_nonzero_slots"] for r in t3_rows)
        supported = [r["episode_mean_cosine"] for r in t3_rows if r["episode_mean_cosine"] is not None]
        first = [r["d1_cosine"] for r in t3_rows if r["d1_cosine"] is not None]
        summary3 = {"episodes": expected, "nonzero_count_equal": sum(r["count_equal"] for r in t3_rows),
                    "nonzero_count_accuracy": sum(r["count_equal"] for r in t3_rows) / expected,
                    "count_difference_distribution": dict(sorted(Counter(r["count_difference_pred_minus_gt"] for r in t3_rows).items())),
                    "common_nonzero_pair_count": common_count, "cosine_supported_episodes": len(supported),
                    "aligned_cosine_pair_pooled": sum(sum(v for v in r["position_aligned_cosines"] if v is not None) for r in t3_rows) / common_count if common_count else None,
                    "aligned_cosine_episode_macro": float(np.mean(supported)) if supported else None,
                    "d1_supported_episodes": len(first), "d1_mean_cosine": float(np.mean(first)) if first else None}
        summary4 = []
        for (resolution, condition), rows in t4_rows.items():
            item = {"resolution": resolution, "condition": condition, "episodes": expected}
            for group in ("region", "object"):
                item[group + "_mean_iou"] = float(np.mean([r[group + "_mean_iou"] for r in rows]))
                item[group + "_fixed_channel_zero_empty_iou"] = float(np.mean([r[group + "_fixed_channel_zero_empty_iou"] for r in rows]))
                item[group + "_empty_group_episodes"] = sum(r[group + "_nonempty_union_channels"] == 0 for r in rows)
            summary4.append(item)
        summaries[split] = {"episodes": expected, "dense_gt_path": str(dense_path),
                            "donor_assignment_path": str(donor_path), "donor_assignment_sha256": hashlib.sha256(donor_path.read_bytes()).hexdigest(),
                            "T2": summary2, "T3": summary3, "T4": summary4}
        (OUT / f"t2_t4_{split}_summary.json").write_text(json.dumps(summaries[split], indent=2) + "\n")
        report = OUT / f"map_quality_{split}.md"
        prefix = report.read_text().split("## T2–T4", 1)[0].split("<!-- Q1_T2_BEGIN -->", 1)[0]
        lines = [prefix.rstrip(), "", "<!-- Q1_T2_BEGIN -->", "## T2: path hit rate", "",
                 "| Condition | Dense: region | Dense: any channel | Sparse reference: region | Sparse reference: any channel | Region cells 0.5m mean / median | Region cells 1m mean / median |",
                 "|---|---:|---:|---:|---:|---|---|"]
        for condition, item in summary2.items():
            values = [f"{item[k]:.10f}" for k in ("dense_region_hit", "dense_any_channel_hit", "reference_region_hit", "reference_any_channel_hit")]
            coverage = [f"{item[k]:.10f} / {item[k + '_median']:.10f}" for k in ("region_cells_0p5m", "region_cells_1m")]
            lines.append("| " + " | ".join([condition, *values, *coverage]) + " |")
        lines += ["", "Definition: arithmetic mean of per-episode hit fractions, using every dense GT locations point or every sparse reference_path point; points outside the raster count as misses, with no height filtering, clipping, interpolation or deduplication. Region coverage counts unique spatial cells occupied by at least one region channel, not the sum of category occupancies.", "",
                  f"Dense GT source: {dense_path}. This is the same _gt locations source consumed by prior.vlnce.VLNCEEpisodeEntry. Both legacy and LLM start_position values use metres (meters_per_unit=1.0). Original map_origin_xz, world_to_map_cells and check_map_frame._region_hit_rate are reused. X/z swapping flips cell coordinates only; donor changes only the raster and retains the target LLM metadata. Missing predictions retain the final-evaluation all-zero inputs.", "",
                  "## T3: direction vectors", "", "| Metric | Value |", "|---|---:|"]
        for key, value in summary3.items():
            if key != "count_difference_distribution":
                lines.append(f"| {key} | {value} |")
        lines += ["", "| Predicted minus GT nonzero count | Episodes | Fraction |", "|---:|---:|---:|"]
        for difference in range(-5, 6):
            count = summary3["count_difference_distribution"].get(difference, 0)
            lines.append(f"| {difference} | {count} | {count / expected:.10f} |")
        lines += ["", "Definition: nonzero means vector norm >0; count accuracy uses the entire split, whereas cosine uses only position-aligned pairs for which both vectors are nonzero. The original _direction_vector_metrics is called after applying that common-nonzero mask; pair-pooled and supported-episode macro averages are both reported, with missing support recorded as null rather than a cosine of zero. d1 uses slot 1 only when both vectors are nonzero.", "",
                  "## T4: per-category spatial IoU", "",
                  "| Resolution | Condition | Region episode-macro IoU | Object episode-macro IoU | Region fixed-channel zero-empty IoU | Object fixed-channel zero-empty IoU | Empty region / object episodes |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for item in summary4:
            values = [f"{item[k]:.10f}" for k in ("region_mean_iou", "object_mean_iou", "region_fixed_channel_zero_empty_iou", "object_fixed_channel_zero_empty_iou")]
            lines.append("| " + " | ".join([item["resolution"], item["condition"], *values,
                                             f"{item['region_empty_group_episodes']} / {item['object_empty_group_episodes']}"]) + " |")
        lines += ["", "Definition: binarize >0 and max-pool the native 100×100 grid by 2×2 for 1m cells or by 10×10 for 5m patches; compute intersection/union separately in each semantic channel, average channels with nonempty union within each episode, then average all episodes (an entirely empty group contributes zero). The supplemental fixed-channel column assigns zero to channels absent from both maps. Empty-union channel IoUs are null in JSONL. Hallucinated channels with nonzero union remain included.", "",
                  f"T2 per episode: {paths[0]}", "", f"T3 per episode: {paths[1]}", "", f"T4 per episode and channel: {paths[2]}", "",
                  "All tables contain raw values only; no bootstrap, significance tests or interpretation. T0–T4 complete.", ""]
        report.write_text("\n".join(lines))
        print(f"{split}: T2–T4 complete", flush=True)
        del predicted
    (OUT / "t2_t4_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
