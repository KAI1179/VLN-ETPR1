"""CPU-only consistency checks and direct-NPZ spot checks for Q1 outputs."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality")


def read_rows(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def main():
    summary = json.loads((OUT / "t2_t4_summary.json").read_text())
    verified = {}
    for split, count in (("val_unseen", 1839), ("val_seen", 778)):
        stages = {stage: read_rows(f"t{stage}_{split}_episodes.jsonl") for stage in (1, 2, 3, 4)}
        for stage, multiplier in ((1, 2), (2, 4), (3, 1), (4, 4)):
            assert len(stages[stage]) == count * multiplier
            assert len({row["episode_id"] for row in stages[stage]}) == count
        assert set(Counter((r["episode_id"], r["condition"]) for r in stages[1]).values()) == {1}
        assert set(Counter((r["episode_id"], r["condition"]) for r in stages[2]).values()) == {1}
        assert set(Counter((r["episode_id"], r["condition"], r["resolution"]) for r in stages[4]).values()) == {1}
        assignment_path = OUT / f"donor_assignment_{split}_seed0.jsonl"
        assert hashlib.sha256(assignment_path.read_bytes()).hexdigest() == summary[split]["donor_assignment_sha256"]
        assignments = {r["episode_id"]: r for r in read_rows(assignment_path.name)}
        for row in assignments.values():
            assert Path(row["scene_id"]).stem != Path(row["donor_scene_id"]).stem
        own_rows = {r["episode_id"]: r for r in stages[2] if r["condition"] == "LLM"}
        for row in stages[2]:
            for kind in ("dense", "reference"):
                region, any_channel = row[f"{kind}_region_hit"], row[f"{kind}_any_channel_hit"]
                assert 0 <= region <= any_channel <= 1
                for fraction in (region, any_channel):
                    hits = fraction * row[f"{kind}_point_count"]
                    assert abs(hits - round(hits)) < 1e-8
                    assert round(hits) <= row[f"{kind}_inside_count"]
            if row["condition"] == "LLM_xz_swapped":
                for field in ("region_cells_0p5m", "region_cells_1m", "origin_xz"):
                    assert row[field] == own_rows[row["episode_id"]][field]
            if row["condition"] == "cross_scene_donor":
                assert row["origin_xz"] == own_rows[row["episode_id"]]["origin_xz"]
                assert row["donor_episode_id"] == assignments[row["episode_id"]]["donor_episode_id"]
        for row in stages[3]:
            pv, gv = np.asarray(row["predicted_vectors"]), np.asarray(row["gt_vectors"])
            pn, gn = np.linalg.norm(pv, axis=1) > 0, np.linalg.norm(gv, axis=1) > 0
            assert int(pn.sum()) == row["predicted_nonzero"] and int(gn.sum()) == row["gt_nonzero"]
            assert row["count_difference_pred_minus_gt"] == int(pn.sum()) - int(gn.sum())
            assert row["common_nonzero_slots"] == int((pn & gn).sum())
            for slot, value in enumerate(row["position_aligned_cosines"]):
                assert (value is not None) == bool(pn[slot] and gn[slot])
                if value is not None:
                    direct = float(np.dot(pv[slot], gv[slot]) / (np.linalg.norm(pv[slot]) * np.linalg.norm(gv[slot])))
                    assert math.isfinite(value) and abs(value - direct) < 2e-6
        for row in stages[4]:
            assert len(row["channel_ious"]) == 37
            for intersection, union, value in zip(row["channel_intersections"], row["channel_unions"], row["channel_ious"]):
                assert 0 <= intersection <= union
                assert (value is None) == (union == 0)
                if value is not None:
                    assert math.isfinite(value) and abs(value - intersection / union) < 1e-12
        for condition, group in summary[split]["T2"].items():
            selected = [r for r in stages[2] if r["condition"] == condition]
            for key in ("dense_region_hit", "dense_any_channel_hit", "reference_region_hit", "reference_any_channel_hit"):
                assert math.isclose(sum(r[key] for r in selected) / count, group[key], abs_tol=1e-12)
        directions = summary[split]["T3"]
        assert directions["nonzero_count_equal"] == sum(r["count_equal"] for r in stages[3])
        assert sum(directions["count_difference_distribution"].values()) == count
        cosines = [v for r in stages[3] for v in r["position_aligned_cosines"] if v is not None]
        assert len(cosines) == directions["common_nonzero_pair_count"]
        assert math.isclose(sum(cosines) / len(cosines), directions["aligned_cosine_pair_pooled"], abs_tol=1e-12)
        for group in summary[split]["T4"]:
            selected = [r for r in stages[4] if r["condition"] == group["condition"] and r["resolution"] == group["resolution"]]
            for key in ("region_mean_iou", "object_mean_iou", "region_fixed_channel_zero_empty_iou", "object_fixed_channel_zero_empty_iou"):
                assert math.isclose(sum(r[key] for r in selected) / count, group[key], abs_tol=1e-12)
        report = (OUT / f"map_quality_{split}.md").read_text()
        assert "Not executed:" not in report and "T0–T4 complete." in report
        # Independently reproduce IoU counts from untouched original NPZ files
        # for three numeric episode IDs, both conditions and both resolutions.
        provenance = {r["episode_id"]: r for r in read_rows(f"t0_{split}_episodes.jsonl")}
        samples = sorted(provenance, key=int)[:3]
        indexed4 = {(r["episode_id"], r["condition"], r["resolution"]): r for r in stages[4]}
        for eid in samples:
            with np.load(provenance[eid]["gt_cache_path"], allow_pickle=True) as data:
                target = data["grid"] > 0
            for condition, source_id in (("LLM", eid), ("cross_scene_donor", assignments[eid]["donor_episode_id"])):
                with np.load(provenance[source_id]["effective_cache_path"], allow_pickle=True) as data:
                    predicted = data["grid"] > 0
                for resolution, factor in (("1m_50x50", 2), ("5m_10x10", 10)):
                    side = 100 // factor
                    pred = predicted.reshape(37, side, factor, side, factor).max(axis=(2, 4))
                    truth = target.reshape(37, side, factor, side, factor).max(axis=(2, 4))
                    row = indexed4[(eid, condition, resolution)]
                    assert (pred & truth).sum(axis=(1, 2)).tolist() == row["channel_intersections"]
                    assert (pred | truth).sum(axis=(1, 2)).tolist() == row["channel_unions"]
        verified[split] = {"episodes": count, "stage_record_counts": {str(k): len(v) for k, v in stages.items()},
                           "donors_cross_scene": True, "donor_assignment_unchanged": True,
                           "path_hit_and_denominator_checks": "passed", "direction_cosine_checks": "passed",
                           "all_channel_iou_arithmetic": "passed", "aggregate_recomputation": "passed",
                           "direct_npz_spot_check_ids": samples,
                           "direct_npz_spot_checks": "passed"}
    (OUT / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
