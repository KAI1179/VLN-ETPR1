"""CPU-only arithmetic and independent geometric spot checks for Q2."""

from functools import lru_cache
import json
import math

import numpy as np

from q2_metrics import GROUPS, LABELS, OUT, TAUS


def rows(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines()]


def slow_edit(a, b):
    @lru_cache(None)
    def recur(i, j):
        if not i:
            return j
        if not j:
            return i
        return min(recur(i - 1, j) + 1, recur(i, j - 1) + 1, recur(i - 1, j - 1) + (a[i - 1] != b[j - 1]))
    return recur(len(a), len(b))


def main():
    summary = json.loads((OUT / "q2_summary.json").read_text())
    q1 = json.loads((OUT / "t1_summary.json").read_text())
    q1_directions = json.loads((OUT / "t2_t4_summary.json").read_text())
    t0 = json.loads((OUT / "t0_summary.json").read_text())
    verification = {}
    for split, expected in (("val_unseen", 1839), ("val_seen", 778)):
        data = {stage: rows(f"q2_t{stage}_{split}_episodes.jsonl") for stage in (5, 6, 7)}
        for stage in data:
            assert len(data[stage]) == expected == len({row["episode_id"] for row in data[stage]})
        indexed = {stage: {r["episode_id"]: r for r in data[stage]} for stage in data}
        for eid in t0[split]["missing_episode_ids"]:
            assert sum(indexed[5][eid]["tolerances"][0]["predicted_cells"]) == 0
            assert not indexed[6][eid]["channels"]
            assert all(x is None for x in indexed[7][eid]["original_labels"])
        for row in data[5]:
            previous = None
            for tau in row["tolerances"]:
                pp, rr = np.array(tau["matched_predicted_cells"]), np.array(tau["matched_gt_cells"])
                pred, gt = np.array(tau["predicted_cells"]), np.array(tau["gt_cells"])
                assert np.all((pp >= 0) & (pp <= pred)) and np.all((rr >= 0) & (rr <= gt))
                if tau["tau_m"] == 0:
                    assert np.array_equal(pp, rr)
                if previous is not None:
                    assert np.all(pp >= previous[0]) and np.all(rr >= previous[1])
                previous = pp, rr
        for aggregate in summary[split]["T5"]:
            index = TAUS.index(aggregate["tau_m"])
            selected = [r["tolerances"][index]["groups"][aggregate["group"]] for r in data[5]]
            for key, value in aggregate["pooled_counts"].items():
                assert sum(r[key] for r in selected) == value
            for key in ("precision", "recall", "f1"):
                assert math.isclose(sum(r[key] for r in selected) / expected, aggregate["episode_macro"][key], abs_tol=1e-12)
        for row in data[6]:
            assert math.isclose(float(np.linalg.norm(row["heading_grid_xz"])), 1.0, abs_tol=1e-10)
            for group, item in row["groups"].items():
                assert item["common_categories"] == item["side_eligible"] + item["centered_either"]
                assert item["total_pairs"] == math.comb(item["common_categories"], 2)
                assert item["total_pairs"] == item["order_eligible"] + item["tie_either"]
                assert item["side_correct"] <= item["side_eligible"] and item["order_correct"] <= item["order_eligible"]
                for pair in item["pairs"]:
                    excluded = abs(pair["pred_difference_m"]) < 1 or abs(pair["gt_difference_m"]) < 1
                    assert excluded == pair["excluded"]
        for group in GROUPS:
            assert sum(r["groups"][group]["common_categories"] for r in data[6]) == q1[split]["conditions"]["LLM"][group + "_all"]["tp"]
        for variant in ("original", "swapped", "global_majority", "position_majority"):
            matrices = np.zeros((5, 4, 4), dtype=int)
            for row in data[7]:
                heading = np.array(row["heading_display"])
                assert np.isfinite(heading).all() and math.isclose(float(np.linalg.norm(heading)), 1.0, abs_tol=1e-10)
                for i, (a, b) in enumerate(zip(row["gt_labels"], row[variant + "_labels"])):
                    if a is not None and b is not None:
                        matrices[i, LABELS.index(a), LABELS.index(b)] += 1
                metric = row[variant + "_metrics"]
                gt, pred = metric["gt_sequence"], metric["pred_sequence"]
                assert metric["sequence_equal"] == (gt == pred)
                assert metric["left_count_equal"] == (gt.count("left") == pred.count("left"))
                assert metric["right_count_equal"] == (gt.count("right") == pred.count("right"))
                length = max(len(gt), len(pred))
                assert metric["normalized_edit_distance"] == (metric["edit_distance"] / length if length else 0.0)
                if int(row["episode_id"]) in (1, 2, 3):
                    assert slow_edit(tuple(gt), tuple(pred)) == metric["edit_distance"]
            stated = summary[split]["T7"][variant]
            assert matrices.sum(axis=0).tolist() == stated["confusion_all"]
            assert matrices[0].tolist() == stated["confusion_d1"]
            assert stated["aligned_support"] == q1_directions[split]["T3"]["common_nonzero_pair_count"]
            for a, b in (("sequence_exact_accuracy", "sequence_equal"), ("mean_normalized_edit_distance", "normalized_edit_distance"),
                         ("left_count_accuracy", "left_count_equal"), ("right_count_accuracy", "right_count_equal")):
                assert math.isclose(stated[a], sum(r[variant + "_metrics"][b] for r in data[7]) / expected, abs_tol=1e-12)
        original = np.array(summary[split]["T7"]["original"]["confusion_all"])
        mirrored = np.array(summary[split]["T7"]["swapped"]["confusion_all"])
        assert np.array_equal(original.sum(axis=1), mirrored.sum(axis=1))
        assert summary[split]["T7"]["global_majority"]["aligned_accuracy"] == original.sum(axis=1).max() / original.sum()
        # Independent brute-force nearest-centre distances and centroids on
        # three episodes; no distance-transform helper is used in this check.
        provenance = {r["episode_id"]: r for r in rows(f"t0_{split}_episodes.jsonl")}
        indexed5 = {r["episode_id"]: r for r in data[5]}
        indexed6 = {r["episode_id"]: r for r in data[6]}
        samples = sorted(provenance, key=int)[:3]
        for eid in samples:
            source = provenance[eid]
            with np.load(source["effective_cache_path"], allow_pickle=True) as f:
                pred = f["grid"] > 0
            with np.load(source["gt_cache_path"], allow_pickle=True) as f:
                gt = f["grid"] > 0
            for channel in range(37):
                p, g = np.argwhere(pred[channel]), np.argwhere(gt[channel])
                if len(p) and len(g):
                    distances = np.sqrt(((p[:, None, :] - g[None, :, :]) ** 2).sum(axis=2)) * 0.5
                    near_gt, near_pred = distances.min(axis=1), distances.min(axis=0)
                    for i, tau in enumerate(TAUS):
                        row = indexed5[eid]["tolerances"][i]
                        assert int((near_gt <= tau).sum()) == row["matched_predicted_cells"][channel]
                        assert int((near_pred <= tau).sum()) == row["matched_gt_cells"][channel]
                    saved = next(r for r in indexed6[eid]["channels"] if r["category"] == channel)
                    assert np.allclose((p.mean(axis=0) + 0.5) * 0.5, saved["pred_centroid_m"], atol=1e-12)
                    assert np.allclose((g.mean(axis=0) + 0.5) * 0.5, saved["gt_centroid_m"], atol=1e-12)
        verification[split] = {"episodes_per_stage": expected, "unique_ids": "passed",
                               "zero_filled_episodes_checked": len(t0[split]["missing_episode_ids"]),
                               "tolerance_bounds_and_monotonicity": "passed",
                               "relation_support_counts": "passed", "common_categories_match_Q1_T1": "passed",
                               "turn_confusion_recomputation": "passed", "turn_support_matches_Q1_T3": "passed",
                               "sequence_and_majority_baselines": "passed", "brute_force_geometry_ids": samples,
                               "brute_force_geometry": "passed"}
    examples = json.loads((OUT / "q2_left_sign_examples.json").read_text())
    assert len(examples) == 3 and all(r["display_cross"] > 0 and r["world_xz_cross"] < 0 for r in examples)
    (OUT / "q2_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
