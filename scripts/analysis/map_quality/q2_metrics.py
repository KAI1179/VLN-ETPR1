"""CPU-only Q2 tolerance, relative-relation and discrete-turn metrics."""

from __future__ import annotations

import gzip
import itertools
import json
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from t0_audit import CACHE, EFFECTIVE, GT, OUT, ROOT, cpu_imports

LABELS = ("straight", "left", "right", "back")
TAUS = (0.0, 2.5, 5.0)
GROUPS = {"object": slice(0, 27), "region": slice(27, 37)}


def divide(a, b):
    return float(a / b) if b else 0.0


def prf(correct_pred, pred, correct_gt, gt):
    precision, recall = divide(correct_pred, pred), divide(correct_gt, gt)
    return {"precision": precision, "recall": recall, "f1": divide(2 * precision * recall, precision + recall)}


def tolerance(pred, gt):
    predicted = pred.sum(axis=(1, 2)).astype(int)
    target = gt.sum(axis=(1, 2)).astype(int)
    pp = np.zeros((len(TAUS), 37), dtype=np.int64)
    rr = np.zeros_like(pp)
    for category in range(37):
        if not predicted[category] or not target[category]:
            continue
        to_gt = distance_transform_edt(~gt[category], sampling=0.5)[pred[category]]
        to_pred = distance_transform_edt(~pred[category], sampling=0.5)[gt[category]]
        for i, tau in enumerate(TAUS):
            pp[i, category] = int(np.count_nonzero(to_gt <= tau))
            rr[i, category] = int(np.count_nonzero(to_pred <= tau))
    result = []
    for i, tau in enumerate(TAUS):
        row = {"tau_m": tau, "predicted_cells": predicted.tolist(), "gt_cells": target.tolist(),
               "matched_predicted_cells": pp[i].tolist(), "matched_gt_cells": rr[i].tolist(),
               "per_category": [prf(pp[i, c], predicted[c], rr[i, c], target[c]) for c in range(37)], "groups": {}}
        for name, channels in GROUPS.items():
            counts = {"matched_pred": int(pp[i, channels].sum()), "pred": int(predicted[channels].sum()),
                      "matched_gt": int(rr[i, channels].sum()), "gt": int(target[channels].sum())}
            row["groups"][name] = {**counts, **prf(*counts.values())}
        result.append(row)
    assert np.array_equal(pp[0], rr[0])
    assert np.all(np.diff(pp, axis=0) >= 0) and np.all(np.diff(rr, axis=0) >= 0)
    return result


def relative_relations(pred, gt, pred_p0, gt_p0, h_grid):
    channels = []
    for category in range(37):
        if not pred[category].any() or not gt[category].any():
            continue
        pc = (np.argwhere(pred[category]).mean(axis=0) + 0.5) * 0.5
        gc = (np.argwhere(gt[category]).mean(axis=0) + 0.5) * 0.5
        pd, gd = pc - pred_p0, gc - gt_p0
        px = float(h_grid[0] * pd[1] - h_grid[1] * pd[0])
        gx = float(h_grid[0] * gd[1] - h_grid[1] * gd[0])
        channels.append({"category": category, "pred_centroid_m": pc.tolist(), "gt_centroid_m": gc.tolist(),
                         "pred_cross_m": px, "gt_cross_m": gx,
                         "pred_centered": abs(px) < 1.0, "gt_centered": abs(gx) < 1.0,
                         "pred_side": int(np.sign(px)), "gt_side": int(np.sign(gx)),
                         "pred_distance_m": float(np.linalg.norm(pd)), "gt_distance_m": float(np.linalg.norm(gd))})
    result = {"channels": channels, "groups": {}}
    for name, selection in GROUPS.items():
        common = [x for x in channels if selection.start <= x["category"] < selection.stop]
        centered = [x for x in common if x["pred_centered"] or x["gt_centered"]]
        eligible = [x for x in common if not x["pred_centered"] and not x["gt_centered"]]
        correct = sum(x["pred_side"] == x["gt_side"] for x in eligible)
        pairs = []
        for a, b in itertools.combinations(common, 2):
            p = a["pred_distance_m"] - b["pred_distance_m"]
            g = a["gt_distance_m"] - b["gt_distance_m"]
            excluded = abs(p) < 1.0 or abs(g) < 1.0
            pairs.append({"a": a["category"], "b": b["category"], "pred_difference_m": p, "gt_difference_m": g,
                          "pred_tie": abs(p) < 1.0, "gt_tie": abs(g) < 1.0, "excluded": excluded,
                          "consistent": None if excluded else bool(np.sign(p) == np.sign(g))})
        valid_pairs = [p for p in pairs if not p["excluded"]]
        pair_correct = sum(p["consistent"] for p in valid_pairs)
        result["groups"][name] = {
            "common_categories": len(common), "centered_pred": sum(x["pred_centered"] for x in common),
            "centered_gt": sum(x["gt_centered"] for x in common), "centered_either": len(centered),
            "side_eligible": len(eligible), "side_correct": correct, "side_accuracy": divide(correct, len(eligible)),
            "total_pairs": len(pairs), "tie_pred": sum(x["pred_tie"] for x in pairs), "tie_gt": sum(x["gt_tie"] for x in pairs),
            "tie_either": len(pairs) - len(valid_pairs), "order_eligible": len(valid_pairs), "order_correct": pair_correct,
            "order_accuracy": divide(pair_correct, len(valid_pairs)), "pairs": pairs,
        }
    return result


def turn_labels(vectors, heading):
    previous = np.asarray(heading, dtype=np.float64)
    labels, angles = [], []
    for vector in np.asarray(vectors, dtype=np.float64):
        if np.linalg.norm(vector) == 0:
            labels.append(None)
            angles.append(None)
            continue
        cross = previous[0] * vector[1] - previous[1] * vector[0]
        angle = math.degrees(math.atan2(cross, float(np.dot(previous, vector))))
        absolute = abs(angle)
        label = "straight" if absolute < 45 else ("back" if absolute > 135 else ("left" if angle > 0 else "right"))
        labels.append(label)
        angles.append(angle)
        previous = vector
    return labels, angles


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        updated = [i]
        for j, y in enumerate(b, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = updated
    return row[-1]


def sequence_metrics(gt_slots, pred_slots):
    gt, pred = [x for x in gt_slots if x is not None], [x for x in pred_slots if x is not None]
    distance = levenshtein(gt, pred)
    return {"gt_sequence": gt, "pred_sequence": pred, "sequence_equal": gt == pred,
            "edit_distance": distance, "normalized_edit_distance": divide(distance, max(len(gt), len(pred))),
            "left_count_equal": gt.count("left") == pred.count("left"), "right_count_equal": gt.count("right") == pred.count("right"),
            "gt_left_count": gt.count("left"), "pred_left_count": pred.count("left"),
            "gt_right_count": gt.count("right"), "pred_right_count": pred.count("right")}


def summarize_turns(rows, variant):
    matrices = np.zeros((5, 4, 4), dtype=int)
    metrics = []
    for row in rows:
        gt, pred = row["gt_labels"], row[variant + "_labels"]
        for position, (a, b) in enumerate(zip(gt, pred)):
            if a is not None and b is not None:
                matrices[position, LABELS.index(a), LABELS.index(b)] += 1
        item = sequence_metrics(gt, pred)
        row[variant + "_metrics"] = item
        metrics.append(item)
    total = matrices.sum(axis=0)
    result = {"matrix_order": list(LABELS), "confusion_all": total.tolist(), "confusion_d1": matrices[0].tolist(),
              "aligned_support": int(total.sum()), "aligned_accuracy": divide(int(np.trace(total)), int(total.sum())),
              "d1_support": int(matrices[0].sum()), "d1_accuracy": divide(int(np.trace(matrices[0])), int(matrices[0].sum())),
              "positions": [{"position": i + 1, "support": int(m.sum()), "accuracy": divide(int(np.trace(m)), int(m.sum()))} for i, m in enumerate(matrices)],
              "sequence_exact_accuracy": float(np.mean([x["sequence_equal"] for x in metrics])),
              "mean_normalized_edit_distance": float(np.mean([x["normalized_edit_distance"] for x in metrics])),
              "left_count_accuracy": float(np.mean([x["left_count_equal"] for x in metrics])),
              "right_count_accuracy": float(np.mean([x["right_count_equal"] for x in metrics]))}
    return result, matrices


def finalize_turns(rows):
    original, matrices = summarize_turns(rows, "original")
    mirrored, _ = summarize_turns(rows, "swapped")
    frequencies = matrices.sum(axis=(0, 2))
    majority = LABELS[int(np.argmax(frequencies))]
    per_position = [LABELS[int(np.argmax(matrix.sum(axis=1)))] for matrix in matrices]
    for row in rows:
        row["global_majority_labels"] = [majority if value is not None else None for value in row["original_labels"]]
        row["position_majority_labels"] = [per_position[i] if value is not None else None for i, value in enumerate(row["original_labels"])]
    global_baseline, _ = summarize_turns(rows, "global_majority")
    position_baseline, _ = summarize_turns(rows, "position_majority")
    keys = ("aligned_accuracy", "d1_accuracy", "sequence_exact_accuracy", "mean_normalized_edit_distance", "left_count_accuracy", "right_count_accuracy")
    return {"original": original, "swapped": mirrored, "global_majority": global_baseline,
            "position_majority": position_baseline, "global_majority_label": majority,
            "position_majority_labels": per_position, "gt_aligned_class_counts": frequencies.tolist(),
            "swap_minus_original": {key: mirrored[key] - original[key] for key in keys},
            "position_swap_minus_original": [b["accuracy"] - a["accuracy"] for a, b in zip(original["positions"], mirrored["positions"])]}


def main():
    _, navigation, maps = cpu_imports()
    from prior import R2R_DIR
    from prior.directions import start_rotation_to_direction_vector

    assert levenshtein([], []) == 0 and levenshtein(["left"], []) == 1
    assert levenshtein(["left", "right"], ["right", "left"]) == 2
    assert turn_labels([[0, 1], [-1, 0], [1, 0]], [0, 1])[0] == ["straight", "left", "back"]
    examples, results = [], {}
    for split, expected in (("val_unseen", 1839), ("val_seen", 778)):
        with gzip.open(R2R_DIR / split / f"{split}.json.gz", "rt") as stream:
            episodes = json.load(stream)["episodes"]
        assert len(episodes) == expected
        tolerance_rows, relation_rows, turn_rows = [], [], []
        h0_mismatches = []
        with (OUT / f"q2_t5_{split}_episodes.jsonl").open("w") as f5, (OUT / f"q2_t6_{split}_episodes.jsonl").open("w") as f6:
            for index, ep in enumerate(episodes):
                eid = str(ep["episode_id"])
                pred = navigation.llm_cached_cognitive_map_to_tensors(
                    ep["scene_id"], f"R2R_{split}_{eid}", "R2R", split,
                    cache_dir=CACHE, model_key=EFFECTIVE, metadata_schema="direction5",
                )
                gt = maps.cached_cognitive_map_to_tensors(ep["scene_id"], f"R2R_{split}_{eid}", namespace=GT, metadata_schema="direction5")
                pg, gg = pred["grid"].numpy() > 0, gt["grid"].numpy() > 0
                assert pg.shape == gg.shape == (37, 100, 100)
                heading = np.asarray(start_rotation_to_direction_vector(ep["start_rotation"]), dtype=np.float64)
                heading /= np.linalg.norm(heading)
                if not np.allclose(heading, gt["start_direction_vector"].numpy(), atol=1e-5):
                    h0_mismatches.append(eid)
                h_grid = np.array([-heading[1], -heading[0]])
                p0 = []
                for tensor_map in (pred, gt):
                    origin = maps.map_origin_xz(ep["start_position"], tensor_map["start_position"], 1.0)
                    p0.append(maps.world_to_map_cells([ep["start_position"]], origin)[0].numpy().astype(np.float64) * 0.5)
                record5 = {"episode_id": eid, "tolerances": tolerance(pg, gg)}
                tolerance_rows.append(record5)
                f5.write(json.dumps(record5) + "\n")
                record6 = {"episode_id": eid, "heading_display": heading.tolist(), "heading_grid_xz": h_grid.tolist(),
                           "pred_p0_grid_m": p0[0].tolist(), "gt_p0_grid_m": p0[1].tolist(),
                           **relative_relations(pg, gg, p0[0], p0[1], h_grid)}
                relation_rows.append(record6)
                f6.write(json.dumps(record6) + "\n")
                p_vectors, g_vectors = pred["map_trajectory_metadata"].numpy(), gt["map_trajectory_metadata"].numpy()
                gl, ga = turn_labels(g_vectors, heading)
                pl, pa = turn_labels(p_vectors, heading)
                sl, sa = turn_labels(p_vectors[:, ::-1], heading)
                turn_rows.append({"episode_id": eid, "heading_display": heading.tolist(), "gt_vectors": g_vectors.tolist(),
                                  "pred_vectors": p_vectors.tolist(), "gt_labels": gl, "gt_angles_degrees": ga,
                                  "original_labels": pl, "original_angles_degrees": pa, "swapped_labels": sl, "swapped_angles_degrees": sa})
                if split == "val_unseen" and eid in ("274", "477", "1101"):
                    first = np.asarray(g_vectors[0], dtype=np.float64)
                    cross = float(heading[0] * first[1] - heading[1] * first[0])
                    assert "left" in ep["instruction"]["instruction_text"].lower().split(".")[0] and cross > 0
                    examples.append({"episode_id": eid, "split": split, "instruction": ep["instruction"]["instruction_text"],
                                     "h0_display": heading.tolist(), "gt_d1_display": first.tolist(), "display_cross": cross,
                                     "world_xz_cross": -cross, "angle_degrees": ga[0], "discrete_label": gl[0]})
                if (index + 1) % 300 == 0:
                    print(f"Q2 {split}: {index + 1}/{expected}", flush=True)
        t5 = []
        for i, tau in enumerate(TAUS):
            for group in GROUPS:
                rows = [row["tolerances"][i]["groups"][group] for row in tolerance_rows]
                counts = [sum(row[key] for row in rows) for key in ("matched_pred", "pred", "matched_gt", "gt")]
                t5.append({"tau_m": tau, "group": group, "episodes": expected,
                           "pooled_counts": dict(zip(("matched_pred", "pred", "matched_gt", "gt"), counts)),
                           "micro": prf(*counts), "episode_macro": {key: float(np.mean([row[key] for row in rows])) for key in ("precision", "recall", "f1")}})
        t6 = {}
        for group in GROUPS:
            rows = [r["groups"][group] for r in relation_rows]
            keys = ("common_categories", "centered_pred", "centered_gt", "centered_either", "side_eligible", "side_correct",
                    "total_pairs", "tie_pred", "tie_gt", "tie_either", "order_eligible", "order_correct")
            counts = {key: sum(row[key] for row in rows) for key in keys}
            t6[group] = {**counts, "episodes": expected, "mean_common_categories": counts["common_categories"] / expected,
                         "side_micro_accuracy": divide(counts["side_correct"], counts["side_eligible"]),
                         "side_episode_macro_accuracy": float(np.mean([row["side_accuracy"] for row in rows])),
                         "side_supported_episodes": sum(row["side_eligible"] > 0 for row in rows),
                         "order_micro_accuracy": divide(counts["order_correct"], counts["order_eligible"]),
                         "order_episode_macro_accuracy": float(np.mean([row["order_accuracy"] for row in rows])),
                         "order_supported_episodes": sum(row["order_eligible"] > 0 for row in rows)}
        t7 = finalize_turns(turn_rows)
        with (OUT / f"q2_t7_{split}_episodes.jsonl").open("w") as stream:
            for row in turn_rows:
                stream.write(json.dumps(row) + "\n")
        results[split] = {"episodes": expected, "cache_key": EFFECTIVE, "cache_dir": str(CACHE), "gt_namespace": GT,
                          "h0_gt_cache_mismatch_ids": h0_mismatches, "T5": t5, "T6": t6, "T7": t7}
        (OUT / f"q2_{split}_summary.json").write_text(json.dumps(results[split], indent=2) + "\n")
        print(f"Q2 {split} completed", flush=True)
    assert len(examples) == 3
    (OUT / "q2_left_sign_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    for split, summary in results.items():
        write_report(split, summary, examples)
    (OUT / "q2_summary.json").write_text(json.dumps(results, indent=2) + "\n")


def write_report(split, summary, examples):
    lines = [f"# Q2 map quality: R2R {split}", "", f"Episodes: {summary['episodes']}. Cache: {EFFECTIVE}. Root: {CACHE}. GT: {GT}.", "",
             "All values are raw fractions or counts. All expected episodes, including zero-filled maps, remain in full-split episode denominators. Micro denominators are the explicitly reported cells/categories/pairs; no eligible comparisons contribute no micro support and zero to full-split episode-macro accuracy.", "",
             "## T5: tolerance F1", "",
             "| Group | Tolerance (m) | Micro P / R / F1 | Episode-macro P / R / F1 | Matched predicted / predicted cells | Matched GT / GT cells |",
             "|---|---:|---|---|---|---|"]
    for row in summary["T5"]:
        values = [" / ".join(f"{row[method][key]:.10f}" for key in ("precision", "recall", "f1")) for method in ("micro", "episode_macro")]
        c = row["pooled_counts"]
        lines.append(f"| {row['group']} | {row['tau_m']} | {values[0]} | {values[1]} | {c['matched_pred']} / {c['pred']} | {c['matched_gt']} / {c['gt']} |")
    lines += ["", "Definition: 37 native 100×100 channels, >0 binarization, 0.5m cell-centre distances via scipy distance_transform_edt(sampling=0.5), same-category nearest-neighbour matches within <=tau. Precision and recall use separate directional matched counts (not one-to-one matching); F1 is their harmonic mean. Micro pools counts across episodes and channels within each group; episode-macro first pools channels within one episode then averages per-episode P/R/F1 over the full split. Empty denominators give zero.", "",
              "## T6: relative relations", "",
              "| Group | Mean common categories | Side consistent / eligible | Side micro / episode-macro | Centered LLM / GT / either | Order consistent / eligible | Order micro / episode-macro | Distance-tied pairs LLM / GT / either |",
              "|---|---:|---|---|---|---|---|---|"]
    for group, r in summary["T6"].items():
        lines.append(f"| {group} | {r['mean_common_categories']:.10f} | {r['side_correct']} / {r['side_eligible']} | {r['side_micro_accuracy']:.10f} / {r['side_episode_macro_accuracy']:.10f} | {r['centered_pred']} / {r['centered_gt']} / {r['centered_either']} | {r['order_correct']} / {r['order_eligible']} | {r['order_micro_accuracy']:.10f} / {r['order_episode_macro_accuracy']:.10f} | {r['tie_pred']} / {r['tie_gt']} / {r['tie_either']} |")
    lines += ["", "Definition: each common nonempty category centroid is the unweighted mean of occupied 0.5m cell centres. p0 is the dataset start position transformed using the original map_origin_xz/world_to_map_cells with each map's own metre-valued start metadata. h0 comes from dataset start_rotation, converted by original start_rotation_to_direction_vector and normalized; grid heading is (-h_up,-h_right). In row/column=(world x,z), cross<0 is left. A category is excluded from side comparison if either map has |cross|<1m; a within-group unordered pair is excluded from order comparison if either map's distance difference is <1m. All exclusions are counted once in 'either'; LLM/GT counts may overlap. Episode macro includes unsupported episodes as zero.", "",
              "| Group | Side-supported episodes | Order-supported episodes | Total common category pairs |", "|---|---:|---:|---:|"]
    for group, r in summary["T6"].items():
        lines.append(f"| {group} | {r['side_supported_episodes']} | {r['order_supported_episodes']} | {r['total_pairs']} |")
    lines += ["", "## T7: direction convention and left-sign checks", "",
              "GT and LLM cached direction_vectors use display (right,up)=(-dz,-dx). Data h0 is Habitat FRONT=(0,0,-1) rotated by start_rotation, projected to world x/z, then converted to that same display convention. For world h=(h_x,h_z), d=(d_x,d_z), A*h=(-h_z,-h_x) and A*d=(-d_z,-d_x); cross_display(A*h,A*d)=h_z*d_x-h_x*d_z=-cross_world_xz(h,d). Positive display cross is left. Example: world forward (0,-1) and left (-1,0) become display (1,0) and (0,1), with cross +1. T6 transforms both h0 and centroid displacement into grid x/z, so the equivalent left sign there is negative.", "",
              f"Evidence: {ROOT}/prior/directions.py:24–50; {ROOT}/prior/cognitive_map_generation.py:95–153; {ROOT}/vlnce_baselines/models/etp_llm/prompts/llm_grid_system.md:3–10; {ROOT}/vlnce_baselines/models/etp_llm/llm_grid_navigation_cache.py:422–438. LLM cache directions are written without an axis conversion. Data-derived h0, not a zero-filled cache heading, is used in T6/T7.", "",
              f"GT cached h0 versus data-derived h0 mismatches (atol=1e-5): {summary['h0_gt_cache_mismatch_ids']}", "",
              "| val_unseen episode | First sentence | h0 display | GT d1 display | Display cross | World x/z cross | Angle (degrees) |", "|---|---|---|---|---:|---:|---:|"]
    for row in sorted(examples, key=lambda x: int(x["episode_id"])):
        sentence = row["instruction"].split(".")[0] + "."
        lines.append(f"| {row['episode_id']} | {sentence} | {row['h0_display']} | {row['gt_d1_display']} | {row['display_cross']:.10f} | {row['world_xz_cross']:.10f} | {row['angle_degrees']:.10f} |")
    lines += ["", "Definition: these three GT first-motion examples have explicit initial left-turn instructions; the displayed raw positive crosses validate the sign, not a one-to-one alignment between every textual turn and the five retained route vectors.", "",
              "### Turn discretization and metrics", "",
              "Nonzero vectors retain their original slot positions; the previous nonzero vector is the heading reference, with the first nonzero vector referenced to data h0. atan2(cross,dot) is evaluated in display coordinates: straight if |theta|<45 degrees, left/right for 45<=|theta|<=135 with positive/negative sign, back if |theta|>135. No angle epsilon or rounding is applied. Common-slot confusion/accuracy require both vectors nonzero. Sequence metrics drop zero slots while preserving order and include every episode. Empty/empty edit distance is zero. Mirror swaps only the two LLM components, not GT or h0.", "",
              "| Method | Aligned accuracy | d1 accuracy | Exact sequence | Mean normalized edit distance | Left-count accuracy | Right-count accuracy |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    keys = ("aligned_accuracy", "d1_accuracy", "sequence_exact_accuracy", "mean_normalized_edit_distance", "left_count_accuracy", "right_count_accuracy")
    for variant in ("original", "swapped", "global_majority", "position_majority"):
        values = " | ".join(f"{summary['T7'][variant][key]:.10f}" for key in keys)
        lines.append(f"| {variant} | {values} |")
    lines.append("| swapped minus original | " + " | ".join(f"{summary['T7']['swap_minus_original'][key]:+.10f}" for key in keys) + " |")
    lines += ["", f"Definition: normalized edit distance is Levenshtein/max(sequence lengths); left/right count accuracy is exact equality of that turn count per episode. Majority labels are empirical GT modes on the same aligned evaluation population, not a trained model: global={summary['T7']['global_majority_label']}, per-position={summary['T7']['position_majority_labels']}. Baselines keep the original LLM nonzero-slot mask and sequence length. Only numerical mirror differences are reported; no threshold for 'obvious' or significance is imposed.", "",
              "| Position | Common-slot support | Original accuracy | Swapped accuracy | Difference | Global-majority accuracy | Position-majority accuracy |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for i in range(5):
        v = [summary["T7"][name]["positions"][i] for name in ("original", "swapped", "global_majority", "position_majority")]
        lines.append(f"| {i + 1} | {v[0]['support']} | {v[0]['accuracy']:.10f} | {v[1]['accuracy']:.10f} | {v[1]['accuracy'] - v[0]['accuracy']:+.10f} | {v[2]['accuracy']:.10f} | {v[3]['accuracy']:.10f} |")
    lines += ["", "Definition: each position denominator is its explicitly reported count of episodes with both corresponding vectors nonzero, unchanged by swapping. d1 means original slot 1, not the first nonzero slot after padding.", ""]
    for variant in ("original", "swapped"):
        for matrix_key, caption in (("confusion_all", "all aligned slots"), ("confusion_d1", "d1 only")):
            lines += [f"### Confusion: {variant}, {caption}", "", "| GT / predicted | straight | left | right | back |", "|---|---:|---:|---:|---:|"]
            for label, row in zip(LABELS, summary["T7"][variant][matrix_key]):
                lines.append("| " + label + " | " + " | ".join(map(str, row)) + " |")
            lines += ["", "Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.", ""]
    lines += ["## Raw outputs", ""]
    for stage in (5, 6, 7):
        lines.append(str(OUT / f"q2_t{stage}_{split}_episodes.jsonl"))
        lines.append("")
    lines += [str(OUT / f"q2_{split}_summary.json"), "", "CPU only; no bootstrap, significance tests, source/default configuration changes or Git writes.", ""]
    (OUT / f"map_quality_q2_{split}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
