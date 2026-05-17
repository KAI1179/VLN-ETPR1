#!/usr/bin/env python3
"""Sanity checks for the ETP-R1 dataset. See `doc/ANALYZE.md`, `Data Sanity Check` section for details."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from math import isfinite, dist
from pathlib import Path
from statistics import mean
from typing import Iterable

from . import (
    ANNOTATION_FILES,
    AnnotationEntry,
    CONNECTIVITY_DIR,
    ConnectivityEntry,
)


R2R_TRAIN_FILE = Path("data/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
R2R_ENVDROP_FILE = Path("data/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz")
POSITION_MATCH_TOLERANCE = 0.25


@dataclass
class CheckStats:
    scene_count: int = 0
    connectivity_entry_count: int = 0
    annotation_entry_count: int = 0
    decoded_instruction_count: int = 0
    ascii_instruction_count: int = 0
    edge_count: int = 0
    task_type_counts: Counter[int] = field(default_factory=Counter)
    annotation_file_counts: dict[str, int] = field(default_factory=dict)


def _format_count_map(counter: Counter[int]) -> str:
    return ", ".join(f"{key}: {counter[key]}" for key in sorted(counter))


def _raise_if_errors(errors: list[str]) -> None:
    if not errors:
        return

    preview = "\n".join(f"- {error}" for error in errors[:20])
    suffix = ""
    if len(errors) > 20:
        suffix = f"\n... and {len(errors) - 20} more"
    raise AssertionError(
        f"ETP-R1 sanity check failed with {len(errors)} issue(s):\n{preview}{suffix}"
    )


def _check_connectivity_entry(
    scan: str,
    image_id: str,
    entry: ConnectivityEntry,
    scene_items: list[tuple[str, ConnectivityEntry]],
    scene_index: dict[str, int],
    errors: list[str],
) -> int:
    edge_count = 0
    point_count = len(scene_items)
    index = scene_index[image_id]

    if len(entry.pose) != 16:
        errors.append(
            f"{scan}/{image_id}: expected pose length 16, got {len(entry.pose)}"
        )
    if len(entry.visible) != point_count:
        errors.append(
            f"{scan}/{image_id}: visible length {len(entry.visible)} != scene size {point_count}"
        )
    if len(entry.unobstructed) != point_count:
        errors.append(
            f"{scan}/{image_id}: unobstructed length {len(entry.unobstructed)} != scene size {point_count}"
        )
    if not isfinite(entry.height):
        errors.append(f"{scan}/{image_id}: non-finite height {entry.height}")

    for coord in entry.position:
        if not isfinite(coord):
            errors.append(f"{scan}/{image_id}: non-finite position {entry.position}")
            break

    if len(entry.unobstructed) == point_count:
        if entry.included:
            edge_count = sum(bool(flag) for flag in entry.unobstructed)

    if len(entry.unobstructed) == point_count:
        for other_id, other_entry in scene_items:
            other_index = scene_index[other_id]
            if entry.unobstructed[other_index] != other_entry.unobstructed[index]:
                errors.append(
                    f"{scan}: asymmetric unobstructed edge between {image_id} and {other_id}"
                )
                break

    return edge_count


def check_connectivity(stats: CheckStats, errors: list[str]) -> None:
    with (CONNECTIVITY_DIR / "scans.txt").open() as f:
        scans = [line.strip() for line in f if line.strip()]

    stats.scene_count = len(scans)

    for scan in scans:
        connectivity_map = ConnectivityEntry.map_for(scan)
        if not connectivity_map:
            errors.append(f"{scan}: connectivity graph is empty")
            continue

        stats.connectivity_entry_count += len(connectivity_map)
        scene_items = list(connectivity_map.items())
        scene_index = {image_id: idx for idx, (image_id, _) in enumerate(scene_items)}

        if len(scene_items) != len(scene_index):
            errors.append(f"{scan}: duplicate image ids in connectivity graph")
            continue

        included_count = sum(1 for entry in connectivity_map.values() if entry.included)
        if included_count == 0:
            errors.append(f"{scan}: no included viewpoints found")

        for image_id, entry in scene_items:
            stats.edge_count += _check_connectivity_entry(
                scan, image_id, entry, scene_items, scene_index, errors
            )


def _iter_path_pairs(path: list[str]) -> Iterable[tuple[str, str]]:
    return zip(path, path[1:])


def _normalize_instruction(text: str) -> str:
    return " ".join(text.split())


def _load_r2r_train_json() -> list[dict]:
    with gzip.open(R2R_TRAIN_FILE) as f:
        return json.load(f)["episodes"]


def _load_r2r_envdrop_json() -> list[dict]:
    with gzip.open(R2R_ENVDROP_FILE) as f:
        return json.load(f)["episodes"]


def _load_r2r_trajectory_groups() -> dict[int, list[dict]]:
    trajectory_groups: dict[int, list[dict]] = {}
    for episode in _load_r2r_train_json():
        trajectory_groups.setdefault(episode["trajectory_id"], []).append(episode)

    for group in trajectory_groups.values():
        group.sort(key=lambda episode: episode["episode_id"])

    return trajectory_groups


def _iter_r2r_trajectory_matches() -> Iterable[tuple[AnnotationEntry, dict]]:
    trajectory_groups = _load_r2r_trajectory_groups()

    for entry in AnnotationEntry.iter_from("R2R_train_enc_xlmr.jsonl"):
        trajectory_id_str, instruction_idx_str = entry.instr_id.split("_")
        trajectory_id = int(trajectory_id_str)
        instruction_idx = int(instruction_idx_str)

        assert (
            trajectory_id in trajectory_groups
        ), f"Trajectory {trajectory_id} not found"
        group = trajectory_groups[trajectory_id]

        assert instruction_idx < len(
            group
        ), f"Instruction index {instruction_idx} for trajectory {trajectory_id} is out of bound"
        yield entry, group[instruction_idx]


def _check_r2r_joint_trajectory_alignment() -> None:
    matched_count = 0
    for entry, episode in _iter_r2r_trajectory_matches():
        matched_count += 1
        assert episode["scene_id"].split("/")[1] == entry.scan, (
            f"{entry.instr_id}: trajectory scene mismatch "
            f"{episode['scene_id']} vs {entry.scan}"
        )
        assert _normalize_instruction(
            episode["instruction"]["instruction_text"]
        ) == _normalize_instruction(
            entry.instruction
        ), f"{entry.instr_id}: instruction mismatch"
        assert len(episode["reference_path"]) == len(
            entry.path
        ), f"{entry.instr_id}: reference_path length mismatch"

    assert matched_count > 0, "No R2R trajectory matches found"


def _check_r2r_envdrop_explanation() -> None:
    train_trajectory_ids = {
        episode["trajectory_id"] for episode in _load_r2r_train_json()
    }
    envdrop_trajectory_ids = {
        episode["trajectory_id"] for episode in _load_r2r_envdrop_json()
    }

    missing_counts: Counter[int] = Counter()
    for entry in AnnotationEntry.iter_from("R2R_train_enc_xlmr.jsonl"):
        trajectory_id = int(entry.instr_id.split("_")[0])
        if trajectory_id not in train_trajectory_ids:
            missing_counts[trajectory_id] += 1

    explained_trajectory_ids = {
        trajectory_id
        for trajectory_id in missing_counts
        if trajectory_id in envdrop_trajectory_ids
    }
    unexplained_trajectory_ids = sorted(set(missing_counts) - explained_trajectory_ids)
    explained_entries = sum(
        missing_counts[trajectory_id] for trajectory_id in explained_trajectory_ids
    )
    unexplained_entries = sum(
        missing_counts[trajectory_id] for trajectory_id in unexplained_trajectory_ids
    )

    assert explained_entries == 3217, (
        "Unexpected envdrop explanation count for train-unmatched ETP entries: "
        f"{explained_entries}"
    )
    assert unexplained_entries == 3, (
        "Unexpected non-envdrop remainder for train-unmatched ETP entries: "
        f"{unexplained_entries} across trajectory_ids {unexplained_trajectory_ids}"
    )
    assert unexplained_trajectory_ids == [1162], (
        "Unexpected unexplained trajectory_ids after envdrop comparison: "
        f"{unexplained_trajectory_ids}"
    )


def _check_r2r_position_convention() -> None:
    candidates = {
        "raw_3_7_11": lambda entry: (entry.pose[3], entry.pose[7], entry.pose[11]),
        "3_11_minus_h_neg7": lambda entry: (
            entry.pose[3],
            entry.pose[11] - entry.height,
            -entry.pose[7],
        ),
        "3_11_neg7": lambda entry: (entry.pose[3], entry.pose[11], -entry.pose[7]),
    }
    candidate_errors = {name: [] for name in candidates}

    matched_count = 0
    for annotation, episode in _iter_r2r_trajectory_matches():
        matched_count += 1
        connectivity_map = ConnectivityEntry.map_for(annotation.scan)
        for viewpoint_id, reference_position in zip(
            annotation.path, episode["reference_path"]
        ):
            connectivity_entry = connectivity_map[viewpoint_id]
            for name, transform in candidates.items():
                position = transform(connectivity_entry)
                candidate_errors[name].append(dist(position, reference_position))

    assert matched_count > 0, "No R2R trajectory matches found for convention check"

    best_candidate = min(
        candidate_errors,
        key=lambda name: mean(candidate_errors[name]),
    )
    assert best_candidate == "3_11_minus_h_neg7", (
        "Unexpected best R2R transform: "
        f"{best_candidate} (means: "
        f"{', '.join(f'{name}={mean(values):.4f}' for name, values in candidate_errors.items())})"
    )
    assert mean(candidate_errors["3_11_minus_h_neg7"]) < POSITION_MATCH_TOLERANCE, (
        "Best candidate transform is still too far from R2R coordinates: "
        f"{mean(candidate_errors['3_11_minus_h_neg7']):.4f}"
    )
    assert mean(candidate_errors["raw_3_7_11"]) > 1.0, (
        "Raw ConnectivityEntry.position unexpectedly matches R2R convention: "
        f"{mean(candidate_errors['raw_3_7_11']):.4f}"
    )


def check_annotations(
    stats: CheckStats,
    errors: list[str],
) -> None:
    seen_instr_ids: set[str] = set()
    for filename in ANNOTATION_FILES:
        entries = AnnotationEntry.iter_from(filename)
        file_counts = 0

        for idx, entry in enumerate(entries):
            file_counts += 1

            stats.annotation_entry_count += 1

            _check_annotation_entry(
                filename,
                idx,
                entry,
                seen_instr_ids,
                stats,
                errors,
            )

        stats.annotation_file_counts[filename] = file_counts


def _check_annotation_entry(
    filename: str,
    idx: int,
    entry: AnnotationEntry,
    seen_instr_ids: set[str],
    stats: CheckStats,
    errors: list[str],
) -> None:
    label = f"{filename}[{idx}]"

    if entry.instr_id in seen_instr_ids:
        errors.append(f"{label}: duplicate instr_id {entry.instr_id}")
    else:
        seen_instr_ids.add(entry.instr_id)

    if not entry.instr_id:
        errors.append(f"{label}: empty instr_id")
    if not entry.path:
        errors.append(f"{label}: empty path")
        return
    if not isfinite(entry.heading):
        errors.append(f"{label}: non-finite heading {entry.heading}")
    if not entry.instr_encoding:
        errors.append(f"{label}: empty instr_encoding")
    if not isinstance(entry.task_type_encoding, int):
        errors.append(f"{label}: task_type_encoding must be int")
    else:
        stats.task_type_counts[entry.task_type_encoding] += 1

    connectivity_map = ConnectivityEntry.map_for(entry.scan)
    if not connectivity_map:
        errors.append(f"{label}: unknown scan {entry.scan}")
        return

    scene_index = {image_id: idx for idx, image_id in enumerate(connectivity_map)}
    for viewpoint in entry.path:
        if viewpoint not in connectivity_map:
            errors.append(
                f"{label}: unknown viewpoint {viewpoint} in scan {entry.scan}"
            )
            return
        if not connectivity_map[viewpoint].included:
            errors.append(f"{label}: path references excluded viewpoint {viewpoint}")

    for src, dst in _iter_path_pairs(entry.path):
        src_entry = connectivity_map[src]
        dst_index = scene_index[dst]
        if (
            dst_index >= len(src_entry.unobstructed)
            or not src_entry.unobstructed[dst_index]
        ):
            errors.append(f"{label}: path step {src} -> {dst} is not unobstructed")
            break

    instruction = entry.instruction
    stats.decoded_instruction_count += 1
    if instruction.isascii():
        stats.ascii_instruction_count += 1
    if not instruction.strip():
        errors.append(f"{label}: decoded instruction is empty")


def print_summary(stats: CheckStats) -> None:
    print("=" * 60)
    print("ETP-R1 SANITY CHECK")
    print("=" * 60)
    print(f"Scenes checked:                {stats.scene_count}")
    print(f"Connectivity entries checked:  {stats.connectivity_entry_count}")
    print(f"Approx. directed edges checked:{stats.edge_count}")
    print(f"Annotation entries checked:    {stats.annotation_entry_count}")
    print(
        f"Instructions decoded:          {stats.decoded_instruction_count} ({stats.ascii_instruction_count} ASCII)"
    )
    print(f"Task types:                    {_format_count_map(stats.task_type_counts)}")
    for filename in ANNOTATION_FILES:
        count = stats.annotation_file_counts.get(filename, 0)
        print(f"{filename:40} {count}")
    print("=" * 60)


def main() -> None:
    stats = CheckStats()
    errors: list[str] = []

    check_connectivity(stats, errors)
    check_annotations(
        stats,
        errors,
    )
    for check in (
        _check_r2r_joint_trajectory_alignment,
        _check_r2r_envdrop_explanation,
        _check_r2r_position_convention,
    ):
        try:
            check()
        except AssertionError as exc:
            errors.append(str(exc))
    print_summary(stats)

    if errors:
        _raise_if_errors(errors)

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
