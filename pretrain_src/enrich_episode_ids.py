import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from jsonlines import Reader, Writer

ANNOTATION_DIR = Path("pretrain_src/datasets/R2R/annotations/pretrain_R2R_RxR")

TARGET_FILES = [
    "R2R_train_enc_xlmr.jsonl",
    "R2R_val_unseen_enc_xlmr.jsonl",
    "rxr_train_guide_xlmr.jsonl",
    "rxr_val_unseen_guide_xlmr.jsonl",
]

R2R_SPLIT_FILES = {
    "train": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"),
    "test": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/test/test.json.gz"),
    "val_seen": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_seen/val_seen.json.gz"),
    "val_unseen": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"),
}

RXR_SPLIT_FILES = {
    "train": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"),
    "test": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/test_challenge/test_challenge_guide.json.gz"),
    "val_seen": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/val_seen/val_seen_guide.json.gz"),
    "val_unseen": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz"),
}


def load_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        w = Writer(f)
        w.write_all(rows)


def backup_once(path: Path):
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def build_r2r_index() -> Dict[int, List[Dict]]:
    index = defaultdict(list) # trajectory_id -> [episode_id & split]
    for split, path in R2R_SPLIT_FILES.items():
        data = load_json_gz(path)
        for ep in data["episodes"]:
            traj_id = int(ep["trajectory_id"])
            index[traj_id].append({
                "episode_id": int(ep["episode_id"]),
                "split": split,
            })
    return index


def build_rxr_index() -> Dict[int, Dict]:
    index = {} # instruction_id -> episode_id & split
    for split, path in RXR_SPLIT_FILES.items():
        data = load_json_gz(path)
        for ep in data["episodes"]:
            instr_id = int(ep["instruction"]["instruction_id"])
            assert index.get(instr_id) is None, f"Instruction id already exists: {instr_id}"
            index[instr_id] = {
                "episode_id": int(ep["episode_id"]),
                "split": split,
            }
    return index


def inject_r2r(row, r2r_index: Dict[int, List[Dict]]):
    row["dataset_name"] = "R2R"
    row["episode_id"] = -1
    row["episode_id_source"] = "missing"

    instr_id = str(row["instr_id"])
    parts = instr_id.rsplit("_", 1)
    if len(parts) != 2:
        return row

    traj_id, idx = parts
    traj_id = int(traj_id)
    idx = int(idx)

    candidates = r2r_index.get(traj_id)
    if candidates is None:
        return row

    idx = int(idx)
    if idx >= len(candidates):
        return row

    row["trajectory_id"] = traj_id
    row["episode_id"] = candidates[idx]["episode_id"]
    row["episode_id_source"] = "trajectory_id+index/" + candidates[idx]["split"]
    return row


def inject_rxr(row, rxr_index: Dict[int, Dict]):
    row["dataset_name"] = "RxR"
    row["episode_id"] = -1
    row["episode_id_source"] = "missing"

    instr_id = int(row["instr_id"])

    row["instruction_id"] = instr_id
    data = rxr_index.get(instr_id)
    if data is None:
        return row

    row["episode_id"] = data["episode_id"]
    row["episode_id_source"] = "instruction_id/" + data["split"]
    return row


def enrich_file(path: Path, r2r_index, rxr_index):
    rows = []
    matched = 0
    missing = 0

    with path.open("r", encoding="utf-8") as f:
        reader = Reader(f)
        for row in reader:
            if path.name.lower().startswith("r2r_"):
                row = inject_r2r(row, r2r_index)
            elif path.name.lower().startswith("rxr"):
                row = inject_rxr(row, rxr_index)
            else:
                row["episode_id"] = -1
                row["episode_id_source"] = "missing"

            if row["episode_id"] == -1:
                missing += 1
            else:
                matched += 1
            rows.append(row)

    return rows, matched, missing


def main():
    r2r_index = build_r2r_index()
    rxr_index = build_rxr_index()

    for filename in TARGET_FILES:
        path = ANNOTATION_DIR / filename
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue

        backup_once(path)
        rows, matched, missing = enrich_file(path, r2r_index, rxr_index)
        dump_jsonl(path, rows)
        print(f"[ok] {filename}: matched={matched}, missing={missing}")


if __name__ == "__main__":
    main()
