import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path


ANNOTATION_DIR = Path("pretrain_src/datasets/R2R/annotations/pretrain_R2R_RxR")

TARGET_FILES = [
    "R2R_train_enc_xlmr.jsonl",
    "R2R_val_unseen_enc_xlmr.jsonl",
    "rxr_train_guide_xlmr.jsonl",
    "rxr_val_unseen_guide_xlmr.jsonl",
]

R2R_SPLIT_FILES = {
    "train": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"),
    "val_unseen": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"),
    "val_seen": Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_seen/val_seen.json.gz"),
}

RXR_SPLIT_FILES = {
    "train": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"),
    "val_unseen": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz"),
    "val_seen": Path("data/datasets/RxR_VLNCE_v0_enc_xlmr/val_seen/val_seen_guide.json.gz"),
}


def load_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def backup_once(path: Path):
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def infer_split(filename: str):
    lower = filename.lower()
    if "val_unseen" in lower:
        return "val_unseen"
    if "val_seen" in lower:
        return "val_seen"
    return "train"


def build_r2r_index():
    index = {}
    for split, path in R2R_SPLIT_FILES.items():
        data = load_json_gz(path)
        groups = defaultdict(list)
        for ep in data["episodes"]:
            groups[str(ep["trajectory_id"])].append(int(ep["episode_id"]))
        for traj_id, episode_ids in groups.items():
            index[(split, traj_id)] = sorted(episode_ids)
    return index


def build_rxr_index():
    index = {}
    for split, path in RXR_SPLIT_FILES.items():
        data = load_json_gz(path)
        for ep in data["episodes"]:
            instr_id = str(ep["instruction"]["instruction_id"])
            index[(split, instr_id)] = int(ep["episode_id"])
    return index


def inject_r2r(row, split, r2r_index):
    row["dataset_name"] = "R2R"
    row["episode_id"] = -1
    row["episode_id_source"] = "missing"

    instr_id = str(row["instr_id"])
    parts = instr_id.rsplit("_", 1)
    if len(parts) != 2:
        return row

    traj_id, idx = parts
    if not idx.isdigit():
        return row

    candidates = r2r_index.get((split, traj_id))
    if candidates is None:
        return row

    idx = int(idx)
    if idx >= len(candidates):
        return row

    row["trajectory_id"] = traj_id
    row["episode_id"] = candidates[idx]
    row["episode_id_source"] = "trajectory_id+index"
    return row


def inject_rxr(row, split, rxr_index):
    row["dataset_name"] = "RxR"
    row["episode_id"] = -1
    row["episode_id_source"] = "missing"

    instr_id = str(row["instr_id"])
    if not instr_id.isdigit():
        return row

    row["instruction_id"] = instr_id
    ep_id = rxr_index.get((split, instr_id))
    if ep_id is None:
        return row

    row["episode_id"] = ep_id
    row["episode_id_source"] = "instruction_id"
    return row


def enrich_file(path: Path, r2r_index, rxr_index):
    split = infer_split(path.name)
    rows = []
    matched = 0
    missing = 0

    for row in load_jsonl(path):
        if path.name.lower().startswith("r2r_"):
            row = inject_r2r(row, split, r2r_index)
        elif path.name.lower().startswith("rxr"):
            row = inject_rxr(row, split, rxr_index)
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
