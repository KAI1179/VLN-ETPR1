"""T2.1: recover the R2R instruction index k of every R2R-CE episode.

Env: OUT, CE_ORIG, RAND100, R2R_DISC; SPLITS (default "val_unseen train rand100").
Writes $OUT/ce_instr_index_<split>.json = {episode_id: {path_id, k, match, ratio}} and md/T2.1.md.
Exit 0 when every split is 100 % covered by exact + fuzzy (>= 0.9) matches, else 1.
"""

from __future__ import annotations

import difflib
import os
import sys
from collections import Counter

from common import issue, reset_issues
from common import (
    ce_episodes,
    dump_json,
    env_path,
    load_json,
    md_path,
    md_table,
    normalize,
    out_dir,
    r2r_split_of,
)

FUZZY_MIN = 0.9


def index_split(split: str) -> tuple[dict[str, dict], dict]:
    r2r = {
        int(x["path_id"]): x
        for x in load_json(env_path("R2R_DISC") / f"R2R_{r2r_split_of(split)}.json")
    }
    index: dict[str, dict] = {}
    notes: dict[str, list] = {
        "fuzzy": [],
        "unmatched": [],
        "no_path": [],
        "ambiguous": [],
    }
    for e in ce_episodes(split):
        eid, pid = str(e["episode_id"]), int(e["trajectory_id"])
        item = r2r.get(pid)
        if item is None:
            index[eid] = {"path_id": pid, "k": -1, "match": "no_path", "ratio": 0.0}
            notes["no_path"].append(eid)
            continue
        text = normalize(e["instruction"]["instruction_text"])
        cands = [normalize(t) for t in item["instructions"]]
        exact = [k for k, c in enumerate(cands) if c == text]
        if exact:
            index[eid] = {"path_id": pid, "k": exact[0], "match": "exact", "ratio": 1.0}
            if len(exact) > 1:
                notes["ambiguous"].append((eid, exact))
            continue
        ratios = [difflib.SequenceMatcher(None, text, c).ratio() for c in cands]
        k = max(range(len(ratios)), key=ratios.__getitem__)
        kind = "fuzzy" if ratios[k] >= FUZZY_MIN else "unmatched"
        index[eid] = {
            "path_id": pid,
            "k": k,
            "match": kind,
            "ratio": round(ratios[k], 4),
        }
        notes[kind].append((eid, round(ratios[k], 4), text[:90], cands[k][:90]))
    return index, notes


def main() -> int:
    splits = os.environ.get("SPLITS", "val_unseen train rand100").split()
    md_path("T2.1")
    reset_issues("T2.1")
    lines = ["## T2.1 k 匹配", ""]
    rows, ok = [], True
    details: list[str] = []
    for split in splits:
        index, notes = index_split(split)
        dump_json(out_dir() / f"ce_instr_index_{split}.json", index)
        c = Counter(v["match"] for v in index.values())
        n = len(index)
        covered = c["exact"] + c["fuzzy"]
        ok &= covered == n
        k_dist = Counter(v["k"] for v in index.values() if v["k"] >= 0)
        rows.append([
            split,
            n,
            c["exact"],
            c["fuzzy"],
            f"{100 * c['fuzzy'] / n:.2f}%",
            c["unmatched"],
            c["no_path"],
            len(notes["ambiguous"]),
            f"{100 * covered / n:.2f}%",
            dict(sorted(k_dist.items())),
        ])
        if notes["fuzzy"]:
            details.append(
                f"{split} fuzzy examples (episode_id, ratio, CE text, R2R text):"
            )
            details += [f"  {x}" for x in notes["fuzzy"][:5]]
        if notes["unmatched"] or notes["no_path"]:
            issue(
                f"T2.1: {split} has {len(notes['unmatched'])} unmatched and {len(notes['no_path'])} no_path episodes; they are left out of the oracle"
            )
            details.append(
                f"{split} UNMATCHED: {notes['unmatched'][:20]} NO_PATH: {notes['no_path'][:20]}"
            )
        if notes["ambiguous"]:
            details.append(
                f"{split} ambiguous exact matches (same text for several k; first k kept): {notes['ambiguous'][:10]}"
            )
    lines.append(
        md_table(
            [
                "split",
                "n",
                "exact",
                "fuzzy",
                "fuzzy %",
                "unmatched",
                "no_path",
                "ambiguous",
                "coverage",
                "k distribution",
            ],
            rows,
        )
    )
    lines += [
        "",
        f"fuzzy = best `difflib.SequenceMatcher` ratio >= {FUZZY_MIN} after normalisation (lower case, collapsed whitespace, trailing '.' removed).",
        "",
    ]
    if details:
        lines += ["```", *details, "```"]
    lines += ["", f"判据（exact + fuzzy 覆盖率 = 100%）：{'满足' if ok else '不满足'}"]
    md_path("T2.1").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
