"""B3: FGR2R val_unseen fields and rand100 coverage.

Env: FGR2R_DIR (clone of YicongHong/Fine-Grained-R2R), REPORTS_DIR (has rand100_ids.json).
"""

from __future__ import annotations

import ast
import sys
from collections import Counter

from _common import env_path, load_json


def parsed(value: object) -> list:
    """FGR2R stores new_instructions as a repr string and chunk_view as a list."""
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    if isinstance(value, list):
        return value
    raise TypeError(f"unexpected field type {type(value).__name__}")


def main() -> int:
    fg_dir = env_path("FGR2R_DIR")
    reports = env_path("REPORTS_DIR")
    if (
        fg_dir is None
        or reports is None
        or not (fg_dir / "data" / "FGR2R_val_unseen.json").is_file()
    ):
        print("B3 SKIP: FGR2R_DIR/data/FGR2R_val_unseen.json missing")
        return 2
    cands = sorted(str(p) for p in fg_dir.rglob("*val_unseen*.json"))
    print("FGR2R val_unseen files:", cands)
    fg = load_json(fg_dir / "data" / "FGR2R_val_unseen.json")
    print("n items", len(fg))
    it = fg[0]
    print("keys:", list(it.keys()))
    for k in (
        "path_id",
        "scan",
        "instr_id",
        "instructions",
        "new_instructions",
        "chunk_view",
        "path",
        "heading",
        "distance",
    ):
        print(k, ":", str(it.get(k))[:300])
    print(
        "type(new_instructions):",
        type(it["new_instructions"]).__name__,
        "| type(chunk_view):",
        type(it["chunk_view"]).__name__,
    )
    ni, cv = parsed(it["new_instructions"]), parsed(it["chunk_view"])
    print(
        "new_instructions parsed: n_instr",
        len(ni),
        "sub-instr per instr",
        [len(x) for x in ni],
    )
    print(
        "chunk_view parsed: n_instr",
        len(cv),
        "chunks per instr",
        [len(x) for x in cv],
        "first:",
        cv[0],
    )
    bad = 0
    for x in fg:
        cvx, nix = parsed(x["chunk_view"]), parsed(x["new_instructions"])
        if len(cvx) != len(nix) or any(len(a) != len(b) for a, b in zip(cvx, nix)):
            bad += 1
        if any(b > len(x["path"]) or a < 1 for instr in cvx for (a, b) in instr):
            bad += 1
    print(
        "items where chunk_view/new_instructions shapes disagree or indices out of path range:",
        bad,
    )
    r2r_path = fg_dir / "source" / "R2R-original" / "R2R_val_unseen.json"
    if r2r_path.is_file():
        r2r = load_json(r2r_path)
        print("R2R-original val_unseen n items", len(r2r), "keys", list(r2r[0].keys()))
        print(
            "path_id sets equal (FGR2R vs R2R original):",
            {x["path_id"] for x in fg} == {x["path_id"] for x in r2r},
        )
    rand = load_json(reports / "rand100_ids.json")
    fg_by_path = {str(x["path_id"]): x for x in fg}
    hit = [t for (_, t, _) in rand if str(t) in fg_by_path]
    print(
        "rand100 trajectory_ids found in FGR2R val_unseen:",
        len(hit),
        "/",
        len(rand),
        "(unique:",
        len({t for (_, t, _) in rand}),
        ")",
    )
    print(
        "missing trajectory_ids:",
        [t for (_, t, _) in rand if str(t) not in fg_by_path][:20],
    )
    c = Counter(t for (_, t, _) in rand)
    print(
        "rand100 trajectories with >1 episode:", {k: v for k, v in c.items() if v > 1}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
