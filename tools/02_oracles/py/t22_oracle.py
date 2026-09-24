"""T2.2: O-progress oracle (clause segmentation) per R2R-CE episode.

Env: OUT, CE_ORIG, RAND100, FGR2R, CONN; SPLITS (default "rand100 val_unseen train").
Needs $OUT/ce_instr_index_<split>.json (T2.1). Writes $OUT/oracle_progress_<split>.json and md/T2.2.md.
Exit 0 when rand100 has all 100 episodes, else 1.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter
from typing import Any

import numpy as np

from common import issue, reset_issues
from common import (
    CLAUSE_TYPES,
    ce_episodes,
    classify,
    cum_arclen,
    dump_json,
    env_path,
    hab_xz,
    load_json,
    md_path,
    md_table,
    out_dir,
    parse_listish,
    r2r_split_of,
    viewpoint_xz,
)


def align(vp_xz: np.ndarray, ref_xz: np.ndarray) -> tuple[list[int], str]:
    """Viewpoint i -> ref_path index. Same length: identity (task 1, B4). Otherwise the nearest ref point,
    constrained to be non-decreasing so the order of the path is kept."""
    if len(vp_xz) == len(ref_xz):
        return list(range(len(vp_xz))), "index"
    out, lo = [], 0
    for p in vp_xz:
        d = np.linalg.norm(ref_xz[lo:] - p, axis=1)
        j = lo + int(np.argmin(d))
        out.append(j)
        lo = j
    return out, "nearest"


def build_episode(e: dict, entry: dict, fg: dict) -> tuple[dict | None, str]:
    k = entry["k"]
    if entry["match"] not in ("exact", "fuzzy"):
        return None, f"k unmatched ({entry['match']})"
    ni = parse_listish(fg["new_instructions"])
    cv = parse_listish(fg["chunk_view"])
    if k >= len(ni) or k >= len(cv):
        return None, f"k={k} beyond FGR2R lists ({len(ni)}, {len(cv)})"
    subs, ranges = ni[k], cv[k]
    if len(subs) != len(ranges):
        return None, f"clauses {len(subs)} != chunk_view {len(ranges)}"
    ref = hab_xz(e["reference_path"])
    vpos = viewpoint_xz(fg["scan"])
    vp = np.array([vpos[v] for v in fg["path"]], dtype=float)
    vmap, kind = align(vp, ref)
    offsets = np.linalg.norm(vp - ref[vmap], axis=1)
    cum = cum_arclen(ref)
    clauses: list[dict[str, Any]] = []
    for i, (tokens, (a1, b1)) in enumerate(zip(subs, ranges)):
        if not (1 <= a1 <= b1 <= len(vp)):
            return None, f"chunk_view range {[a1, b1]} outside path of {len(vp)}"
        a, b = vmap[a1 - 1], vmap[b1 - 1]
        text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        clauses.append({
            "idx": i,
            "text": text,
            "vp_range_1b": [a1, b1],
            "ref_range_0b": [a, b],
            "length_m": round(float(cum[b] - cum[a]), 4),
            "type": classify(text),
        })
    starts = [int(vmap[a1 - 1]) for _, (a1, _b1) in zip(subs, ranges)]
    ends = [int(vmap[b1 - 1]) for _, (_a1, b1) in zip(subs, ranges)]
    contiguous = (
        starts[0] == 0
        and ends[-1] == len(ref) - 1
        and all(s in (pe, pe + 1) for s, pe in zip(starts[1:], ends[:-1]))
    )
    return {
        "episode_id": int(e["episode_id"]),
        "path_id": int(e["trajectory_id"]),
        "k": k,
        "scene": fg["scan"],
        "clauses": clauses,
        "ref_path_xz": [[round(float(x), 4), round(float(z), 4)] for x, z in ref],
        "cum_arclen_m": [round(float(c), 4) for c in cum],
        "max_single_point_offset_m": round(float(offsets.max()), 4),
        "alignment": kind,
        "contiguous": bool(contiguous),
        "geodesic_distance": float(
            (e.get("info") or {}).get("geodesic_distance", float("nan"))
        ),
    }, ""


def build_split(split: str) -> tuple[dict, dict]:
    index = load_json(out_dir() / f"ce_instr_index_{split}.json")
    fg = {
        int(x["path_id"]): x
        for x in load_json(env_path("FGR2R") / f"FGR2R_{r2r_split_of(split)}.json")
    }
    eps, skipped = [], {}
    for e in ce_episodes(split):
        eid = str(e["episode_id"])
        entry = index.get(eid)
        item = fg.get(int(e["trajectory_id"]))
        if entry is None or item is None:
            skipped[eid] = "no T2.1 entry" if entry is None else "path_id not in FGR2R"
            continue
        ep, why = build_episode(e, entry, item)
        if ep is None:
            skipped[eid] = why
        else:
            eps.append(ep)
    meta = {
        "split": split,
        "convention": "habitat x-z plane, horizontal metres; mp3d(x,y,z)=(hab_x,-hab_z,hab_y); vp_range_1b from FGR2R chunk_view, ref_range_0b into reference_path",
        "n_episodes": len(eps),
        "skipped": skipped,
    }
    return {**meta, "episodes": eps}, meta


def summarize(split: str, data: dict) -> list[str]:
    eps = data["episodes"]
    n_cl = [len(e["clauses"]) for e in eps]
    types = Counter(c["type"] for e in eps for c in e["clauses"])
    zero = sum(1 for e in eps for c in e["clauses"] if c["length_m"] == 0)
    offs = [e["max_single_point_offset_m"] for e in eps]
    align_c = Counter(e["alignment"] for e in eps)
    lines = [f"### {split}", ""]
    lines.append(
        md_table(
            [
                "n",
                "skipped",
                "clauses min / median / max",
                "zero-length clauses",
                "non-contiguous",
                "alignment",
                "max point offset P50 / P95 / max (m)",
            ],
            [
                [
                    len(eps),
                    len(data["skipped"]),
                    f"{min(n_cl)} / {int(np.median(n_cl))} / {max(n_cl)}"
                    if n_cl
                    else "-",
                    zero,
                    sum(not e["contiguous"] for e in eps),
                    dict(align_c),
                    (
                        f"{np.percentile(offs, 50):.3f} / {np.percentile(offs, 95):.3f} / {max(offs):.3f}"
                        if offs
                        else "-"
                    ),
                ]
            ],
        )
    )
    total = sum(types.values()) or 1
    lines += [
        "",
        "clause count per episode: " + str(dict(sorted(Counter(n_cl).items()))),
        "",
    ]
    lines.append(
        md_table(
            ["type", *CLAUSE_TYPES],
            [
                ["n", *[types[t] for t in CLAUSE_TYPES]],
                ["share", *[f"{100 * types[t] / total:.1f}%" for t in CLAUSE_TYPES]],
            ],
        )
    )
    if data["skipped"]:
        lines += ["", f"skipped (first 10): {dict(list(data['skipped'].items())[:10])}"]
    lines.append("")
    return lines


def main() -> int:
    splits = os.environ.get("SPLITS", "rand100 val_unseen train").split()
    lines = ["## T2.2 oracle", ""]
    ok = True
    md_path("T2.2")
    reset_issues("T2.2")
    for split in splits:
        data, _ = build_split(split)
        n_near = sum(e["alignment"] == "nearest" for e in data["episodes"])
        if data["skipped"] or n_near:
            issue(
                f"T2.2: {split}: {len(data['skipped'])} episodes skipped, {n_near} aligned by nearest point (path and reference_path lengths differ)"
            )
        dump_json(out_dir() / f"oracle_progress_{split}.json", data)
        lines += summarize(split, data)
        if split == "rand100":
            ok &= data["n_episodes"] == 100
            rng = random.Random(0)
            sample = rng.sample(data["episodes"], min(10, len(data["episodes"])))
            lines += ["#### rand100 抽查 10 条（seed 0）", "", "```"]
            for e in sample:
                lines.append(
                    f"episode {e['episode_id']}  path {e['path_id']}  k={e['k']}  ref points {len(e['ref_path_xz'])}  arclen {e['cum_arclen_m'][-1]:.2f} m"
                )
                for c in e["clauses"]:
                    lines.append(
                        f"  [{c['idx']}] {c['type']:8s} vp{c['vp_range_1b']} ref{c['ref_range_0b']} {c['length_m']:6.2f} m  {c['text']}"
                    )
            lines += ["```", ""]
    lines.append(f"判据（rand100 100 条全部生成）：{'满足' if ok else '不满足'}")
    md_path("T2.2").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
