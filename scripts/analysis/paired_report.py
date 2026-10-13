#!/usr/bin/env python3
"""Paired-episode report from decision dumps (scripts/analysis/dump_eval.sh).

Inputs are the eval_results directories holding decisions_<split>_r*.jsonl.
The first --run is the reference (baseline); every other run is paired
against it episode by episode.  Optional --forced runs (same labels) hold the
teacher-forced dumps (protocol 2), where all models saw identical states.

    python scripts/analysis/paired_report.py \
        --features data/logs/analysis/features_val_unseen.json \
        --run baseline=data/logs/checkpoints/base_dump_free_iter25000_val_unseen/eval_results \
              try5=.../try5_dump_free_iter28000_val_unseen/eval_results \
              kd=.../kd_dual_dump_free_iter20000_val_unseen/eval_results \
        --forced baseline=.../base_dump_forced_iter25000_val_unseen/eval_results \
                 try5=... kd=... \
        --out data/logs/analysis/report_val_unseen.md
(all LABEL=DIR pairs of one option follow that option, space separated)

Sections (all numbers are plain counts and means; no intervals):
  1  per-run SR / SPL / OSR / nDTW / stop error, and the paired 2x2 success
     table (both / ref-only / other-only / neither) per pair
  2  hard-subset table: SR and SPL of every run on the GT-derived strata
     (multi-fork, multi-room, decoy, long path, spatial-language, LLM-map
     good/bad) -- the main-text table
  3  discordant-pair profile: mean feature values of the episodes only the
     reference solves vs only the other run solves
  4  failure modes per run (stop error / wrong branch at a fork / off path
     elsewhere / other), from the free-rollout dump
  5  efficiency on episodes every run solves: steps, path length, revisits
  6  step-level decision accuracy vs the DAgger expert, overall and on fork
     steps by number of options (free dump, and forced dump when given); for
     forced dumps also the per-state 2x2 (ref right/wrong x other right/wrong)
  7  first divergence between the reference and each run (free dumps): at
     which step the executed actions first differ, whether that state is a
     fork, and who matched the expert there

A "fork" is a state with >= 2 unvisited ghost nodes within --local-radius of
the agent in directions more than --fork-angle apart.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tap import Tap


class Args(Tap):
    run: List[str]
    """LABEL=DIR, free-rollout dumps; the first is the reference."""
    forced: List[str] = []
    """LABEL=DIR, teacher-forced dumps (optional)."""
    features: Optional[str] = None
    """episode_features.py output; without it only sections 1, 4-7 are written."""
    split: str = "val_unseen"
    out: Optional[str] = None
    local_radius: float = 3.0
    fork_angle: float = 45.0
    min_forks: int = 2
    min_room_changes: int = 2
    min_decoys: int = 1
    long_quantile: float = 0.67
    """Long-path stratum = path length at or above this quantile."""
    spatial_quantile: float = 0.5
    """Spatial-language stratum = n_spatial_words at or above this quantile."""


# ----------------------------------------------------------------- loading


def load_dump(directory: str, split: str) -> Dict[str, dict]:
    files = sorted(glob.glob(os.path.join(directory, f"decisions_{split}_r*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no decisions_{split}_r*.jsonl in {directory}")
    eps: Dict[str, dict] = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["kind"] == "step":
                    eps.setdefault(str(r["ep"]), {"steps": []})["steps"].append(r)
                elif r["kind"] == "episode":
                    e = eps.setdefault(str(r["ep"]), {"steps": []})
                    e["metric"] = r["metric"]
                    e["traj"] = r.get("traj", [])
                    e["scene"] = r.get("scene")
    for ep, e in eps.items():
        e["steps"].sort(key=lambda r: r["step"])
        if "metric" not in e:  # unfinished episode (killed job)
            e["metric"] = None
    return eps


def parse_runs(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected LABEL=DIR, got {it!r}")
        label, d = it.split("=", 1)
        out.append((label, d))
    return out


# ------------------------------------------------------------ step features


def local_options(step: dict, radius: float, angle_deg: float) -> Tuple[int, List[int]]:
    """Number of distinct directions among unvisited ghosts near the agent, and
    the indices (into step['vps']) of those local candidates."""
    cur = np.asarray(step["cur_pos"], dtype=float)
    cands = []
    for i, (vp, pos, vis) in enumerate(zip(step["vps"], step["pos"], step["visited"])):
        if vp is None or vis or pos is None or not str(vp).startswith("g"):
            continue
        d = np.asarray(pos, dtype=float) - cur
        dist = math.hypot(d[0], d[2])
        if dist <= radius:
            cands.append((i, math.atan2(d[2], d[0])))
    clusters: List[float] = []
    thr = math.radians(angle_deg)
    for _, bearing in cands:
        if all(abs((bearing - c + math.pi) % (2 * math.pi) - math.pi) > thr for c in clusters):
            clusters.append(bearing)
    return len(clusters), [i for i, _ in cands]


def annotate(eps: Dict[str, dict], radius: float, angle: float) -> None:
    for e in eps.values():
        seen = set()
        e["n_revisits"] = 0
        e["first_dev"] = None
        for s in e["steps"]:
            s["n_options"], s["local"] = local_options(s, radius, angle)
            s["is_fork"] = s["n_options"] >= 2
            s["valid"] = s["teacher"] != -100
            s["is_stop"] = s["teacher"] == 0
            s["correct"] = bool(s["valid"] and s["student"] == s["teacher"])
            if s["cur_vp"] in seen:
                e["n_revisits"] += 1
            seen.add(s["cur_vp"])
            if e["first_dev"] is None and s["valid"] and not s["is_stop"] and not s["correct"]:
                e["first_dev"] = s
        m = e["metric"] or {}
        if m.get("success", 0) >= 1:
            mode = "success"
        elif m.get("oracle_success", 0) >= 1:
            mode = "stop_error"
        elif e["first_dev"] is not None and e["first_dev"]["is_fork"]:
            mode = "wrong_branch"
        elif e["first_dev"] is not None:
            mode = "off_path"
        else:
            mode = "other"
        e["mode"] = mode


# ----------------------------------------------------------------- tables


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def mean_metric(eps: Dict[str, dict], ids: Sequence[str], key: str) -> float:
    vals = [eps[i]["metric"][key] for i in ids if eps[i]["metric"] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def success_set(eps: Dict[str, dict], ids: Sequence[str]) -> set:
    return {i for i in ids if eps[i]["metric"] and eps[i]["metric"]["success"] >= 1}


def strata(features: Dict[str, dict], ids: Sequence[str], args: Args) -> Dict[str, List[str]]:
    fs = {i: features[i] for i in ids if i in features}
    def have(k):
        return [v[k] for v in fs.values() if v.get(k) is not None and v.get(k) != -1]
    out: Dict[str, List[str]] = {"all": list(ids)}
    out[f"multi-fork (forks>={args.min_forks})"] = [i for i, v in fs.items() if (v.get("n_forks_graph") or 0) >= args.min_forks]
    out["single-fork (forks<=1)"] = [i for i, v in fs.items() if v.get("n_forks_graph") is not None and 0 <= v["n_forks_graph"] <= 1]
    out[f"multi-room (changes>={args.min_room_changes})"] = [i for i, v in fs.items() if (v.get("n_room_changes") or 0) >= args.min_room_changes]
    out[f"decoy (decoys>={args.min_decoys})"] = [i for i, v in fs.items() if (v.get("n_decoys") or 0) >= args.min_decoys]
    if have("path_len_m"):
        thr = float(np.quantile(have("path_len_m"), args.long_quantile))
        out[f"long path (>= {thr:.1f} m)"] = [i for i, v in fs.items() if v.get("path_len_m", 0) >= thr]
        out[f"short path (< {thr:.1f} m)"] = [i for i, v in fs.items() if v.get("path_len_m", 0) < thr]
    if have("n_spatial_words"):
        thr = float(np.quantile(have("n_spatial_words"), args.spatial_quantile))
        out[f"spatial language (words>={thr:.0f})"] = [i for i, v in fs.items() if v.get("n_spatial_words", 0) >= thr]
    if have("llm_region_hit"):
        thr = float(np.median(have("llm_region_hit")))
        out[f"LLM map good (hit>={thr:.2f})"] = [i for i, v in fs.items() if v.get("llm_region_hit") is not None and v["llm_region_hit"] >= thr]
        out[f"LLM map poor (hit<{thr:.2f})"] = [i for i, v in fs.items() if v.get("llm_region_hit") is not None and v["llm_region_hit"] < thr]
    return out


FEATURE_KEYS = ["path_len_m", "n_forks_graph", "n_forks3_graph", "n_room_changes",
                "n_landmarks", "n_decoys", "n_decoy_categories", "n_spatial_words", "llm_region_hit"]


def feature_profile(features: Dict[str, dict], ids: Sequence[str]) -> List[str]:
    cells = []
    for k in FEATURE_KEYS:
        xs = [features[i][k] for i in ids if i in features and features[i].get(k) is not None and features[i].get(k) != -1]
        cells.append(f"{np.mean(xs):.2f}" if xs else "-")
    return cells


# ------------------------------------------------------------- sections


def section_overview(runs, eps_by, ids) -> str:
    rows = []
    for label, _ in runs:
        e = eps_by[label]
        rows.append([label, len(ids), pct(mean_metric(e, ids, "success")), pct(mean_metric(e, ids, "spl")),
                     pct(mean_metric(e, ids, "oracle_success")), pct(mean_metric(e, ids, "ndtw")),
                     pct(mean_metric(e, ids, "oracle_success") - mean_metric(e, ids, "success")),
                     f"{mean_metric(e, ids, 'steps_taken'):.1f}", f"{mean_metric(e, ids, 'path_length'):.2f}"])
    out = ["## 1. Overview (common episodes)", "",
           md_table(["run", "n", "SR", "SPL", "OSR", "nDTW", "OSR-SR", "steps", "PL (m)"], rows), ""]
    ref, _ = runs[0]
    sref = success_set(eps_by[ref], ids)
    rows = []
    for label, _ in runs[1:]:
        s = success_set(eps_by[label], ids)
        both = len(sref & s); ref_only = len(sref - s); other_only = len(s - sref)
        neither = len(ids) - both - ref_only - other_only
        rows.append([f"{ref} vs {label}", both, ref_only, other_only, neither, other_only - ref_only])
    out += [md_table(["pair", "both", f"{ref} only", "other only", "neither", "net"], rows), ""]
    return "\n".join(out)


def section_subsets(runs, eps_by, groups) -> str:
    header = ["subset", "n"] + [f"{l} SR" for l, _ in runs] + [f"{l} SPL" for l, _ in runs] + [f"dSR vs {runs[0][0]}" for _ in runs[1:]]
    rows = []
    for name, ids in groups.items():
        if not ids:
            continue
        sr = [mean_metric(eps_by[l], ids, "success") for l, _ in runs]
        spl = [mean_metric(eps_by[l], ids, "spl") for l, _ in runs]
        rows.append([name, len(ids)] + [pct(x) for x in sr] + [pct(x) for x in spl] + [f"{100 * (x - sr[0]):+.2f}" for x in sr[1:]])
    return "\n".join(["## 2. Hard subsets (GT-derived strata)", "", md_table(header, rows), ""])


def section_discordant(runs, eps_by, ids, features) -> str:
    ref = runs[0][0]
    sref = success_set(eps_by[ref], ids)
    rows = []
    for label, _ in runs[1:]:
        s = success_set(eps_by[label], ids)
        for name, sub in [(f"{ref} only", sorted(sref - s)), (f"{label} only", sorted(s - sref)),
                          ("both", sorted(sref & s)), ("neither", sorted(set(ids) - sref - s))]:
            rows.append([f"{ref} vs {label}", name, len(sub)] + feature_profile(features, sub))
    return "\n".join(["## 3. Discordant-pair profile (mean features)", "",
                      md_table(["pair", "set", "n"] + FEATURE_KEYS, rows), ""])


MODES = ["success", "stop_error", "wrong_branch", "off_path", "other"]


def section_modes(runs, eps_by, ids) -> str:
    rows = []
    for label, _ in runs:
        c = Counter(eps_by[label][i]["mode"] for i in ids)
        rows.append([label] + [c.get(m, 0) for m in MODES])
    return "\n".join(["## 4. Failure modes (free rollout)", "",
                      "stop_error = oracle success without success; wrong_branch = first deviation from the expert at a fork state; off_path = first deviation elsewhere; other = never deviated from the expert yet failed.", "",
                      md_table(["run"] + MODES, rows), ""])


def section_efficiency(runs, eps_by, ids) -> str:
    common = set(ids)
    for label, _ in runs:
        common &= success_set(eps_by[label], ids)
    common = sorted(common)
    rows = []
    for label, _ in runs:
        e = eps_by[label]
        rows.append([label, len(common), f"{mean_metric(e, common, 'steps_taken'):.2f}",
                     f"{mean_metric(e, common, 'path_length'):.2f}",
                     f"{np.mean([e[i]['n_revisits'] for i in common]):.2f}" if common else "-",
                     pct(mean_metric(e, common, "spl"))])
    return "\n".join(["## 5. Efficiency on episodes every run solves", "",
                      md_table(["run", "n", "steps", "PL (m)", "revisits", "SPL"], rows), ""])


def step_accuracy_rows(label: str, eps: Dict[str, dict], ids: Sequence[str]) -> List[List[str]]:
    steps = [s for i in ids for s in eps[i]["steps"] if s["valid"]]
    def acc(sel):
        return (f"{100 * np.mean([s['correct'] for s in sel]):.2f}", len(sel)) if sel else ("-", 0)
    groups = [
        ("all valid steps", steps),
        ("stop steps (expert=STOP)", [s for s in steps if s["is_stop"]]),
        ("move, 1 option", [s for s in steps if not s["is_stop"] and s["n_options"] <= 1]),
        ("move, 2 options", [s for s in steps if not s["is_stop"] and s["n_options"] == 2]),
        ("move, >=3 options", [s for s in steps if not s["is_stop"] and s["n_options"] >= 3]),
    ]
    rows = []
    for name, sel in groups:
        a, n = acc(sel)
        rows.append([label, name, n, a])
    return rows


def section_steps(title: str, runs, eps_by, ids, paired: bool) -> str:
    rows = []
    for label, _ in runs:
        rows += step_accuracy_rows(label, eps_by[label], ids)
    out = [f"## {title}", "", md_table(["run", "states", "n", "acc vs expert (%)"], rows), ""]
    if paired and len(runs) > 1:
        ref = runs[0][0]
        prow = []
        for label, _ in runs[1:]:
            a = {(i, s["step"]): s for i in ids for s in eps_by[ref][i]["steps"] if s["valid"] and not s["is_stop"]}
            b = {(i, s["step"]): s for i in ids for s in eps_by[label][i]["steps"] if s["valid"] and not s["is_stop"]}
            keys = sorted(set(a) & set(b))
            for name, sel in [("all move states", keys), ("fork states", [k for k in keys if a[k]["is_fork"]])]:
                both = sum(a[k]["correct"] and b[k]["correct"] for k in sel)
                ro = sum(a[k]["correct"] and not b[k]["correct"] for k in sel)
                oo = sum(b[k]["correct"] and not a[k]["correct"] for k in sel)
                prow.append([f"{ref} vs {label}", name, len(sel), both, ro, oo, len(sel) - both - ro - oo, oo - ro])
        out += ["Per-state 2x2 on identical states (teacher-forced):", "",
                md_table(["pair", "states", "n", "both right", f"{ref} only", "other only", "neither", "net"], prow), ""]
    return "\n".join(out)


def section_divergence(runs, eps_by, ids) -> str:
    ref = runs[0][0]
    rows = []
    for label, _ in runs[1:]:
        A, B = eps_by[ref], eps_by[label]
        sa, sb = success_set(A, ids), success_set(B, ids)
        rec = Counter()
        for i in ids:
            k = None
            for sa_, sb_ in zip(A[i]["steps"], B[i]["steps"]):
                if sa_["cur_vp"] != sb_["cur_vp"]:
                    break
                ea = sa_["vps"][sa_["executed"]] if sa_["executed"] < len(sa_["vps"]) else None
                eb = sb_["vps"][sb_["executed"]] if sb_["executed"] < len(sb_["vps"]) else None
                if ea != eb:
                    k = (sa_, sb_)
                    break
            if k is None:
                rec["identical"] += 1
                continue
            sa_, sb_ = k
            outcome = "both" if i in sa and i in sb else f"{ref} only" if i in sa else f"{label} only" if i in sb else "neither"
            who = "ref matches expert" if sa_["correct"] else "other matches expert" if sb_["correct"] else "neither matches expert"
            rec[(outcome, "fork" if sa_["is_fork"] else "non-fork", who)] += 1
        rows.append([f"{ref} vs {label}", "identical trajectories", "", "", rec.pop("identical", 0)])
        for (outcome, fork, who), n in sorted(rec.items(), key=lambda kv: (-kv[1])):
            rows.append([f"{ref} vs {label}", outcome, fork, who, n])
    return "\n".join(["## 7. First divergence between runs (free rollouts)", "",
                      "The first step whose executed waypoint differs; the state is shared up to there.", "",
                      md_table(["pair", "episode outcome", "state", "who matched expert there", "n"], rows), ""])


# ------------------------------------------------------------------ main


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    runs = parse_runs(args.run)
    eps_by = {label: load_dump(d, args.split) for label, d in runs}
    for e in eps_by.values():
        annotate(e, args.local_radius, args.fork_angle)
    ids = sorted(set.intersection(*[set(i for i, e in eps.items() if e["metric"] is not None) for eps in eps_by.values()]))
    features = json.load(open(args.features)) if args.features else {}

    parts = [f"# Paired-episode report ({args.split}, {len(ids)} common episodes)", ""]
    parts.append(section_overview(runs, eps_by, ids))
    if features:
        groups = strata(features, ids, args)
        parts.append(section_subsets(runs, eps_by, groups))
        parts.append(section_discordant(runs, eps_by, ids, features))
    parts.append(section_modes(runs, eps_by, ids))
    parts.append(section_efficiency(runs, eps_by, ids))
    parts.append(section_steps("6a. Decision accuracy vs expert (free rollout)", runs, eps_by, ids, paired=False))
    if args.forced:
        forced = parse_runs(args.forced)
        f_by = {label: load_dump(d, args.split) for label, d in forced}
        for e in f_by.values():
            annotate(e, args.local_radius, args.fork_angle)
        f_ids = sorted(set.intersection(*[set(e) for e in f_by.values()]))
        parts.append(section_steps(f"6b. Decision accuracy vs expert (teacher-forced, {len(f_ids)} episodes)", forced, f_by, f_ids, paired=True))
    parts.append(section_divergence(runs, eps_by, ids))
    text = "\n".join(parts)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
