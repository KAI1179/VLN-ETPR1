#!/usr/bin/env python3
"""Paired-episode report from decision dumps (scripts/analysis/dump_eval.sh).

    python scripts/analysis/paired_report.py \\
        --features data/logs/analysis/features_val_unseen.json \\
        --own base=data/logs/checkpoints/pa_base_own_val_unseen/eval_results \\
              try5=data/logs/checkpoints/pa_try5_own_val_unseen/eval_results \\
              kd=data/logs/checkpoints/pa_kd_own_val_unseen/eval_results \\
        --shared base=data/logs/checkpoints/pa_base_shared_val_unseen/eval_results \\
                 try5=data/logs/checkpoints/pa_try5_shared_val_unseen/eval_results \\
                 kd=data/logs/checkpoints/pa_kd_shared_val_unseen/eval_results \\
        --out data/logs/analysis/report_val_unseen.md

The first label of each option is the reference every other run is paired
against.  All numbers are counts and means; no intervals.

Reference action of a step ("ref"):
  own rollouts     the SPL expert's choice at the policy's own state;
  shared states    the expert action every model executed (D8 replay), so
                   all models are scored against the same action on the
                   same state.
Steps where the episode is cut (last allowed step, or no ghost left) carry a
forced STOP and are not scored.

Step category, from ref:
  stop       ref = STOP (agent within 1.5 m of the goal);
  backtrack  ref is a ghost not observed from the current node (the agent
             must travel back through the graph to reach it);
  turn       ref is a candidate of the current node more than --turn-angle
             away from the agent's heading;
  straight   the remaining candidate steps.

Sections
  0  feature distributions (to pick stratum thresholds)
  1  per-run SR / SPL / OSR / nDTW / OSR-SR on common episodes, paired 2x2
  2  hard-subset SR / SPL table on GT-derived strata
  3  mean features of the discordant episodes
  4  failure modes (own rollouts): stop error / wrong turn / off path / other
  5  efficiency on episodes every run solves: steps, path length, backtracks
  6  decision accuracy vs ref by step category: own rollouts; shared states
     with the per-state 2x2 (same state verified by position + graph)
  7  first divergence between own rollouts that share a state prefix
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tap import Tap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CATEGORIES = ["stop", "backtrack", "turn", "straight"]


class Args(Tap):
    own: List[str]
    """LABEL=DIR, own-rollout dumps; the first is the reference."""
    shared: List[str] = []
    """LABEL=DIR, shared-state dumps (optional)."""
    features: Optional[str] = None
    """episode_features.py output; without it sections 0, 2, 3 are skipped."""
    split: str = "val_unseen"
    out: Optional[str] = None
    turn_angle: float = 45.0
    option_angle: float = 45.0
    """Degrees: candidates of the current node closer in bearing are one option."""
    fork_key: str = "n_forks3_graph"
    """Feature for the multi-fork stratum (n_forks3_graph, n_forks_graph or n_turns)."""
    min_forks: int = 2
    min_turns: int = 2
    min_room_changes: int = 2
    min_decoy_categories: int = 2
    long_quantile: float = 0.67
    spatial_quantile: float = 0.5


# ----------------------------------------------------------------- loading


def load_dump(directory: str, split: str) -> Tuple[dict, Dict[str, dict]]:
    files = sorted(glob.glob(os.path.join(directory, f"decisions_{split}_r*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no decisions_{split}_r*.jsonl in {directory}")
    header: dict = {}
    eps: Dict[str, dict] = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["kind"] == "header":
                    header = header or r
                elif r["kind"] == "step":
                    eps.setdefault(str(r["ep"]), {"steps": [], "metric": None})["steps"].append(r)
                elif r["kind"] == "episode":
                    e = eps.setdefault(str(r["ep"]), {"steps": [], "metric": None})
                    e["metric"] = r["metric"]
                    e["traj"] = r.get("traj", [])
    for e in eps.values():
        e["steps"].sort(key=lambda r: r["step"])
    return header, eps


def parse_runs(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected LABEL=DIR, got {it!r}")
        out.append(tuple(it.split("=", 1)))  # type: ignore[arg-type]
    return out


# ------------------------------------------------------------ step features


def _bearing_from_heading(step: dict, idx: int) -> float:
    """|angle| between the agent's forward direction and the vp at idx (rad)."""
    cur = np.asarray(step["cur_pos"], dtype=float)
    tgt = np.asarray(step["pos"][idx], dtype=float)
    d = tgt - cur
    h = step["heading"]
    fx, fz = -math.sin(h), -math.cos(h)
    return abs(math.atan2(fx * d[2] - fz * d[0], fx * d[0] + fz * d[2]))


def _options(step: dict, angle_deg: float) -> int:
    cur = np.asarray(step["cur_pos"], dtype=float)
    thr = math.radians(angle_deg)
    clusters: List[float] = []
    for pos, local in zip(step["pos"], step.get("front_cur", [])):
        if not local:
            continue
        d = np.asarray(pos, dtype=float) - cur
        b = math.atan2(d[2], d[0])
        if all(abs((b - c + math.pi) % (2 * math.pi) - math.pi) > thr for c in clusters):
            clusters.append(b)
    return len(clusters)


def signature(step: dict) -> tuple:
    return (
        tuple(round(x, 3) for x in step["cur_pos"]),
        tuple(step["vps"]),
        tuple(None if p is None else tuple(round(x, 3) for x in p) for p in step["pos"]),
    )


def annotate(header: dict, eps: Dict[str, dict], args: Args) -> None:
    shared = bool(header.get("d8_shared_states"))
    max_steps = int(header.get("max_steps", 15) or 15)
    for e in eps.values():
        e["n_backtracks"] = 0
        e["first_dev"] = None
        for s in e["steps"]:
            s["ref"] = s["executed"] if shared else s["teacher"]
            cut = s["no_vp_left"] or s["step"] >= max_steps - 1
            s["valid"] = (not cut) and s["ref"] != -100 and s["ref"] < len(s["vps"])
            s["n_options"] = _options(s, args.option_angle)
            if not s["valid"]:
                s["cat"] = None
            elif s["ref"] == 0:
                s["cat"] = "stop"
            elif not s["front_cur"][s["ref"]]:
                s["cat"] = "backtrack"
            elif _bearing_from_heading(s, s["ref"]) > math.radians(args.turn_angle):
                s["cat"] = "turn"
            else:
                s["cat"] = "straight"
            s["correct"] = bool(s["valid"] and s["student"] == s["ref"])
            ex = s["executed"]
            if 0 < ex < len(s["vps"]) and not s["front_cur"][ex]:
                e["n_backtracks"] += 1
            if e["first_dev"] is None and s["valid"] and s["cat"] != "stop" and not s["correct"]:
                e["first_dev"] = s
        m = e["metric"] or {}
        if m.get("success", 0) >= 1:
            e["mode"] = "success"
        elif m.get("oracle_success", 0) >= 1:
            e["mode"] = "stop_error"
        elif e["first_dev"] is not None and e["first_dev"]["cat"] == "turn":
            e["mode"] = "wrong_turn"
        elif e["first_dev"] is not None:
            e["mode"] = "off_path"
        else:
            e["mode"] = "other"


# ----------------------------------------------------------------- tables


def md_table(header: List[str], rows: List[List]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.2f}"


def mean_metric(eps: Dict[str, dict], ids: Sequence[str], key: str) -> float:
    vals = [eps[i]["metric"][key] for i in ids if eps[i]["metric"] is not None and key in eps[i]["metric"]]
    return float(np.mean(vals)) if vals else float("nan")


def success_set(eps: Dict[str, dict], ids: Sequence[str]) -> set:
    return {i for i in ids if eps[i]["metric"] and eps[i]["metric"]["success"] >= 1}


FEATURE_KEYS = ["path_len_m", "n_turns", "n_forks_graph", "n_forks3_graph", "n_room_changes",
                "n_landmarks", "n_decoys", "n_decoy_categories", "n_spatial_words", "llm_region_hit"]


def _vals(features, ids, k):
    return [features[i][k] for i in ids if i in features and features[i].get(k) is not None and features[i].get(k) != -1]


def section_distributions(features, ids) -> str:
    rows = []
    for k in FEATURE_KEYS:
        xs = _vals(features, ids, k)
        if xs:
            q = np.quantile(xs, [0.1, 0.25, 0.5, 0.75, 0.9])
            rows.append([k, len(xs), f"{np.mean(xs):.2f}"] + [f"{v:.2f}" for v in q])
    return "\n".join(["## 0. Feature distributions (common episodes)", "",
                      md_table(["feature", "n", "mean", "p10", "p25", "p50", "p75", "p90"], rows), ""])


def strata(features: Dict[str, dict], ids: Sequence[str], args: Args) -> Dict[str, List[str]]:
    fs = {i: features[i] for i in ids if i in features}

    def ge(k, t):
        return [i for i, v in fs.items() if v.get(k) is not None and v[k] != -1 and v[k] >= t]

    def lt(k, t):
        return [i for i, v in fs.items() if v.get(k) is not None and v[k] != -1 and v[k] < t]

    out: Dict[str, List[str]] = {"all": list(ids)}
    fk = args.fork_key
    out[f"multi-fork ({fk}>={args.min_forks})"] = ge(fk, args.min_forks)
    out[f"few forks ({fk}<{args.min_forks})"] = lt(fk, args.min_forks)
    out[f"multi-turn (n_turns>={args.min_turns})"] = ge("n_turns", args.min_turns)
    out[f"few turns (n_turns<{args.min_turns})"] = lt("n_turns", args.min_turns)
    out[f"multi-room (n_room_changes>={args.min_room_changes})"] = ge("n_room_changes", args.min_room_changes)
    out[f"decoys (n_decoy_categories>={args.min_decoy_categories})"] = ge("n_decoy_categories", args.min_decoy_categories)
    xs = _vals(features, ids, "path_len_m")
    if xs:
        t = float(np.quantile(xs, args.long_quantile))
        out[f"long path (>= {t:.1f} m)"] = ge("path_len_m", t)
        out[f"short path (< {t:.1f} m)"] = lt("path_len_m", t)
    xs = _vals(features, ids, "n_spatial_words")
    if xs:
        t = float(np.quantile(xs, args.spatial_quantile))
        out[f"spatial language (words>={t:.0f})"] = ge("n_spatial_words", t)
    xs = _vals(features, ids, "llm_region_hit")
    if xs:
        t = float(np.median(xs))
        out[f"LLM map good (hit>={t:.2f})"] = ge("llm_region_hit", t)
        out[f"LLM map poor (hit<{t:.2f})"] = lt("llm_region_hit", t)
    return {k: v for k, v in out.items() if v}


def feature_profile(features: Dict[str, dict], ids: Sequence[str]) -> List[str]:
    return [f"{np.mean(xs):.2f}" if (xs := _vals(features, ids, k)) else "-" for k in FEATURE_KEYS]


# ------------------------------------------------------------- sections


def section_overview(runs, eps_by, ids) -> str:
    rows = []
    for label, _ in runs:
        e = eps_by[label]
        sr, osr = mean_metric(e, ids, "success"), mean_metric(e, ids, "oracle_success")
        rows.append([label, len(ids), pct(sr), pct(mean_metric(e, ids, "spl")), pct(osr),
                     pct(mean_metric(e, ids, "ndtw")), pct(osr - sr),
                     f"{mean_metric(e, ids, 'steps_taken'):.1f}", f"{mean_metric(e, ids, 'path_length'):.2f}"])
    out = ["## 1. Own rollouts on common episodes", "",
           md_table(["run", "n", "SR", "SPL", "OSR", "nDTW", "OSR-SR", "steps", "PL (m)"], rows), ""]
    ref = runs[0][0]
    sref = success_set(eps_by[ref], ids)
    rows = []
    for label, _ in runs[1:]:
        s = success_set(eps_by[label], ids)
        both, ro, oo = len(sref & s), len(sref - s), len(s - sref)
        rows.append([f"{ref} vs {label}", both, ro, oo, len(ids) - both - ro - oo, oo - ro])
    if rows:
        out += [md_table(["pair", "both succeed", f"{ref} only", "other only", "neither", "net"], rows), ""]
    return "\n".join(out)


def section_subsets(runs, eps_by, groups) -> str:
    ref = runs[0][0]
    header = ["subset", "n"] + [f"{l} SR" for l, _ in runs] + [f"{l} SPL" for l, _ in runs] \
        + [f"{l} - {ref} SR" for l, _ in runs[1:]]
    rows = []
    for name, ids in groups.items():
        sr = [mean_metric(eps_by[l], ids, "success") for l, _ in runs]
        spl = [mean_metric(eps_by[l], ids, "spl") for l, _ in runs]
        rows.append([name, len(ids)] + [pct(x) for x in sr] + [pct(x) for x in spl]
                    + [f"{100 * (x - sr[0]):+.2f}" for x in sr[1:]])
    return "\n".join(["## 2. Hard subsets (GT-derived strata, own rollouts)", "", md_table(header, rows), ""])


def section_discordant(runs, eps_by, ids, features) -> str:
    ref = runs[0][0]
    sref = success_set(eps_by[ref], ids)
    rows = []
    for label, _ in runs[1:]:
        s = success_set(eps_by[label], ids)
        for name, sub in [(f"{ref} only", sref - s), (f"{label} only", s - sref),
                          ("both", sref & s), ("neither", set(ids) - sref - s)]:
            rows.append([f"{ref} vs {label}", name, len(sub)] + feature_profile(features, sorted(sub)))
    return "\n".join(["## 3. Mean features of discordant episodes", "",
                      md_table(["pair", "succeeds", "n"] + FEATURE_KEYS, rows), ""])


MODES = ["success", "stop_error", "wrong_turn", "off_path", "other"]


def section_modes(runs, eps_by, ids) -> str:
    rows = [[label] + [Counter(eps_by[label][i]["mode"] for i in ids).get(m, 0) for m in MODES]
            for label, _ in runs]
    return "\n".join(["## 4. Failure modes (own rollouts)", "",
                      "stop_error: reached the goal area but stopped elsewhere; wrong_turn: first disagreement "
                      "with the expert at a turn step; off_path: first disagreement at a straight or backtrack "
                      "step; other: never disagreed with the expert yet failed.", "",
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
                     f"{np.mean([e[i]['n_backtracks'] for i in common]):.2f}" if common else "-",
                     pct(mean_metric(e, common, "spl"))])
    return "\n".join(["## 5. Efficiency on episodes every run solves", "",
                      md_table(["run", "n", "low-level steps", "PL (m)", "backtracks", "SPL"], rows), ""])


def _acc(sel: List[dict]) -> str:
    return f"{100 * np.mean([s['correct'] for s in sel]):.2f}" if sel else "-"


def section_accuracy(title: str, runs, eps_by, ids) -> str:
    header = ["run"] + [f"{c} (n / acc%)" for c in CATEGORIES] + ["turn, >=3 options", "all moves"]
    rows = []
    for label, _ in runs:
        steps = [s for i in ids for s in eps_by[label][i]["steps"] if s["valid"]]
        cells = [label]
        for c in CATEGORIES:
            sel = [s for s in steps if s["cat"] == c]
            cells.append(f"{len(sel)} / {_acc(sel)}")
        sel = [s for s in steps if s["cat"] == "turn" and s["n_options"] >= 3]
        cells.append(f"{len(sel)} / {_acc(sel)}")
        sel = [s for s in steps if s["cat"] != "stop"]
        cells.append(f"{len(sel)} / {_acc(sel)}")
        rows.append(cells)
    return "\n".join([f"## {title}", "", md_table(header, rows), ""])


def section_shared_pairs(runs, eps_by, ids) -> str:
    ref = runs[0][0]
    rows, checked, mismatched = [], 0, 0
    for label, _ in runs[1:]:
        pairs = []
        for i in ids:
            for a, b in zip(eps_by[ref][i]["steps"], eps_by[label][i]["steps"]):
                checked += 1
                same_ref = a["ref"] == b["ref"] or not (a["valid"] and b["valid"])  # cut steps differ by design
                if a["step"] != b["step"] or signature(a) != signature(b) or not same_ref:
                    mismatched += 1
                    break  # states after a mismatch are not shared
                if a["valid"] and b["valid"]:
                    pairs.append((a, b))
        for cat in CATEGORIES[1:] + ["all moves"]:
            sel = [(a, b) for a, b in pairs if (a["cat"] == cat if cat != "all moves" else a["cat"] != "stop")]
            both = sum(a["correct"] and b["correct"] for a, b in sel)
            ro = sum(a["correct"] and not b["correct"] for a, b in sel)
            oo = sum(b["correct"] and not a["correct"] for a, b in sel)
            rows.append([f"{ref} vs {label}", cat, len(sel), both, ro, oo, len(sel) - both - ro - oo, oo - ro])
    note = (f"State check: {checked} step pairs compared, {mismatched} episode(s) cut at a state mismatch "
            "(position, graph or reference action differ; expected 0 under D8 replay).")
    return "\n".join(["Per-state 2x2 on identical states:", "", note, "",
                      md_table(["pair", "category", "n", "both right", f"{ref} only", "other only", "neither", "net"], rows), ""])


def section_divergence(runs, eps_by, ids) -> str:
    ref = runs[0][0]
    rows = []
    for label, _ in runs[1:]:
        A, B = eps_by[ref], eps_by[label]
        sa, sb = success_set(A, ids), success_set(B, ids)
        rec: Counter = Counter()
        for i in ids:
            found = None
            for x, y in zip(A[i]["steps"], B[i]["steps"]):
                if signature(x) != signature(y):
                    found = "state differs first"
                    break
                if x["executed"] != y["executed"]:
                    found = (x, y)
                    break
            if found is None:
                rec[("identical until one run ends", "", "")] += 1
                continue
            if isinstance(found, str):
                rec[(found, "", "")] += 1
                continue
            x, y = found
            outcome = ("both" if i in sa and i in sb else f"{ref} only" if i in sa
                       else f"{label} only" if i in sb else "neither")
            who = (f"{ref} = expert" if x["correct"] and not y["correct"] else
                   f"{label} = expert" if y["correct"] and not x["correct"] else
                   "neither = expert" if not x["correct"] and not y["correct"] else "both = expert")
            rec[(outcome, x["cat"] or "cut", who)] += 1
        for (outcome, cat, who), n in sorted(rec.items(), key=lambda kv: -kv[1]):
            rows.append([f"{ref} vs {label}", outcome, cat, who, n])
    return "\n".join(["## 7. First divergence between own rollouts", "",
                      "The first step where the two runs, on the same state, execute different actions.", "",
                      md_table(["pair", "who succeeds", "step category", "who agreed with expert there", "n"], rows), ""])


# ------------------------------------------------------------------ main


def load_runs(items, args):
    runs = parse_runs(items)
    eps_by = {}
    for label, d in runs:
        header, eps = load_dump(d, args.split)
        annotate(header, eps, args)
        eps_by[label] = eps
    ids = sorted(set.intersection(*[{i for i, e in eps.items() if e["metric"] is not None}
                                    for eps in eps_by.values()]), key=lambda x: int(x) if x.isdigit() else x)
    return runs, eps_by, ids


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    runs, eps_by, ids = load_runs(args.own, args)
    features = json.load(open(args.features)) if args.features else {}

    parts = [f"# Paired-episode report ({args.split})", "",
             f"Own rollouts: {len(ids)} episodes common to all runs.", ""]
    if features:
        parts.append(section_distributions(features, ids))
    parts.append(section_overview(runs, eps_by, ids))
    if features:
        parts.append(section_subsets(runs, eps_by, strata(features, ids, args)))
        if len(runs) > 1:
            parts.append(section_discordant(runs, eps_by, ids, features))
    parts.append(section_modes(runs, eps_by, ids))
    parts.append(section_efficiency(runs, eps_by, ids))
    parts.append(section_accuracy("6a. Decision accuracy vs expert, own rollouts", runs, eps_by, ids))
    if args.shared:
        s_runs, s_eps, s_ids = load_runs(args.shared, args)
        parts.append(section_accuracy(f"6b. Decision accuracy vs expert, shared states ({len(s_ids)} episodes)",
                                      s_runs, s_eps, s_ids))
        if len(s_runs) > 1:
            parts.append(section_shared_pairs(s_runs, s_eps, s_ids))
    if len(runs) > 1:
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
