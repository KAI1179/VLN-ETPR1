#!/usr/bin/env python3
"""Rank R2R-CE episodes for the qualitative figure (CPU only, Python >= 3.8).

Criteria (baseline not used):
  1. ours succeeds with high SPL;
  2. the predicted heading sequence matches the reference path;
  3. the predicted map is accurate along the reference path.

Inputs per split: ours per-episode metrics, the R2R-CE episode file (reference_path, start_position),
and the predicted raster maps R2R_<split>_<ep>.npz (keys: grid, direction_vectors, start_position).

Conventions (verified on the exported paper cases):
  - grid channels 0..26 are objects, 27..36 the ten region categories;
  - cell (row, col) covers world (x, z) = origin + (row, col) * 0.5 m, origin = start_world_xz - npz start_position
    (registration by the starting pose only, as assumed by the generator);
  - a stored direction (a, b) is the world direction (x, z) = (-b, -a);
  - GT heading directions follow the paper: normalized displacements along the reference path,
    a new direction whenever its cosine with the last recorded one is below 0.8, at most K = 5.

Example (ParaCloud paths as used for the paper cases):
  python rank_cases_by_map.py --out reports/paper_cases/map_cases
Outputs: <out>_all.json (every scored episode), <out>_top.md (filtered and ranked shortlist).
"""
import argparse, glob, gzip, json, math, os
import numpy as np

REGION0 = 27
CELL = 0.5
DEF = {
    "metrics": "/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/checkpoints/paper_case_ours_{split}/eval_results/stats_ep_ckpt_27000_{split}_r0_w1.json",
    "dataset": "/home/xukai/code/ETP-R1-main/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{split}/{split}.json.gz",
    "raster": "/data/xukai/etp-r1-snapshot/runtime-data/data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/{split}/cognitive_maps/raster",
}


# ---------------------------------------------------------------- loading
def _get(d, names):
    for n in names:
        if n in d:
            return d[n]
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} in keys {sorted(d)[:20]}")


def load_metrics(path):
    """stats_ep JSON {ep: {...}} or a JSONL with one row per episode (metrics at top level or under 'metrics')."""
    rows = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[str(_get(r, ["episode_id", "ep_id", "id"]))] = r.get("metrics", r)
    else:
        with open(path) as f:
            rows = {str(k): v for k, v in json.load(f).items()}
    out = {}
    for ep, r in rows.items():
        if not isinstance(r, dict):
            continue
        out[ep] = dict(success=float(_get(r, ["success", "SR", "sr"])), spl=float(_get(r, ["spl", "SPL"])),
                       ndtw=float(_get(r, ["ndtw", "nDTW"])), ne=float(_get(r, ["distance_to_goal", "NE", "ne"])))
    return out


def load_dataset(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        eps = json.load(f)["episodes"]
    return {str(e["episode_id"]): e for e in eps}


def find_npz(raster_root, split, ep, scene=None):
    if scene:
        p = os.path.join(raster_root, scene, f"R2R_{split}_{ep}.npz")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(raster_root, "*", f"R2R_{split}_{ep}.npz"))
    return hits[0] if hits else None


# ---------------------------------------------------------------- metrics
def densify(P, step=0.25):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def gt_dirs(P, K=5, thr=0.8):
    dirs = []
    for a, b in zip(P[:-1], P[1:]):
        v = b - a; n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        if not dirs or float(v @ dirs[-1]) < thr:
            dirs.append(v)
        if len(dirs) == K:
            break
    return dirs


def ang(u, v):
    return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v)))


def region_cells(grid):
    reg = (grid[REGION0:] > 0).any(0)
    a = reg[0::2, 0::2]
    aligned = all(np.array_equal(a, reg[i::2, j::2]) for i in (0, 1) for j in (0, 1))
    if aligned:   # generator output is max-pooled to 1 m: score 1 m blocks
        return (np.argwhere(reg[0::2, 0::2]) * 2 + 1.0) * CELL, 1.0
    return (np.argwhere(reg) + 0.5) * CELL, CELL


def score(npz_path, ep):
    z = np.load(npz_path, allow_pickle=True)
    S = np.asarray(ep["start_position"], float)[[0, 2]]
    origin = S - np.asarray(z["start_position"], float)
    rel, size = region_cells(z["grid"])
    gt = np.asarray(ep["reference_path"], float)[:, [0, 2]]
    res = dict(path_len=float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()))
    if len(rel) == 0:
        res.update(cov=0.0, prec=0.0)
    else:
        cen = origin + rel
        D = np.linalg.norm(densify(gt)[:, None] - cen[None], axis=-1)
        res["cov"] = float((D.min(1) <= 1.0).mean())      # route points within 1 m of a predicted region cell
        res["prec"] = float((D.min(0) <= 2.0).mean())     # predicted region cells within 2 m of the route
    pd = [np.array([-b, -a]) for a, b in np.asarray(z["direction_vectors"], float) if abs(a) + abs(b) > 1e-6]
    gd = gt_dirs(gt)
    errs = [abs(ang(p, g)) for p, g in zip(pd, gd)]
    turns = lambda ds: [ang(a, b) for a, b in zip(ds[:-1], ds[1:])]
    tp, tg = turns(pd), turns(gd)
    side = [(a > 0) == (b > 0) for a, b in zip(tp, tg) if abs(a) >= 30 and abs(b) >= 30]
    res.update(k_pred=len(pd), k_gt=len(gd), dir_err=[round(e) for e in errs],
               mean_err=float(np.mean(errs)) if errs else 180.0, max_err=float(max(errs)) if errs else 180.0,
               turn_side_ok=f"{sum(side)}/{len(side)}")
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val_unseen", "val_seen"])
    ap.add_argument("--metrics", default=DEF["metrics"])
    ap.add_argument("--dataset", default=DEF["dataset"])
    ap.add_argument("--raster", default=DEF["raster"])
    ap.add_argument("--out", default="map_cases")
    ap.add_argument("--min_spl", type=float, default=0.85)
    ap.add_argument("--min_kgt", type=int, default=3, help="at least 3 GT directions = at least 2 turns")
    ap.add_argument("--max_dk", type=int, default=1)
    ap.add_argument("--max_err", type=float, default=30.0, help="largest error over index-aligned directions")
    ap.add_argument("--min_cov", type=float, default=0.8)
    ap.add_argument("--min_prec", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    rows, missing = [], 0
    for sp in a.splits:
        met = load_metrics(a.metrics.format(split=sp))
        ds = load_dataset(a.dataset.format(split=sp))
        root = a.raster.format(split=sp)
        for ep, m in met.items():
            if m["success"] < 0.5 or ep not in ds:
                continue
            e = ds[ep]
            scene = e["scene_id"].split("/")[-2] if "/" in e["scene_id"] else None
            p = find_npz(root, sp, ep, scene)
            if p is None:
                missing += 1
                continue
            r = dict(split=sp, ep=ep, scene=scene, **m, **score(p, e))
            r["instruction"] = " ".join(e["instruction"]["instruction_text"].split()) if isinstance(e.get("instruction"), dict) else ""
            r["score"] = r["spl"] + r["cov"] + r["prec"] + 1 - min(r["mean_err"], 90) / 90
            rows.append(r)
    with open(a.out + "_all.json", "w") as f:
        json.dump(rows, f, indent=1)

    keep = [r for r in rows if r["spl"] >= a.min_spl and r["k_gt"] >= a.min_kgt and abs(r["k_pred"] - r["k_gt"]) <= a.max_dk
            and r["max_err"] <= a.max_err and r["cov"] >= a.min_cov and r["prec"] >= a.min_prec]
    keep.sort(key=lambda r: -r["score"])
    def n_pass(me, cv):
        return sum(1 for r in rows if r["spl"] >= a.min_spl and r["k_gt"] >= a.min_kgt and abs(r["k_pred"] - r["k_gt"]) <= a.max_dk
                   and r["max_err"] <= me and r["cov"] >= cv and r["prec"] >= a.min_prec)
    grid = ", ".join(f"err<={me}/cov>={cv}: {n_pass(me, cv)}" for me in (30, 45) for cv in (0.8, 0.7))
    lines = [f"# Map-faithful successful episodes: {len(keep)} pass / {len(rows)} successful episodes scored "
             f"({missing} without a raster map)", "", f"Pass counts under relaxed thresholds: {grid}", "",
             f"Filters: SPL>={a.min_spl}, K_gt>={a.min_kgt}, |K_pred-K_gt|<={a.max_dk}, max dir err<={a.max_err} deg, "
             f"cov>={a.min_cov}, prec>={a.min_prec}", "",
             "| # | split | ep | scene | SPL | nDTW | path m | K pred/gt | dir err (deg) | turn sides | cov | prec | instruction |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(keep[:a.top]):
        lines.append(f"| {i + 1} | {r['split']} | {r['ep']} | {r['scene']} | {r['spl']:.2f} | {r['ndtw']:.2f} | {r['path_len']:.1f} | "
                     f"{r['k_pred']}/{r['k_gt']} | {r['dir_err']} | {r['turn_side_ok']} | {r['cov']:.2f} | {r['prec']:.2f} | {r['instruction']} |")
    with open(a.out + "_top.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + min(a.top, len(keep)) + 2]))


if __name__ == "__main__":
    main()
