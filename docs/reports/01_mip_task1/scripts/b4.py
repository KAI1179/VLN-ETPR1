import json
import gzip
import numpy as np

W = "/home/user/agentic-nav"
CONN = "/home/user/VLN-ETPR1/precompute_img_features/connectivity"
R2R_DISC = f"{W}/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json"
VLNCE = f"{W}/MIP/splits/r2r/rand100/rand100.json.gz"  # full val_unseen.json.gz not available on this host
print("CONN:", CONN, "| R2R_DISC:", R2R_DISC, "| VLNCE episodes:", VLNCE)
vl = json.load(gzip.open(VLNCE))["episodes"]
r2r = {str(x["path_id"]): x for x in json.load(open(R2R_DISC))}
rand = json.load(open(f"{W}/reports/rand100_ids.json"))
by_eid = {str(e["episode_id"]): e for e in vl}
names = ["[x,z,y]*[1,-1,1]", "[x,z,y]", "identity [x,y,z]", "[x,-z,y]"]
ok = 0
ok_any = 0
report = []
best_count = {n: 0 for n in names}
per_map_ok = {n: 0 for n in names}
seq_ok = 0
for eid, tid, iid in rand[:100]:
    e = by_eid.get(str(eid))
    x = r2r.get(str(tid))
    if e is None or x is None:
        report.append((tid, "missing"))
        continue
    scan = x["scan"]
    conn = json.load(open(f"{CONN}/{scan}_connectivity.json"))
    pos = {
        c["image_id"]: np.array([c["pose"][3], c["pose"][7], c["pose"][11]])
        for c in conn
    }
    vp_xyz = np.stack([
        pos[v] for v in x["path"]
    ])  # R2R viewpoint coords (MP3D frame, z up)
    ref = np.array(e["reference_path"])  # CE reference path (habitat frame: x, y up, z)
    cand = [
        ref[:, [0, 2, 1]] * np.array([1, -1, 1]),
        ref[:, [0, 2, 1]],
        ref,
        ref[:, [0, 2, 1]] * np.array([1, 1, -1]),
    ]
    errs = []
    for c in cand:
        d = np.linalg.norm(vp_xyz[:, None, :] - c[None, :, :], axis=-1).min(1)
        errs.append(float(d.mean()))
    i = int(np.argmin(errs))
    best_count[names[i]] += 1
    for n, v in zip(names, errs):
        if v < 0.5:
            per_map_ok[n] += 1
    # sequence-wise check under mapping 0 (same length, same order, per-point distance)
    c0 = cand[0]
    seq = (len(x["path"]) == len(ref)) and float(
        np.linalg.norm(vp_xyz - c0, axis=1).max()
    ) < 0.5
    seq_ok += int(seq)
    report.append((
        tid,
        len(x["path"]),
        len(ref),
        [round(v, 2) for v in errs],
        "seq_match" if seq else "seq_diff",
    ))
    if min(errs) < 0.5:
        ok += 1
print("axis mappings tested:", names)
print(
    "episodes with mean viewpoint->reference_path distance < 0.5 m under best axis mapping:",
    ok,
    "/",
    len(rand),
)
print("best mapping histogram:", best_count)
print("episodes < 0.5 m per mapping:", per_map_ok)
print(
    "episodes where mapping[0] gives same-length, same-order path with max per-point error < 0.5 m:",
    seq_ok,
    "/",
    len(rand),
)
print("first 10:", report[:10])
print(
    "all mean errors under mapping[0]: max =",
    max(r[3][0] for r in report if r[1] != "missing"),
)
# altitude check: habitat y vs mp3d z offset
diffs = []
for eid, tid, iid in rand[:100]:
    e = by_eid[str(eid)]
    x = r2r[str(tid)]
    conn = json.load(open(f"{CONN}/{x['scan']}_connectivity.json"))
    pos = {
        c["image_id"]: np.array([c["pose"][3], c["pose"][7], c["pose"][11]])
        for c in conn
    }
    diffs.append(float(pos[x["path"][0]][2] - e["start_position"][1]))
print(
    "mp3d_z(first viewpoint) - habitat_y(start): mean %.3f min %.3f max %.3f (camera height offset)"
    % (np.mean(diffs), np.min(diffs), np.max(diffs))
)
