import json
import gzip
import numpy as np

W = "/home/user/agentic-nav"
CONN = "/home/user/VLN-ETPR1/precompute_img_features/connectivity"
r2r = {
    str(x["path_id"]): x
    for x in json.load(
        open(f"{W}/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json")
    )
}
vl = json.load(gzip.open(f"{W}/MIP/splits/r2r/rand100/rand100.json.gz"))["episodes"]
rand = json.load(open(f"{W}/reports/rand100_ids.json"))
by_eid = {str(e["episode_id"]): e for e in vl}
rows = []
for eid, tid, iid in rand:
    e = by_eid[str(eid)]
    x = r2r[str(tid)]
    conn = json.load(open(f"{CONN}/{x['scan']}_connectivity.json"))
    pos = {
        c["image_id"]: np.array([c["pose"][3], c["pose"][7], c["pose"][11]])
        for c in conn
    }
    vp = np.stack([pos[v] for v in x["path"]])  # mp3d (x, y, z-up), camera position
    ref = np.array(e["reference_path"])  # habitat (x, y-up, z)
    ref_mp3d = ref[:, [0, 2, 1]] * np.array([
        1,
        -1,
        1,
    ])  # mp3d_x = hab_x, mp3d_y = -hab_z, mp3d_z = hab_y
    same_len = len(vp) == len(ref)
    horiz = (
        np.linalg.norm((vp - ref_mp3d)[:, :2], axis=1) if same_len else None
    )  # per-point, same order
    vert = (vp - ref_mp3d)[:, 2] if same_len else None
    # order-free horizontal nearest-neighbour error, too
    d2 = np.linalg.norm(vp[:, None, :2] - ref_mp3d[None, :, :2], axis=-1).min(1)
    rows.append((
        eid,
        tid,
        len(vp),
        len(ref),
        same_len,
        float(horiz.max()) if same_len else None,
        float(vert.mean()) if same_len else None,
        float(vert.min()) if same_len else None,
        float(vert.max()) if same_len else None,
        float(d2.mean()),
    ))
n_same = sum(r[4] for r in rows)
print("episodes with len(R2R path) == len(reference_path):", n_same, "/", len(rows))
print(
    "episodes with same length AND max per-point HORIZONTAL error < 0.5 m (same order):",
    sum(1 for r in rows if r[4] and r[5] < 0.5),
    "/",
    len(rows),
)
print(
    "episodes with same length AND max per-point HORIZONTAL error < 0.05 m:",
    sum(1 for r in rows if r[4] and r[5] < 0.05),
    "/",
    len(rows),
)
print("max horizontal per-point error over all:", max(r[5] for r in rows if r[4]))
v = np.array([r[6] for r in rows if r[4]])
vmin = min(r[7] for r in rows if r[4])
vmax = max(r[8] for r in rows if r[4])
print(
    "vertical offset mp3d_z(camera) - hab_y(floor): mean %.3f, per-point min %.3f, max %.3f"
    % (v.mean(), vmin, vmax)
)
print(
    "order-free horizontal NN mean error < 0.5 m:",
    sum(1 for r in rows if r[9] < 0.5),
    "/",
    len(rows),
)
print(
    "first 8 rows (eid, tid, nvp, nref, same_len, max_horiz, mean_vert, min_vert, max_vert, nn_horiz):"
)
for r in rows[:8]:
    print("  ", tuple(round(x, 3) if isinstance(x, float) else x for x in r))
# start position / goal vs first / last viewpoint
sp = []
gl = []
for eid, tid, iid in rand:
    e = by_eid[str(eid)]
    x = r2r[str(tid)]
    conn = json.load(open(f"{CONN}/{x['scan']}_connectivity.json"))
    pos = {
        c["image_id"]: np.array([c["pose"][3], c["pose"][7], c["pose"][11]])
        for c in conn
    }
    s = np.array(e["start_position"])
    s = s[[0, 2, 1]] * np.array([1, -1, 1])
    g = np.array(e["goals"][0]["position"])
    g = g[[0, 2, 1]] * np.array([1, -1, 1])
    sp.append(float(np.linalg.norm((pos[x["path"][0]] - s)[:2])))
    gl.append(float(np.linalg.norm((pos[x["path"][-1]] - g)[:2])))
print(
    "start_position vs first R2R viewpoint, horizontal: max %.3f m; goal vs last viewpoint: max %.3f m"
    % (max(sp), max(gl))
)
