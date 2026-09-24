"""B4: R2R viewpoint coordinates vs. CE reference_path, per axis mapping.

Env: CONN (dir of {scan}_connectivity.json), R2R_DISC (R2R_val_unseen.json),
VLNCE_EPISODES (val_unseen.json.gz or rand100.json.gz), REPORTS_DIR (rand100_ids.json).
Reports the task book's 3D criterion and the horizontal-only criterion (the MP3D
pose is the camera, ~1.4 m above the navmesh point the CE path uses).
"""

from __future__ import annotations

import sys

import numpy as np

from _common import env_path, episodes_of, load_json, viewpoint_positions

MAPPINGS = {
    "[x,z,y]*[1,-1,1]": lambda r: r[:, [0, 2, 1]] * np.array([1, -1, 1]),
    "[x,z,y]": lambda r: r[:, [0, 2, 1]],
    "identity": lambda r: r,
    "[x,z,y]*[1,1,-1]": lambda r: r[:, [0, 2, 1]] * np.array([1, 1, -1]),
}


def main() -> int:
    conn, disc, vlnce, reports = (
        env_path("CONN"),
        env_path("R2R_DISC"),
        env_path("VLNCE_EPISODES"),
        env_path("REPORTS_DIR"),
    )
    if (
        not all(p is not None and p.exists() for p in (conn, disc, vlnce))
        or reports is None
    ):
        print(
            "B4 SKIP: need CONN, R2R_DISC, VLNCE_EPISODES, REPORTS_DIR:",
            conn,
            disc,
            vlnce,
            reports,
        )
        return 2
    assert conn and disc and vlnce
    print("CONN:", conn, "| R2R_DISC:", disc, "| VLNCE:", vlnce)
    r2r = {str(x["path_id"]): x for x in load_json(disc)}
    by_eid = {str(e["episode_id"]): e for e in episodes_of(vlnce)}
    rand = load_json(reports / "rand100_ids.json")
    names = list(MAPPINGS)
    ok3d, best, same_len, horiz05, horiz005, nn05 = (
        0,
        dict.fromkeys(names, 0),
        0,
        0,
        0,
        0,
    )
    vert: list[float] = []
    rows = []
    for eid, tid, _ in rand:
        e, x = by_eid.get(str(eid)), r2r.get(str(tid))
        if e is None or x is None:
            rows.append((eid, tid, "missing"))
            continue
        pos = viewpoint_positions(conn, x["scan"])
        vp = np.array([pos[v] for v in x["path"]], dtype=float)
        ref = np.array(e["reference_path"], dtype=float)
        errs = {}
        for n, fn in MAPPINGS.items():
            c = fn(ref)
            errs[n] = float(
                np.linalg.norm(vp[:, None, :] - c[None, :, :], axis=-1).min(1).mean()
            )
        bn = min(errs.items(), key=lambda kv: kv[1])[0]
        best[bn] += 1
        ok3d += int(errs[bn] < 0.5)
        c0 = MAPPINGS[names[0]](ref)
        nn05 += int(
            float(
                np.linalg.norm(vp[:, None, :2] - c0[None, :, :2], axis=-1).min(1).mean()
            )
            < 0.5
        )
        if len(vp) == len(ref):
            same_len += 1
            h = np.linalg.norm((vp - c0)[:, :2], axis=1)
            horiz05 += int(float(h.max()) < 0.5)
            horiz005 += int(float(h.max()) < 0.05)
            vert.extend((vp - c0)[:, 2].tolist())
            rows.append((
                eid,
                tid,
                len(vp),
                len(ref),
                {k: round(v, 2) for k, v in errs.items()},
                round(float(h.max()), 3),
            ))
        else:
            rows.append((
                eid,
                tid,
                len(vp),
                len(ref),
                {k: round(v, 2) for k, v in errs.items()},
                "len differs",
            ))
    n = len(rand)
    print("axis mappings tested (mp3d <- habitat):", names)
    print(
        "task-book criterion — mean 3D viewpoint->reference_path distance < 0.5 m under best mapping:",
        ok3d,
        "/",
        n,
    )
    print("best mapping histogram:", best)
    print("len(R2R path) == len(reference_path):", same_len, "/", n)
    print(
        "same length AND max per-point HORIZONTAL error < 0.5 m under",
        names[0],
        ":",
        horiz05,
        "/",
        n,
    )
    print("same length AND max per-point HORIZONTAL error < 0.05 m:", horiz005, "/", n)
    print("order-free horizontal NN mean error < 0.5 m:", nn05, "/", n)
    if vert:
        v = np.array(vert)
        print(
            "vertical offset mp3d_z(camera) - hab_y(floor): mean %.3f min %.3f max %.3f"
            % (v.mean(), v.min(), v.max())
        )
    print("first 10 rows (eid, tid, n_vp, n_ref, mean3d per mapping, max horizontal):")
    for r in rows[:10]:
        print("  ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
