# Four selected qualitative-map cases: completed export

**Steps1–3 complete. The user rendered the recorded poses on GPU6; CLI preparation, verification and packaging were CPU-only. No policy was rerun. No Git commit or push was performed.**

## Summary

SPL/nDTW/cov/prec are fractions; NE is metres; direction errors are degrees.

| Split | Episode | Scene | SPL | nDTW | NE | High frames | K pred/GT | Mean/max direction error | cov/prec |
|---|---|---|---|---|---|---|---|---|---|
| val_seen | 302 | 1LXtFkjw3qL | 1.000000 | 0.922665 | 1.481580 | 11 | 3/4 | 4.132712/4.418620 | 1.000000/0.830769 |
| val_unseen | 303 | zsNo4HB9uLZ | 0.953169 | 0.940363 | 1.053509 | 9 | 3/4 | 15.360877/18.159363 | 1.000000/0.863636 |
| val_seen | 431 | sKLMLpTHeUy | 1.000000 | 0.953161 | 1.307240 | 5 | 3/3 | 10.664135/26.663493 | 1.000000/0.918919 |
| val_seen | 732 | aayBHfsNo7d | 0.931680 | 0.852103 | 1.012604 | 9 | 2/3 | 5.379716/10.735468 | 0.935484/0.606742 |

Direction error/cov/prec are copied from the task-provided map_cases_all.json, not recomputed by this exporter. K pred and GT were independently counted from the caches.

## Instructions and predicted region categories

### val_seen episode 302

Exit the bathroom toward the room and pass the black curtain, then pass the green couch and go out the white door on the right and stop at the top of the stairs.

Predicted regions: 33: private room; 34: bathroom/sanitary.

Direction errors per matched slot from selection table: [4, 4, 4]; turn-side agreement: 2/2.

Directory: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/map_vis/val_seen_vis/302`.

### val_unseen episode 303

Walk down the hallway and take a left. At the end of the hallway take a left and a right at the next hall and enter the bedroom on your left.

Predicted regions: 30: utility/service; 32: circulation; 33: private room; 34: bathroom/sanitary.

Direction errors per matched slot from selection table: [18, 16, 12]; turn-side agreement: 2/2.

Directory: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/map_vis/val_unseen_vis/303`.

### val_seen episode 431

walk away from closet, past sinks, turn right into bedroom, stop in front of doorway on right.

Predicted regions: 32: circulation; 33: private room; 34: bathroom/sanitary.

Direction errors per matched slot from selection table: [1, 4, 27]; turn-side agreement: 2/2.

Directory: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/map_vis/val_seen_vis/431`.

### val_seen episode 732

Walk toward the staircase and to the left of the stairs. Turn left by the table and chairs and stop in front of the desk.

Predicted regions: 28: living/social space; 31: work/study; 32: circulation; 34: bathroom/sanitary; 35: dining/food.

Direction errors per matched slot from selection table: [11, 0]; turn-side agreement: 1/1.

Directory: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/map_vis/val_seen_vis/732`.

## Step1 validation

| Case | Stats/JSONL metrics | Selection-table success/SPL/nDTW/NE | Raster exact | Direction values/K pred | K GT | Goal included |
|---|---|---|---|---|---|---|
| val_seen/302 | PASS | PASS | PASS | PASS | PASS | PASS |
| val_unseen/303 | PASS | PASS | PASS | PASS | PASS | PASS |
| val_seen/431 | PASS | PASS | PASS | PASS | PASS | PASS |
| val_seen/732 | PASS | PASS | PASS | PASS | PASS | PASS |

Detailed exact source paths and checks: [validation.json](validation.json).

- High-level frames are terminal stats high_level_step +1 because the stats field is a zero-based last-step index.
- Every case has the existing meta.json schema, full map NPZ/source, channel names, directions, per-frame poses/nodes/probabilities and both recorded trajectories.
- At Step1, topdown.origin_world_xz_m and meters_per_pixel_xz were present but null/pending. The user-run renderer subsequently filled the actual values, and Step3 confirmed they are finite and agree with each crop. Requested resolution is0.02m; effective grid pitches are approximately0.02000–0.020014m because native bounds/raster dimensions are rounded.
- All four goals equal gt_path[-1] exactly, so the unchanged reference/trajectory crop includes each goal. The native crop adds3m padding and covers both arms.

## GPU command: user executes one line

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-case-mpl /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/map_vis.py --render
```

Order: val_seen302 → val_unseen303 → val_seen431 → val_seen732. One simulator/case at a time. No navigation weights are loaded and no policy is rerun.

Per case: ours/step_XX_{pano,topdown,fpv}.png for every high-level frame, topdown_crop.png, cogmap_{regions,objects,heading}{,_projected}.png; render_status becomes complete.

## Differences from previous workflow

- The same materials.prepare/render and d7_case2 rendering functions are reused; existing model/trainer/eval/materials.py files are not modified.
- The current exporter has no ours-only switch and requires both arms. Per task instructions, zeromap is exported/rendered too; it is not a selection criterion and no script is changed just to omit it.
- Previous local top20 outputs used vis/<ep> and vis/val_seen/<ep>. Only the existing --output location changes to map_vis/{val_unseen_vis,val_seen_vis}/<ep>, matching the requested package layout.
- Map-frame registration remains the same assumed translation from start pose. It is not proof of physically exact room placement.
- Panoramas are twelve overlapping90-degree perspective views,7680×480 total. Heading rays are numbered directions, not invented path locations.

## Stop point1

Passed: the user confirmed all four renders completed, using CUDA_VISIBLE_DEVICES=6 in place of the command's GPU7 placeholder. Step3 was executed only after that confirmation.

## Step3: final completeness verification and package

All checks passed; the independent read-only CPU audit agreed with the final
packaging verifier:

- All34ours frames have panorama, top-down and FPV PNGs; the33zeromap frames were also checked.
- Every panorama is7680×480 and every FPV is640×480. All PNGs passed PIL verification.
- Each case has all six `cogmap_{regions,objects,heading}{,_projected}.png` files and `topdown_crop.png`.
- Every `meta.json` has `render_status=complete`, finite actual top-down origin/resolution, and a crop size matching its image.
- The complete ours trajectory, reference path and target are inside each recorded crop.
- Archive file members exactly equal the249verified case files; no outer `map_vis/` prefix or absolute archive paths are used.

| Case | Ours frames | Zeromap frames | Ours files | Zeromap files | Shared files | Total case files |
|---|---:|---:|---:|---:|---:|---:|
|val_seen/302|11|11|34|34|10|78|
|val_unseen/303|9|9|28|28|10|66|
|val_seen/431|5|5|16|16|10|42|
|val_seen/732|9|8|28|25|10|63|
|Total|34|33|106|103|40|249|

Each model directory contains three PNGs per frame plus `map_inputs.npz`.
The10shared files per case are `meta.json`, `cogmap_inputs.npz`,
`cogmap_source.npz`, `topdown_crop.png` and the six cognitive-map PNGs.
The full per-file inventory and actual top-down transforms are recorded in
[render_validation.json](render_validation.json).

Archive: [map_vis_4cases.tar.gz](map_vis_4cases.tar.gz)

- Absolute path: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/paper_cases/map_vis/map_vis_4cases.tar.gz`
- Size: **199,470,688 bytes** (199.47MB; approximately190.23MiB).
- SHA256: `a309ac43163badaca767921702471f3960d1e15f01c7206d41059f303919bad1`

```text
val_seen_vis/
  302/
  431/
  732/
val_unseen_vis/
  303/
```

The archive includes zeromap because the unchanged source exporter requires
both arms. All raw records and machine-specific source/image paths inside
`meta.json` are preserved; the archive's file paths themselves are relative.

## Stop point2

Verification, packaging and report update completed. No further GPU tasks,
background jobs, Git commits or pushes were started.
