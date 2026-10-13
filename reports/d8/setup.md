# D8 execution setup

## Accepted protocol

- Baseline: ETP-R1 iter25000, no map.
- Try5: no-KD iter28000, **with its own LLM map**.
- KD FULL: iter27000, **with its own LLM map**.
- All use control, val_unseen, 150 instruction tokens and 15 high-level steps.
- The user authorized local GPU execution on GPU 4–7. Execute one inference job at a time, never concurrent model jobs. No GPU 0–3 use.
- Shared expert-replay states (S) are the primary probe; own rollouts (O) are descriptive.
- Dump the complete 1536-dimensional STOP-head input using a detached read-only pre-hook.
- Compare episode, position, step, candidate IDs and coordinates for **all** S frames; stop if states differ.
- Step-only control uses t / 15, not future trajectory length t / T.
- Primary readout: balanced-accuracy paired scene-bootstrap difference; supporting improvement requires its 95% CI lower bound above zero.
- The label measures geodesic goal proximity, not complete instruction-stage progress.

## CPU work completed

`scripts/distill/termination_taxonomy.py` produced
`reports/d8/part_a_termination.md` and `.json` from existing four-rank D7-d evaluations.
True termination reasons are used; missing-cache episodes remain U.
Bootstrap: 2000 scene-cluster draws, seed 0. Paired comparisons exclude U.

Full-precision historical cut points from `reports/d7d_kd_control.json` are
8.236761794539953 / 10.376267476756617 m, giving tercile counts 612 / 612 / 615.
Rounded task-book values are used only for display; no episode lengths were changed.

## Remaining execution checks

Feature-hook equivalence passed on four episodes in one scene with one environment:
success, oracle_success, SPL, steps_taken and distance_to_goal are exactly identical
between hook-off and hook-on. The diff is `reports/d8/hook_equivalence.json`.
Shared replay smoke passed for the same four episodes, 25 frames, all three models:
positions, steps, candidate IDs/coordinates and counts are identical.
The initial single-scene smoke run with four environments could repeat episodes;
the corrected smoke protocol uses one environment. This was a sampling issue,
not a model change. For full S replay, use the shared episode-ID intersection
and verify each rank's coverage before comparison.
After dumps, verify shared states and O metrics, then run CPU labels and linear probes.
No GPU job was started by the termination-taxonomy script.

## Shared replay details

S uses the existing SPL/geodesic DAgger expert, not a new navigation controller.
Baseline generates expert actions; Try5/KD replay the saved action IDs and check
the current position and candidate graph before every action. Each model rebuilds
its own encoded graph from the same observation sequence. The oracle route/goal
is used for expert actions and CPU labels, not fed into policy forward.

All S runs use the common 1833 cached episodes, intersected with the original
rank assignment. O remains baseline 1839, Try5/KD 1833 plus six U metric entries.
Expert trajectories can take shortest-route shortcuts relative to the instruction
reference path: S on-GT is reported as a diagnostic, not used to filter the main
probe. S metrics are not own-policy performance.

## Local execution

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d8/run_all.sh
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/progress_probe.py
```

The first command is strictly sequential; it was started only after the existing
17600 checkpoint evaluation released GPU 4–7. No D8 model inference overlaps
another D8 inference. CUDA/conda packages were not changed.

## Serial launch wrapper

`scripts/submit/d8-dump.sh` calls `scripts/eval_d8/run_all.sh` in the foreground.
Local default: `GPUS=4,5,6,7`; explicit `GPUS` overrides this.
Under Slurm, the default GPU list comes from the allocation's
`CUDA_VISIBLE_DEVICES`, rather than assuming physical GPU indices 4–7.
The wrapper's `REPO_ROOT` can be overridden; the dump implementation must
still use paths valid on that machine. No Slurm submission is performed by
the wrapper itself.
