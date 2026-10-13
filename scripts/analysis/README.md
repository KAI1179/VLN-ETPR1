# Paired-episode analysis (baseline vs try5 vs KD)

Three steps; only step 2 needs GPUs.

Nothing here runs unless `EVAL.decision_dump True` is passed explicitly: the
trainer hooks are inert by default, every dump writes to its own
`<RUN>_dump_<protocol>_...` directory, and no training, GRPO or existing
evaluation path is touched.  Models as in reports/d8/setup.md: baseline
ETP-R1 iter25000 (no map), try5 no-KD iter28000 with its own LLM map, KD FULL
iter27000 with its own LLM map (`MAP_VIEW=none`).

## 1. Episode features (CPU, once per split)

```bash
python scripts/analysis/episode_features.py --split val_unseen --out data/logs/analysis/features_val_unseen.json
```

GT-path / scene / instruction features per episode (forks on the MP3D
connectivity graph, room changes, decoys, spatial words, LLM-map region hit
rate).  Needs the MP3D `.house`-derived box cache (`prior`), the connectivity
json and the LLM navigation cache; `--no-llm` / `--no-categories` skip the two
heavy columns.

## 2. Decision dumps (GPU, 6 runs = 3 models x 2 protocols)

```bash
# protocol 1: policy drives            protocol 2: expert drives (same states for every model)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/analysis/dump_eval.sh baseline free   release_r2r_dagger 25000
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/analysis/dump_eval.sh baseline forced release_r2r_dagger 25000
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/analysis/dump_eval.sh try5     free   <TRY5_RUN> <ITER>
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/analysis/dump_eval.sh try5     forced <TRY5_RUN> <ITER>
CUDA_VISIBLE_DEVICES=0,1,2,3 MAP_VIEW=none bash scripts/analysis/dump_eval.sh kd free   <KD_RUN> <ITER>
CUDA_VISIBLE_DEVICES=0,1,2,3 MAP_VIEW=none bash scripts/analysis/dump_eval.sh kd forced <KD_RUN> <ITER>
```

`baseline` reads the checkpoint from `CKPT` (default
`/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth`);
`try5`/`kd` read `data/logs/checkpoints/<RUN>/ckpt.iter<ITER>.pth` (override
with `CKPT`) and the run's `eval_args.txt` when present.  Each run writes
`data/logs/checkpoints/<RUN>_dump_<free|forced>_iter<ITER>_val_unseen/eval_results/decisions_val_unseen_r*_w*.jsonl`
next to the usual `stats_*.json`.  The free protocol is a normal evaluation
plus the dump (its SR should match the run's known number); the forced
protocol's SR is meaningless (the expert drives) -- only its per-step records
are used.

## 3. Report (CPU)

```bash
python scripts/analysis/paired_report.py --features data/logs/analysis/features_val_unseen.json \
  --run baseline=data/logs/checkpoints/release_r2r_dagger_dump_free_iter25000_val_unseen/eval_results \
        try5=data/logs/checkpoints/<TRY5_RUN>_dump_free_iter<ITER>_val_unseen/eval_results \
        kd=data/logs/checkpoints/<KD_RUN>_dump_free_iter<ITER>_val_unseen/eval_results \
  --forced baseline=data/logs/checkpoints/release_r2r_dagger_dump_forced_iter25000_val_unseen/eval_results \
           try5=data/logs/checkpoints/<TRY5_RUN>_dump_forced_iter<ITER>_val_unseen/eval_results \
           kd=data/logs/checkpoints/<KD_RUN>_dump_forced_iter<ITER>_val_unseen/eval_results \
  --out data/logs/analysis/report_val_unseen.md
```

The first `--run` is the reference every other run is paired against.  The
report is markdown: overview + paired 2x2, hard-subset SR/SPL table, profile
of the discordant episodes, failure modes, efficiency on commonly solved
episodes, step-level decision accuracy (free and forced, with the per-state
2x2 for forced), and first-divergence analysis.  Thresholds for the strata
and the fork definition are options (`--help`).

To iterate on the report elsewhere, copy back only `features_val_unseen.json`
and the six `decisions_*.jsonl` files (a few MB each); nothing else is needed.
