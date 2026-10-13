# Paired-episode analysis (baseline vs try5 vs KD FULL)

Nothing here changes any existing job: the trainer hook only records, it is
off unless `EVAL.decision_dump True` is passed, every run writes to its own
`pa_*` directories, and existing `d7d_*` / `d8_*` results are only read.
Models and every evaluation setting are those of `scripts/eval_d7/d7d_eval.sh`
(baseline iter25000 without map; try5 no-KD iter28000 and KD FULL iter27000
with their own LLM map; control, 4 envs per rank, 150 tokens, 15 high-level
steps).  GPU jobs follow the D8 rule: GPUs 4-7, one model job at a time.

## 1. Episode features (CPU only, etpr1-uv)

```bash
python scripts/analysis/episode_features.py --split val_unseen --out data/logs/analysis/features_val_unseen.json
```

Per-episode strata from the GT path, the MP3D region boxes and connectivity
graph, and the instruction: turns, forks, room changes, decoys, spatial
words, LLM-map region hit rate.  The script hides all GPUs from itself.

## 2. Decision dumps (GPU 4-7, etpr1-py38, strictly one after another)

```bash
bash scripts/analysis/dump_eval.sh base own
bash scripts/analysis/dump_eval.sh try5 own
bash scripts/analysis/dump_eval.sh kd   own
bash scripts/analysis/dump_eval.sh base shared   # creates the shared states
bash scripts/analysis/dump_eval.sh try5 shared   # replays them, checked per step
bash scripts/analysis/dump_eval.sh kd   shared
```

* `own` (protocol 1): the policy drives.  Afterwards `scripts/eval_d8/verify.py`
  compares success / OSR / SPL / steps / NE of every episode with the existing
  D7-d run of the same model and must print `PASS` -- this proves the dump does
  not change the rollout.
* `shared` (protocol 2): the D8 mechanism.  base walks the SPL expert and saves
  each state; try5 and kd execute the same actions and abort if position or
  candidate graph ever differ.  Episode set: the 1833 D7-d non-U episodes.
  The SR printed by these runs is the expert's, not the model's.

Outputs: `data/logs/checkpoints/pa_<model>_<own|shared>_val_unseen/eval_results/decisions_val_unseen_r*_w*.jsonl`
(plus the usual stats files), and for `shared` the D8-format state files in
`data/logs/pa_shared/<model>/`.

## 3. Report (CPU)

```bash
python scripts/analysis/paired_report.py --features data/logs/analysis/features_val_unseen.json \
  --own    base=data/logs/checkpoints/pa_base_own_val_unseen/eval_results \
           try5=data/logs/checkpoints/pa_try5_own_val_unseen/eval_results \
           kd=data/logs/checkpoints/pa_kd_own_val_unseen/eval_results \
  --shared base=data/logs/checkpoints/pa_base_shared_val_unseen/eval_results \
           try5=data/logs/checkpoints/pa_try5_shared_val_unseen/eval_results \
           kd=data/logs/checkpoints/pa_kd_shared_val_unseen/eval_results \
  --out data/logs/analysis/report_val_unseen.md
```

The first label is the reference.  Sections: feature distributions, own-rollout
SR/SPL/OSR + paired 2x2, hard-subset table, discordant-episode features,
failure modes, efficiency, decision accuracy vs the expert by step category
(stop / backtrack / turn / straight) on own rollouts and on shared states with
the per-state 2x2, first divergence.  Counts and means only.

To have the report made elsewhere, copy back `features_val_unseen.json` and the
`decisions_*.jsonl` files of the six runs; nothing else is needed.
