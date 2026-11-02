#!/usr/bin/env bash
# User-run workflow: one GPU evaluation, CPU turn table, then append data dumps.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
DATA=/tmp/pa-data
RUN=pa_zero_own_val_unseen
cd "$ROOT"
export GPUS="${GPUS:-4,5,6,7}"
bash "$ROOT/scripts/analysis/dump_eval.sh" zero own
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/analysis/turn_table.py" \
    --features "$ROOT/data/logs/analysis/features_val_unseen.json" \
    --base "$ROOT/data/logs/checkpoints/$RUN/eval_results" \
    --ours "$ROOT/data/logs/checkpoints/pa_try5_own_val_unseen/eval_results" \
    --out "$ROOT/data/logs/checkpoints/$RUN/turn_table.md"

# /tmp/pa-data is the existing data-only repository, not the source worktree.
[[ -z "$(git -C "$DATA" status --porcelain)" ]]
REMOTE="$(git -C "$ROOT" remote get-url origin)"
git -C "$DATA" fetch "$REMOTE" data/pa-dumps
git -C "$DATA" merge --ff-only FETCH_HEAD
FILES=()
for source in "$ROOT/data/logs/checkpoints/$RUN/eval_results"/decisions_*.jsonl; do
    target="${RUN}__$(basename "$source").gz"
    gzip -c "$source" > "$DATA/$target"
    FILES+=("$target")
done
git -C "$DATA" add -- "${FILES[@]}"
if ! git -C "$DATA" diff --cached --quiet; then
    git -C "$DATA" -c user.name="$(git -C "$ROOT" config user.name)" \
        -c user.email="$(git -C "$ROOT" config user.email)" \
        commit -m 'Add zero-input own-rollout decision dumps'
fi
git -C "$DATA" push "$REMOTE" HEAD:refs/heads/data/pa-dumps
