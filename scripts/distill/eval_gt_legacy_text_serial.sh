#!/usr/bin/env bash
# Evaluate completed checkpoints of the text-map GT legacy run serially.
# All selected GPUs cooperate on ONE checkpoint at a time.
#
# List checkpoints (does not evaluate or write anything):
#   bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_gt_legacy_text_serial.sh
# Preview commands without evaluating or writing anything:
#   GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_gt_legacy_text_serial.sh --dry-run 12000 16000 19200
# Evaluate the requested checkpoints, in the given order:
#   GPUS=4,5,6,7 MASTER_PORT=29720 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_gt_legacy_text_serial.sh 12000 16000 19200
#
# GPUS: required for evaluation; CUDA_VISIBLE_DEVICES is an accepted fallback.
# NUM_ENVS=4 (per GPU), SPLIT=val_unseen (or val_seen).
# BACK_ALGO is fixed to control, matching eval_run.sh's standard evaluation.
# RUN_NAME may select another text-form, no-coordinate-fusion GT legacy run.
# Existing valid result JSONs are reused. No training checkpoint is modified.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
EVAL_RUN="$ROOT/scripts/distill/eval_run.sh"
RUN_NAME="${RUN_NAME:-dagger_nocoord_gt_legacy_text}"
RUN_DIR="$ROOT/data/logs/checkpoints/$RUN_NAME"
SPLIT="${SPLIT:-val_unseen}"
NUM_ENVS="${NUM_ENVS:-4}"
MASTER_PORT="${MASTER_PORT:-29720}"
DRY_RUN=False

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    sed -n '2,18p' "$0"
    exit 0
fi
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=True; shift; fi
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 2; }
[[ "$SPLIT" == val_unseen || "$SPLIT" == val_seen ]] || { echo 'SPLIT must be val_unseen or val_seen' >&2; exit 2; }
[[ -d "$RUN_DIR" ]] || { echo "Run not found: $RUN_DIR" >&2; exit 1; }

result_json() {
    echo "$ROOT/data/logs/checkpoints/${RUN_NAME}_eval_iter${1}_${SPLIT}/eval_results/stats_ckpt_${1}_${SPLIT}.json"
}

if [[ $# == 0 ]]; then
    "$PYTHON" -B - "$RUN_DIR" "$ROOT" "$RUN_NAME" "$SPLIT" <<'PY'
from pathlib import Path
import re
import sys
run, root, name, split = sys.argv[1:]
items = []
for path in Path(run).glob("ckpt.iter*.pth"):
    match = re.fullmatch(r"ckpt\.iter(\d+)\.pth", path.name)
    if match:
        items.append((int(match[1]), path))
print(f"Run: {run}\nAvailable checkpoints (no evaluation started):")
for iteration, path in sorted(items):
    result = Path(root) / f"data/logs/checkpoints/{name}_eval_iter{iteration}_{split}/eval_results/stats_ckpt_{iteration}_{split}.json"
    print(f"  {iteration:6d}  {'evaluated' if result.is_file() else 'not evaluated'}  {path}")
PY
    exit 0
fi

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Set GPUS explicitly, e.g. GPUS=4,5,6,7 (choose idle GPUs).' >&2; exit 2; }
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ENVS must be a positive integer' >&2; exit 2; }
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT <= 65535 )) || { echo 'Invalid MASTER_PORT' >&2; exit 2; }
[[ "${BACK_ALGO:-control}" == control ]] || { echo 'This launcher uses BACK_ALGO=control only; result paths must not mix backtracking settings.' >&2; exit 2; }
IFS=',' read -ra GPU_LIST <<< "$GPUS"
declare -A GPU_SEEN=() ITER_SEEN=() OPTIONS=()
for gpu in "${GPU_LIST[@]}"; do
    [[ -z "${GPU_SEEN[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
    GPU_SEEN[$gpu]=1
done
ITERS=("$@")
for iteration in "${ITERS[@]}"; do
    [[ "$iteration" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid iteration: $iteration" >&2; exit 2; }
    [[ -z "${ITER_SEEN[$iteration]:-}" ]] || { echo "Duplicate iteration: $iteration" >&2; exit 2; }
    ITER_SEEN[$iteration]=1
    [[ -s "$RUN_DIR/ckpt.iter${iteration}.pth" ]] || { echo "Checkpoint not found: $RUN_DIR/ckpt.iter${iteration}.pth" >&2; exit 1; }
done
[[ -f "$RUN_DIR/eval_args.txt" ]] || { echo "Missing $RUN_DIR/eval_args.txt; refusing raster/LLM defaults." >&2; exit 1; }
while read -r key value; do
    [[ -n "$key" && "$key" != \#* ]] || continue
    OPTIONS[$key]="$value"
done < "$RUN_DIR/eval_args.txt"
for pair in \
    'TRAINER_NAME=SS-ETP-PriorGT' \
    'MODEL.policy_name=PriorGTTry5Policy' \
    'MODEL.MAP_ENCODER.input_form=text' \
    'MODEL.MAP_ENCODER.source=prior_gt' \
    'MODEL.MAP_ENCODER.cache_namespace=gt.legacy.r1p5.direction5.v1' \
    'MODEL.MAP_ENCODER.coordinate_fusion=False' \
    'MODEL.MAP_ENCODER.load_pretrained_map_modules=False' \
    'MODEL.elevation_axis=y'; do
    key="${pair%%=*}"; expected="${pair#*=}"
    [[ "${OPTIONS[$key]:-}" == "$expected" ]] || { echo "Unexpected $key in eval_args.txt; expected $expected" >&2; exit 1; }
done
PRETRAINED="${OPTIONS[MODEL.pretrained_path]:-}"
[[ -f "$PRETRAINED" ]] || { echo "Missing pretrained checkpoint: $PRETRAINED" >&2; exit 1; }
# The training process can still be writing its latest checkpoint. Read only
# the ZIP directory, not multi-GB tensors, and reject an unfinished archive.
"$PYTHON" -B - "$RUN_DIR" "${ITERS[@]}" <<'PY'
from pathlib import Path
import sys
import zipfile
for iteration in sys.argv[2:]:
    path = Path(sys.argv[1]) / f"ckpt.iter{iteration}.pth"
    if not zipfile.is_zipfile(path):
        raise SystemExit(f"Checkpoint is incomplete or not a torch ZIP archive; wait for saving to finish: {path}")
PY
echo "run=$RUN_NAME split=$SPLIT GPUs=$GPUS (${#GPU_LIST[@]} processes x $NUM_ENVS envs) back_algo=control"
echo "Model settings: $RUN_DIR/eval_args.txt"
cat "$RUN_DIR/eval_args.txt"

if [[ "$DRY_RUN" == True ]]; then
    for iteration in "${ITERS[@]}"; do
        printf 'env CUDA_VISIBLE_DEVICES=%q MASTER_PORT=%q NUM_ENVS=%q BACK_ALGO=control SPLIT=%q bash %q %q %q %q\n' \
            "$GPUS" "$MASTER_PORT" "$NUM_ENVS" "$SPLIT" "$EVAL_RUN" "$RUN_NAME" "$iteration" "$PRETRAINED"
    done
    echo 'Dry run only: no evaluation, no output files, no GPU processes.'
    exit 0
fi

# Prevent two copies of this driver from evaluating the same run concurrently.
exec 9>"$RUN_DIR/.eval_text_serial.lock"
flock -n 9 || { echo "Another serial evaluator holds $RUN_DIR/.eval_text_serial.lock" >&2; exit 1; }
STAMP="$(date +%Y%m%dT%H%M%S)"
SUMMARY="$RUN_DIR/eval_text_serial_${SPLIT}_${STAMP}.md"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
export PYTHONDONTWRITEBYTECODE=1

failed=()
for iteration in "${ITERS[@]}"; do
    if [[ -f "$(result_json "$iteration")" ]]; then
        echo "[iter $iteration] Existing result, reusing: $(result_json "$iteration")"
        continue
    fi
    echo "[iter $iteration] Starting; waits until completion before the next checkpoint."
    if ! CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$MASTER_PORT" NUM_ENVS="$NUM_ENVS" \
        BACK_ALGO=control SPLIT="$SPLIT" \
        bash "$EVAL_RUN" "$RUN_NAME" "$iteration" "$PRETRAINED"; then
        failed+=("$iteration")
        echo "[iter $iteration] FAILED; see $RUN_DIR/eval_iter${iteration}_${SPLIT}.log" >&2
    fi
done

# Reuse the existing result reader/Markdown formatter, restricted to the
# requested iterations. Missing/corrupt results cannot be reported as success.
if ! "$PYTHON" -B - "$ROOT" "$RUN_NAME" "$SPLIT" "$SUMMARY" "${ITERS[@]}" <<'PY'
import json
import math
from pathlib import Path
import sys
root, run, split, output = sys.argv[1:5]
sys.path.insert(0, str(Path(root) / "scripts/distill"))
from summarize_eval import table
results, failures, paths = {}, [], []
for raw in sys.argv[5:]:
    iteration = int(raw)
    path = Path(root) / f"data/logs/checkpoints/{run}_eval_iter{iteration}_{split}/eval_results/stats_ckpt_{iteration}_{split}.json"
    try:
        metric = json.loads(path.read_text())
        assert all(math.isfinite(metric[k]) for k in ("success", "spl", "oracle_success", "distance_to_goal"))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        failures.append(f"iter {iteration}: {error}")
        continue
    results[iteration] = metric
    paths.append(f"- iter {iteration}: {path}")
text = f"# {run}: {split}, text GT legacy, back_algo=control\n\n"
text += table(results) if results else "No completed results."
text += "\n\n" + "\n".join(paths)
if failures:
    text += "\n\nMissing or invalid results:\n" + "\n".join(f"- {x}" for x in failures)
Path(output).write_text(text + "\n")
print("\n" + text + "\n\nSummary: " + output)
raise SystemExit(bool(failures))
PY
then
    exit 1
fi
[[ ${#failed[@]} == 0 ]] || { echo "Evaluation commands failed for iterations: ${failed[*]}" >&2; exit 1; }
