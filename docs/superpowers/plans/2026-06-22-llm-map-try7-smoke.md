# LLM Map Try7 Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable smoke-test launcher for evaluating LLM4/LLM5-derived cognitive-map caches with the PriorGT Try7 checkpoint on R2R.

**Architecture:** Keep temporary experiment orchestration out of `run_r2r/main_server.bash` by adding a focused script under `scripts/tries/`. The script mirrors the main launcher’s distributed GPU setup and R2R eval overrides, but exposes the LLM cache namespace as an argument. `docs/NOTE.md` records the smoke-test matrix and defers broader RxR/fallback policy.

**Tech Stack:** Bash, Habitat VLN-CE launcher config overrides, existing `SS-ETP-LLM`/`LLMPolicy`, Markdown docs.

---

## File Structure

- Create `scripts/tries/smoke-llm-map-try7.sh`: runs R2R eval using Try7 weights and a selectable LLM-Navigation cache key.
- Modify `docs/NOTE.md`: replaces the unfinished `Nav 1` pipeline note with concrete smoke-test notes.

---

### Task 1: Add Try7 LLM-Map Smoke Launcher

**Files:**
- Create: `scripts/tries/smoke-llm-map-try7.sh`

- [ ] **Step 1: Create the script**

Add this complete file:

```bash
#!/bin/bash
set -euo pipefail

export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/gpu-detection.bash"

CACHE_ALIAS="${1:-llm5}"
MASTER_PORT="${2:-2333}"

case "${CACHE_ALIAS}" in
      llm4)
      CACHE_MODEL_KEY="llm4"
      ;;
      llm5)
      CACHE_MODEL_KEY="llm5"
      ;;
      *)
      CACHE_MODEL_KEY="${CACHE_ALIAS}"
      ;;
esac

configure_distributed_gpu_vars

EXP_CONFIG="run_r2r/iter_train.yaml"
MAP_NUM_ENVS=4
GT_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/prior_gt/store2/try7_step_435000.pt"
GT_DAGGER_CKPT="data/logs/checkpoints/release_r2r_priorgt_dagger/store/try7.iter29600.pth"
EXP_NAME="smoke_r2r_llm_map_try7_${CACHE_MODEL_KEY}"

COMMON_ARGS="--exp-config ${EXP_CONFIG}
      SIMULATOR_GPU_IDS ${GPU_IDS}
      TORCH_GPU_IDS ${GPU_IDS}
      GPU_NUMBERS ${GPU_NUMBERS}
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True"

LLM_TRY7_ARGS="TRAINER_NAME SS-ETP-LLM
      MODEL.policy_name LLMPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.MAP_ENCODER.llm_cache_model_key ${CACHE_MODEL_KEY}
      MODEL.pretrained_path ${GT_PRETRAINED_CKPT}"

echo "###### LLM map + Try7 smoke eval ######"
echo "cache_model_key=${CACHE_MODEL_KEY}"
echo "pretrained_ckpt=${GT_PRETRAINED_CKPT}"
echo "dagger_ckpt=${GT_DAGGER_CKPT}"
echo "exp_name=${EXP_NAME}"
echo "master_port=${MASTER_PORT}"

cd "${REPO_ROOT}"
python -m torch.distributed.launch \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port "${MASTER_PORT}" \
      run.py \
      --exp_name "${EXP_NAME}" \
      --run-type eval \
      ${COMMON_ARGS} \
      NUM_ENVIRONMENTS ${MAP_NUM_ENVS} \
      ${LLM_TRY7_ARGS} \
      EVAL.CKPT_PATH_DIR "${GT_DAGGER_CKPT}" \
      IL.back_algo control
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/tries/smoke-llm-map-try7.sh
```

Expected: command exits with status 0.

- [ ] **Step 3: Validate shell syntax**

Run:

```bash
bash -n scripts/tries/smoke-llm-map-try7.sh
```

Expected: command exits with status 0 and prints nothing.

- [ ] **Step 4: Commit launcher**

Run:

```bash
git add scripts/tries/smoke-llm-map-try7.sh
git commit -m "chore: add llm try7 smoke launcher"
```

Expected: commit succeeds and includes only the launcher file.

---

### Task 2: Document Smoke-Test Matrix

**Files:**
- Modify: `docs/NOTE.md`

- [ ] **Step 1: Update the LLM pipeline section**

Replace the current unfinished pipeline block:

```markdown
### 基于 LLM 的 pipeline

- Nav 1
    - 只使用 VLNCE 数据预训练
    - LLM4 的微调模型
```

with:

```markdown
### 基于 LLM 的 pipeline

- Smoke: LLM-derived cognitive-map cache + Try 7 navigation checkpoint
    - Script: `scripts/tries/smoke-llm-map-try7.sh`
    - Matrix: LLM4 cache, LLM5 cache
    - Checkpoint: `data/logs/checkpoints/release_r2r_priorgt_dagger/store/try7.iter29600.pth`
    - R2R evaluation only for the first smoke pass
    - Missing `.npz` cache entries use current LLM-Navigation skip behavior
    - RxR English-only filtering and fallback/zero-map policies are deferred
```

Keep the existing LLM4 note under the LLM4 bullet:

```markdown
    - -> 生成的预训练 navigation cache 仅 8621/109507
```

- [ ] **Step 2: Review the note diff**

Run:

```bash
git diff -- docs/NOTE.md
```

Expected: diff only updates the LLM pipeline section and preserves the existing LLM4 cache coverage note.

- [ ] **Step 3: Commit docs**

Run:

```bash
git add docs/NOTE.md
git commit -m "docs: note llm try7 smoke matrix"
```

Expected: commit succeeds and includes only `docs/NOTE.md`.

---

### Task 3: Final Verification

**Files:**
- Verify: `scripts/tries/smoke-llm-map-try7.sh`
- Verify: `docs/NOTE.md`

- [ ] **Step 1: Run shell syntax check**

Run:

```bash
bash -n scripts/tries/smoke-llm-map-try7.sh
```

Expected: command exits with status 0 and prints nothing.

- [ ] **Step 2: Run Python static checks**

Run:

```bash
ruff check
```

Expected: command exits with status 0. If unrelated existing lint issues appear, record them and do not fix unrelated code.

- [ ] **Step 3: Run type checks**

Run:

```bash
ty check
```

Expected: command exits with status 0. If unrelated existing type issues appear, record them and do not fix unrelated code.

- [ ] **Step 4: Inspect final worktree**

Run:

```bash
git status --short
```

Expected: no unexpected unstaged files. `docs/NOTE.md` should no longer be dirty after its docs commit.
