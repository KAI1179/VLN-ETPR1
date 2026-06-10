detect_gpu_count() {
  local visible count

  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    visible="${CUDA_VISIBLE_DEVICES// /}"
    if [ -n "${visible}" ] && [ "${visible}" != "-1" ] && [ "${visible}" != "NoDevFiles" ]; then
      IFS=',' read -ra _visible_gpu_ids <<< "${visible}"
      count=0
      for _gpu_id in "${_visible_gpu_ids[@]}"; do
        if [ -n "${_gpu_id}" ]; then
          count=$((count + 1))
        fi
      done
      if [ "${count}" -gt 0 ]; then
        echo "${count}"
        return 0
      fi
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [ -n "${count}" ] && [ "${count}" -gt 0 ] 2>/dev/null; then
      echo "${count}"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
try:
    import torch
    count = torch.cuda.device_count()
except Exception:
    count = 0
print(count if count > 0 else 1)
PY
    return 0
  fi

  echo 1
}

make_local_gpu_ids() {
  local count="$1"
  local ids="["
  local i

  for ((i = 0; i < count; i++)); do
    ids+="${i}"
    if ((i < count - 1)); then
      ids+=","
    fi
  done
  ids+="]"
  echo "${ids}"
}

configure_distributed_gpu_vars() {
  local detected_gpu_count
  detected_gpu_count="$(detect_gpu_count)"

  NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_NUMBERS:-${detected_gpu_count}}}"
  GPU_NUMBERS="${GPU_NUMBERS:-${NPROC_PER_NODE}}"
  GPU_IDS="${GPU_IDS:-$(make_local_gpu_ids "${GPU_NUMBERS}")}"

  export NPROC_PER_NODE
  export GPU_NUMBERS
  export GPU_IDS
}

configure_pretrain_gpu_vars() {
  NUM_GPUS="${NUM_GPUS:-$(detect_gpu_count)}"
  export NUM_GPUS
}
