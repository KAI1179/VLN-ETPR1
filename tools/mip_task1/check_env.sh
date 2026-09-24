#!/usr/bin/env bash
# check_env.sh — can an EXISTING python environment run MIP? Read-only apart from the MIP clone.
#   conda activate <env>; bash tools/mip_task1/check_env.sh            # checks `python` on PATH
#   PY=/path/to/python bash tools/mip_task1/check_env.sh               # checks that interpreter
# Also records GPU / EGL facts. Exit code = checks/check_env.py's (0 as-is, 3 after install, 4 no, 5 unknown).
set -u -o pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR=${WORKDIR:-$HOME/agentic-nav}
MIP_DIR=${MIP_DIR:-$WORKDIR/MIP}
LOGDIR=$WORKDIR/logs
PY=${PY:-$(command -v python || command -v python3)}
mkdir -p "$LOGDIR"
logf=$LOGDIR/ENV-check-$(date +%Y%m%d-%H%M%S).log
if [ ! -f "$MIP_DIR/requirements.txt" ]; then
  echo "MIP not cloned yet -> cloning into $MIP_DIR (the same thing step A1 does)"
  git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git "$MIP_DIR" || { echo "clone failed"; exit 5; }
fi
{
  echo "== host"; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
  echo "egl vendors: $(ls /usr/share/glvnd/egl_vendor.d/ 2>&1 | tr '\n' ' ')"
  echo "== interpreter under test: $PY"
  [ -f "$WORKDIR/egl.env" ] && { echo "sourcing $WORKDIR/egl.env"; . "$WORKDIR/egl.env"; }
  MIP_DIR=$MIP_DIR MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet "$PY" "$REPO_ROOT/tools/mip_task1/checks/check_env.py" 2>&1 | tee "$LOGDIR/.check_env.last"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 4 ] && [ "$rc" -ne 5 ] && grep -q "render probe (EGL context, empty scene): FAIL" "$LOGDIR/.check_env.last"; then
    echo; echo "== render probe failed: running the EGL diagnosis (checks/egl_probe.py) — this tries several env combinations"
    WORKDIR=$WORKDIR "$PY" "$REPO_ROOT/tools/mip_task1/checks/egl_probe.py"
  fi
  echo "exit code: $rc"
} 2>&1 | tee "$logf"
rc=$(grep "^exit code:" "$logf" | tail -1 | sed 's/exit code: //')
echo "log: $logf"
exit "${rc:-5}"
