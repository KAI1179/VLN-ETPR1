#!/usr/bin/env bash
# install_env.sh — create (or fill) the python environment for MIP, then verify it.
#   bash tools/01_mip_task1/install_env.sh                       # new env at $WORKDIR/MIP/envs/mip (conda 3.11 > venv > uv)
#   PY=/path/to/python bash tools/01_mip_task1/install_env.sh    # install into an EXISTING interpreter instead
# Writes only under WORKDIR (plus the interpreter you point PY at). Retries pip at most twice.
set -u -o pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$HOME/agentic-nav}
MIP_DIR=${MIP_DIR:-$WORKDIR/MIP}
ENV_DIR=${ENV_DIR:-$MIP_DIR/envs/mip}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
LOGDIR=$WORKDIR/logs
mkdir -p "$LOGDIR"
logf=$LOGDIR/A2-install.log
retry() { local n=$1; shift; local i; for ((i = 1; i <= n; i++)); do "$@" && return 0; echo "attempt $i/$n failed: $*"; [ "$i" -lt "$n" ] && sleep $((5 * i)); done; return 1; }

if [ ! -f "$MIP_DIR/requirements.txt" ]; then
  git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git "$MIP_DIR" || exit 1
fi
(cd "$MIP_DIR" && git submodule update --init) || exit 1

if [ -z "${PY:-}" ]; then
  if [ -x "$ENV_DIR/bin/python" ]; then
    echo "reusing $ENV_DIR"
  elif command -v conda > /dev/null 2>&1; then
    retry 3 conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION" || exit 1
  elif python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
    python3 -m venv "$ENV_DIR" && "$ENV_DIR/bin/python" -m pip install -q --upgrade pip setuptools wheel || exit 1
  elif command -v uv > /dev/null 2>&1; then
    uv venv --seed --python "$PYTHON_VERSION" "$ENV_DIR" || exit 1
  else
    echo "no conda, no python3 in 3.10-3.13, no uv: cannot create an environment"; exit 1
  fi
  PY=$ENV_DIR/bin/python
fi
echo "installing into: $PY ($("$PY" --version 2>&1))" | tee "$logf"
(cd "$MIP_DIR" && retry 3 "$PY" -m pip install -r requirements.txt) 2>&1 | tee -a "$logf" | grep -E -v "^\s+(━|Downloading|Using cached)" | tail -40
if ! "$PY" -c "import habitat_sim; assert habitat_sim.__version__ == '0.3.3', habitat_sim.__version__" 2>> "$logf"; then
  echo "habitat_sim 0.3.3 not importable after install; see $logf (glibc: $(ldd --version | head -1); kernel: $(uname -r))"; exit 1
fi
echo "install ok; verifying with check_env.sh"
PY=$PY WORKDIR=$WORKDIR MIP_DIR=$MIP_DIR bash "$TOOL_DIR/check_env.sh"
echo
echo "to use this interpreter for the task: PY=$PY WORKDIR=$WORKDIR bash $TOOL_DIR/run_task1.sh"
