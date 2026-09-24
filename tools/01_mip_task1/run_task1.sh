#!/usr/bin/env bash
# run_task1.sh — MIP task book #1 (environment + data id checks), executed by a human on a
# server that has (ideally) MP3D scenes, a GPU and provider keys. Claude writes the script,
# the human runs it, the results travel back through git (collect.sh).
#
# The python environment is NOT built here: run tools/01_mip_task1/install_env.sh first (or verify an
# existing one with tools/01_mip_task1/check_env.sh) and point PY at its interpreter.
#
# Usage (run inside tmux; the whole thing is resumable, every step is idempotent):
#   WORKDIR=$HOME/agentic-nav PY=$HOME/agentic-nav/MIP/envs/mip/bin/python bash tools/01_mip_task1/run_task1.sh
#   STEPS=A7,C2 RUN_PAID=1 MODEL_A7=<row> HARNESS_A7=mini bash tools/01_mip_task1/run_task1.sh
#
# Knobs (environment variables, all optional):
#   WORKDIR            root for everything this script writes (default $HOME/agentic-nav)
#   PY                 the interpreter with MIP installed (default $WORKDIR/MIP/envs/mip/bin/python)
#   STEPS              comma list of steps (A0,A1,...,C3) or "all" (default)
#   DATA_SEARCH_ROOTS  space-separated dirs searched for mp3d / vlnce / connectivity data
#   MP3D_DIR VLNCE_DIR CONN R2R_DISC   explicit data paths (skip the search)
#   RUN_PAID=1         allow A7-3 and C2 (they spend tokens); MODEL_A7 / HARNESS_A7 pick the seat
#   PROXY_URL          http proxy to try when github is unreachable (default http://127.0.0.1:37890)
#   DEEP_FIND=1        also run `find /` (slow) when the search roots turn up nothing
# Rules kept from the task book: writes only under WORKDIR (+ the repo checkout for collect.sh),
# never edits MIP sources, never downloads MP3D, retries a failing command at most 2 times,
# writes every step to $REPORT with PASS / FAIL / SKIP and moves on.

set -u -o pipefail

# ── configuration ──────────────────────────────────────────────────────────
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKS=$TOOL_DIR/checks
WORKDIR=${WORKDIR:-$HOME/agentic-nav}
REPORT=$WORKDIR/reports/task1_report.md
LOGDIR=$WORKDIR/logs
REPORTS_DIR=$WORKDIR/reports
STEPS=${STEPS:-all}
RUN_PAID=${RUN_PAID:-0}
MODEL_A7=${MODEL_A7:-}
HARNESS_A7=${HARNESS_A7:-auto}
PROXY_URL=${PROXY_URL:-http://127.0.0.1:37890}
DEEP_FIND=${DEEP_FIND:-0}
DATA_SEARCH_ROOTS=${DATA_SEARCH_ROOTS:-"$HOME/code/ETP-R1-snapshot/ETP-R1/data $HOME/run/ETP-R1/data $HOME/data /data $HOME/datasets"}
MIP_DIR=$WORKDIR/MIP
ENV_DIR=$MIP_DIR/envs/mip
FGR2R_DIR=$WORKDIR/Fine-Grained-R2R
PY=${PY:-$ENV_DIR/bin/python}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# EGL settings found by checks/egl_probe.py (via check_env.sh), if any
# shellcheck disable=SC1091
[ -f "$WORKDIR/egl.env" ] && . "$WORKDIR/egl.env"
export REPORTS_DIR

mkdir -p "$WORKDIR" "$REPORTS_DIR" "$LOGDIR"

# ── helpers ────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >&2; }
want() { [ "$STEPS" = all ] || [[ ",$STEPS," == *",$1,"* ]]; }

# retry <n> <cmd...>: up to n attempts (task book: at most 2 retries -> n=3)
retry() {
  local n=$1; shift; local i
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    log "attempt $i/$n failed: $*"
    [ "$i" -lt "$n" ] && sleep $((5 * i))
  done
  return 1
}

# excerpt <file> [n]: first and last n lines of a log
excerpt() {
  local f=$1 n=${2:-30}
  [ -s "$f" ] || { echo "(empty)"; return; }
  if [ "$(wc -l < "$f")" -le $((2 * n)) ]; then cat "$f"; else head -n "$n" "$f"; echo "... [$(wc -l < "$f") lines total] ..."; tail -n "$n" "$f"; fi
}

# record <step> <STATUS> <command text> <logfile|-> <conclusion...>
record() {
  local step=$1 status=$2 cmd=$3 logf=$4; shift 4
  {
    echo
    echo "## $step — $status"
    echo
    echo "$(ts) · host $(hostname)"
    echo
    echo '```bash'; echo "$cmd"; echo '```'
    if [ "$logf" != "-" ] && [ -f "$logf" ]; then
      echo; echo "key output (\`$logf\`):"; echo '```'; excerpt "$logf"; echo '```'
    fi
    echo; echo "conclusion: $*"
  } >> "$REPORT"
  log "$step -> $status"
}

# run_logged <logfile> <cmd...>: run, tee to log, return exit code
run_logged() {
  local logf=$1; shift
  "$@" > "$logf" 2>&1
  local rc=$?
  return $rc
}

py_ok() { [ -x "$PY" ]; }

# ── A0: environment ────────────────────────────────────────────────────────
step_A0() {
  local logf=$LOGDIR/A0.log
  {
    echo "WORKDIR=$WORKDIR"; echo "REPORT=$REPORT"; echo "LOGDIR=$LOGDIR"
    echo "hostname: $(hostname)"; echo "whoami: $(whoami)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tr '\n' ';')"
    echo "df: $(df -h "$WORKDIR" | tail -1)"
    echo "python3: $(python3 --version 2>&1)"; echo "conda: $(conda --version 2>&1 | head -1)"
    echo "uv: $(uv --version 2>&1 | head -1)"; echo "node: $(node --version 2>&1 | head -1)"
    echo "claude: $(command -v claude 2>/dev/null || echo none) $(claude --version 2>/dev/null | head -1)"
    echo "glibc: $(ldd --version | head -1)"; echo "kernel: $(uname -r)"
    echo "egl vendors: $(ls /usr/share/glvnd/egl_vendor.d/ 2>&1 | tr '\n' ' ')"
    # proxy: keep what the shell has; otherwise try the reverse tunnel
    if [ -z "${HTTPS_PROXY:-}" ] && curl -sS -m 8 -o /dev/null -x "$PROXY_URL" https://github.com 2>/dev/null; then
      export HTTP_PROXY=$PROXY_URL HTTPS_PROXY=$PROXY_URL NO_PROXY=localhost,127.0.0.1
      echo "proxy: $PROXY_URL (reachable, exported)"
    else
      echo "proxy: ${HTTPS_PROXY:-none} (shell setting kept)"
    fi
    echo "github http code: $(curl -sS -m 10 -o /dev/null -w '%{http_code}' https://github.com 2>&1)"
    echo "pypi http code: $(curl -sS -m 10 -o /dev/null -w '%{http_code}' https://pypi.org/simple/pip/ 2>&1)"
  } > "$logf" 2>&1
  if [ ! -f "$REPORT" ]; then
    { echo "# Task 1 Report"; echo; echo "generated by tools/01_mip_task1/run_task1.sh, started $(ts) on $(hostname)"; } > "$REPORT"
  fi
  record A0 INFO "env probe (see log)" "$logf" "environment recorded; proxy=${HTTPS_PROXY:-none}"
}

# ── A1: clone ──────────────────────────────────────────────────────────────
step_A1() {
  local logf=$LOGDIR/A1.log
  if [ -d "$MIP_DIR/.git" ]; then
    echo "MIP already cloned" > "$logf"
  elif ! retry 3 git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git "$MIP_DIR" >> "$logf" 2>&1; then
    record A1 FAIL "git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git" "$logf" "clone failed after 3 attempts; try the ssh remote (git@github.com:jianzhou0420/MIP.git) if ~/.ssh/config maps github.com to ssh.github.com:443"
    return 1
  fi
  (cd "$MIP_DIR" && git submodule update --init >> "$logf" 2>&1 && echo "HEAD $(git rev-parse HEAD)" && git submodule status) >> "$logf" 2>&1
  local sub_n; sub_n=$(ls "$MIP_DIR/thirdparty/EmbodiedScore-envs" 2>/dev/null | wc -l)
  if [ "$sub_n" -gt 0 ]; then
    record A1 PASS "git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git; git rev-parse HEAD; git submodule status" "$logf" "MIP $(cd "$MIP_DIR" && git rev-parse --short HEAD), submodule thirdparty/EmbodiedScore-envs non-empty ($sub_n entries). INSTALL.md prefers venv (py3.10-3.13) over conda; wheels come from GitHub releases, not PyPI"
  else
    record A1 FAIL "git submodule status" "$logf" "submodule thirdparty/EmbodiedScore-envs is empty"
    return 1
  fi
}

# ── A2: verify the python environment (built separately by install_env.sh) ──
step_A2() {
  local logf=$LOGDIR/A2.log
  [ -d "$MIP_DIR" ] || { record A2 SKIP "-" - "MIP not cloned (A1)"; return 1; }
  if ! py_ok; then
    record A2 FAIL "PY=$PY" - "no interpreter at $PY: run tools/01_mip_task1/install_env.sh, or PY=<python> for an existing env (check it first with check_env.sh)"
    return 1
  fi
  local ver; ver=$("$PY" -c "import habitat_sim; print(habitat_sim.__version__)" 2>&1)
  {
    echo "interpreter: $PY ($("$PY" --version 2>&1))"
    echo "habitat_sim: $ver"
    "$PY" -m pip list 2>/dev/null | grep -i -E "habitat|embodiedscore|litellm|hydra|omegaconf|mini-swe|claude-agent"
    ls "$LOGDIR"/A2-install.log "$LOGDIR"/ENV-check-*.log 2>/dev/null | sed 's/^/related log: /'
  } > "$logf" 2>&1
  if [ "$ver" = "0.3.3" ] && "$PY" -c "import embodiedscore_envs, litellm, hydra, minisweagent" 2>> "$logf"; then
    record A2 PASS "python -c 'import habitat_sim, embodiedscore_envs, litellm, hydra, minisweagent'" "$logf" "habitat_sim 0.3.3 and the MIP stack import from $PY (installed by install_env.sh / verified by check_env.sh, logs listed above)"
  else
    record A2 FAIL "python -c 'import habitat_sim ...'" "$logf" "habitat_sim=$ver or a MIP package missing in $PY: run install_env.sh (PY=$PY to install into this env)"
    return 1
  fi
}

# ── A3: pytest ─────────────────────────────────────────────────────────────
step_A3() {
  local logf=$LOGDIR/A3.log
  py_ok || { record A3 SKIP "-" - "no env (A2)"; return 1; }
  (cd "$MIP_DIR" && "$PY" -m pytest tests -q -rs) > "$logf" 2>&1
  local rc=$?
  local summary; summary=$(grep -E "passed|failed|error" "$logf" | tail -1)
  if [ $rc -eq 0 ]; then record A3 PASS "python -m pytest tests -q -rs" "$logf" "$summary"; else record A3 FAIL "python -m pytest tests -q -rs" "$logf" "$summary; failing tests: $(grep -E '^FAILED' "$logf" | head -5 | tr '\n' ' ')"; fi
}

# ── A4: model backend configuration (read-only) ────────────────────────────
step_A4() {
  local logf=$LOGDIR/A4.log
  [ -d "$MIP_DIR" ] || { record A4 SKIP "-" - "MIP not cloned"; return 1; }
  (cd "$MIP_DIR" && {
    echo "== config dirs"; ls -d conf config configs exp_workspace/bareES/configs 2>/dev/null
    echo "== models.yaml"; cat exp_workspace/bareES/configs/models/models.yaml
    echo "== harness/mini.yaml"; cat exp_workspace/bareES/configs/harness/mini.yaml
    echo "== api_base / keys in code"; grep -rn -i -E "litellm|DASHSCOPE|OPENAI_API_KEY|ANTHROPIC_API_KEY|OLLAMA_URL|api_base|base_url" --include=*.py --include=*.yaml --include=*.md . | grep -v -E "thirdparty|envs/" | head -80
  }) > "$logf" 2>&1
  record A4 PASS "cat exp_workspace/bareES/configs/models/models.yaml; grep -rn api_base ..." "$logf" "model rows live in exp_workspace/bareES/configs/models/models.yaml (row -> harness -> {id, api_base, params}); add a model = add a row; api_base is honoured by the mini seat only (core/harnesses/mini_swe.py), cc/codex use their CLI logins"
}

# ── A5: find data ──────────────────────────────────────────────────────────
find_first_dir() { # <name> <predicate-glob-inside> roots...
  local name=$1 inner=$2; shift 2; local r d
  for r in "$@"; do
    [ -d "$r" ] || continue
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      if [ -z "$inner" ] || compgen -G "$d/$inner" > /dev/null; then echo "$d"; return 0; fi
    done < <(find "$r" -maxdepth 6 -type d -name "$name" 2>/dev/null)
  done
  return 1
}

step_A5() {
  local logf=$LOGDIR/A5.log
  : > "$logf"
  # shellcheck disable=SC2206
  local roots=($DATA_SEARCH_ROOTS "$WORKDIR")
  MP3D_DIR=${MP3D_DIR:-$(find_first_dir mp3d "*/*.glb" "${roots[@]}")}
  VLNCE_DIR=${VLNCE_DIR:-$(find_first_dir "R2R_VLNCE_v1-3_preprocessed" "val_unseen/val_unseen.json.gz" "${roots[@]}")}
  CONN=${CONN:-$(find_first_dir connectivity "*_connectivity.json" "${roots[@]}" "$REPO_ROOT/precompute_img_features")}
  if [ -z "${R2R_DISC:-}" ]; then
    local r; for r in "${roots[@]}" "$FGR2R_DIR/source/R2R-original"; do
      [ -d "$r" ] || continue
      R2R_DISC=$(find "$r" -maxdepth 6 -name "R2R_val_unseen.json" 2>/dev/null | head -1); [ -n "$R2R_DISC" ] && break
    done
  fi
  if [ "$DEEP_FIND" = 1 ]; then
    [ -n "$MP3D_DIR" ] || MP3D_DIR=$(find / -maxdepth 8 -type d -name mp3d -not -path '/proc/*' 2>/dev/null | head -1)
    [ -n "$VLNCE_DIR" ] || VLNCE_DIR=$(find / -maxdepth 9 -type d -name "R2R_VLNCE_v1-3_preprocessed" -not -path '/proc/*' 2>/dev/null | head -1)
  fi
  {
    echo "search roots: ${roots[*]}"
    echo "MP3D_DIR=${MP3D_DIR:-<not found>}"; [ -n "$MP3D_DIR" ] && echo "  scans with glb: $(find "$MP3D_DIR" -maxdepth 2 -name '*.glb' 2>/dev/null | wc -l), navmesh: $(find "$MP3D_DIR" -maxdepth 2 -name '*.navmesh' 2>/dev/null | wc -l)"
    echo "VLNCE_DIR=${VLNCE_DIR:-<not found>}"; [ -n "$VLNCE_DIR" ] && ls "$VLNCE_DIR"
    echo "CONN=${CONN:-<not found>}"; [ -n "$CONN" ] && echo "  files: $(ls "$CONN" | grep -c _connectivity.json)"
    echo "R2R_DISC=${R2R_DISC:-<not found; the B3 FGR2R clone ships source/R2R-original/R2R_val_unseen.json>}"
    echo "rand100 (ships with MIP): $MIP_DIR/splits/r2r/rand100"; ls -la "$MIP_DIR/splits/r2r/rand100" 2>/dev/null
  } >> "$logf"
  cat > "$WORKDIR/data_paths.env" <<EOT
MP3D_DIR=${MP3D_DIR:-}
VLNCE_DIR=${VLNCE_DIR:-}
CONN=${CONN:-}
R2R_DISC=${R2R_DISC:-}
EOT
  local st=PASS; [ -n "$MP3D_DIR" ] && [ -n "$VLNCE_DIR" ] || st=PARTIAL
  record A5 "$st" "find (roots: $DATA_SEARCH_ROOTS) for mp3d / R2R_VLNCE_v1-3_preprocessed / connectivity / R2R_val_unseen.json" "$logf" "paths saved to $WORKDIR/data_paths.env; mp3d=${MP3D_DIR:-none} vlnce=${VLNCE_DIR:-none} (R2R-CE full split is public: if missing, download R2R_VLNCE_v1-3_preprocessed to $WORKDIR/data/vlnce/ and set VLNCE_DIR)"
}

load_paths() { [ -f "$WORKDIR/data_paths.env" ] && . "$WORKDIR/data_paths.env"; export CONN R2R_DISC; }

# ── A6: data directory ─────────────────────────────────────────────────────
step_A6() {
  local logf=$LOGDIR/A6.log
  [ -d "$MIP_DIR" ] || { record A6 SKIP "-" - "MIP not cloned"; return 1; }
  load_paths
  local r2r=$MIP_DIR/data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed
  mkdir -p "$MIP_DIR/data/embodiedscore/scenes" "$r2r"
  # real directory inside WORKDIR, per-split links into the shared data tree (never writes there)
  if [ -n "${VLNCE_DIR:-}" ]; then
    local s; for s in "$VLNCE_DIR"/*/; do s=${s%/}; [ "$(basename "$s")" = rand100 ] && continue; ln -sfn "$s" "$r2r/$(basename "$s")"; done
    for f in "$VLNCE_DIR"/*.json.gz; do [ -f "$f" ] && ln -sfn "$f" "$r2r/$(basename "$f")"; done
  fi
  ln -sfn "$MIP_DIR/splits/r2r/rand100" "$r2r/rand100"
  [ -n "${MP3D_DIR:-}" ] && ln -sfn "$MP3D_DIR" "$MIP_DIR/data/embodiedscore/scenes/mp3d"
  {
    ls -la "$r2r"; echo; ls -la "$r2r/rand100/"
    echo; echo "scenes/mp3d -> $(readlink "$MIP_DIR/data/embodiedscore/scenes/mp3d" 2>/dev/null || echo '<absent>')"
    echo "mp3d scan dirs: $(ls "$MIP_DIR/data/embodiedscore/scenes/mp3d" 2>/dev/null | wc -l)"
  } > "$logf" 2>&1
  local n; n=$(ls "$MIP_DIR/data/embodiedscore/scenes/mp3d" 2>/dev/null | wc -l)
  if [ "$n" -ge 90 ]; then record A6 PASS "ln -sfn ... data/embodiedscore/{datasets/vlnce,scenes/mp3d}" "$logf" "$n mp3d scans linked; rand100 linked"; else record A6 PARTIAL "ln -sfn ..." "$logf" "rand100 linked; mp3d scans: $n (need 90) — A7 will be skipped unless mp3d is present"; fi
}

# ── A7: the three checks ───────────────────────────────────────────────────
newest_run_dir() { find "$MIP_DIR/outputs" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }

a7_artifacts() { # <run_dir> >> logfile
  local d=$1 logf=$2
  {
    echo "== run dir: $d"; ls "$d"
    echo "== summary.json"; cat "$d/summary.json" 2>/dev/null || echo "<no summary.json>"
    echo "== episode_0.jsonl: $(wc -l < "$d/episode_0.jsonl" 2>/dev/null || echo 0) lines"; head -5 "$d/episode_0.jsonl" 2>/dev/null | cut -c1-400
    echo "== stats.html numbers"; grep -o -E "[0-9,]+ tokens|\\\$[0-9.]+|[0-9.]+ s" "$d/stats.html" 2>/dev/null | head -20
    echo "== env_server.log tail"; tail -15 "$d/env_server.log" 2>/dev/null
  } >> "$logf" 2>&1
  [ -f "$d/summary.json" ] && cp "$d/summary.json" "$REPORTS_DIR/$(basename "$d")_summary.json"
  [ -f "$d/stats.html" ] && cp "$d/stats.html" "$REPORTS_DIR/$(basename "$d")_stats.html"
}

run_mip() { # <logfile> <runner args...>; retries once with the EGL vendor override on EGL errors
  local logf=$1; shift
  (cd "$MIP_DIR" && timeout 7200 "$PY" runner.py "$@") > "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && grep -q -i -E "EGL|OpenGL|display" "$logf" "$(newest_run_dir)/env_server.log" 2>/dev/null; then
    if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
      echo "== EGL error: retrying with __EGL_VENDOR_LIBRARY_FILENAMES=10_nvidia.json" >> "$logf"
      (cd "$MIP_DIR" && __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json timeout 7200 "$PY" runner.py "$@") >> "$logf" 2>&1
      rc=$?
    fi
    { echo "== egl vendors:"; ls /usr/share/glvnd/egl_vendor.d/ 2>&1; nvidia-smi 2>&1 | head -15; } >> "$logf"
  fi
  return $rc
}

claude_logged_in() { command -v claude > /dev/null 2>&1 && claude auth status 2>/dev/null | grep -q '"loggedIn": true'; }

step_A7() {
  local logf=$LOGDIR/A7-1.log
  py_ok || { record A7-1 SKIP "-" - "no env (A2)"; return 1; }
  if [ "$(ls "$MIP_DIR/data/embodiedscore/scenes/mp3d" 2>/dev/null | wc -l)" -lt 1 ]; then
    record A7-1 SKIP "python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0" - "no MP3D scenes linked (A5/A6); A7-2/A7-3 not run"
    return 1
  fi
  local cmd1="python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0"
  run_mip "$logf" std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0; local rc=$?
  local d; d=$(newest_run_dir); echo "exit code: $rc" >> "$logf"; [ -n "$d" ] && a7_artifacts "$d" "$logf"
  if [ $rc -ne 0 ] || [ ! -f "$d/summary.json" ]; then
    record A7-1 FAIL "$cmd1" "$logf" "exit $rc; see env_server.log tail above (EGL / scene loading); A7-2 and A7-3 not run"
    return 1
  fi
  record A7-1 PASS "$cmd1" "$logf" "exit 0; run dir $d; summary.json copied to reports/"

  # (2) real harness on the scripted endpoint
  local h=$HARNESS_A7 m=${MODEL_A7:-fable-5}
  if [ "$h" = auto ]; then if claude_logged_in; then h=cc; else h=mini; fi; fi
  [ "$h" = cc ] && m=fable-5
  logf=$LOGDIR/A7-2.log
  local cmd2="python runner.py std_r2r_es_bareES harness=$h model=$m api=fake run.episodes=0"
  run_mip "$logf" std_r2r_es_bareES harness="$h" model="$m" api=fake run.episodes=0; rc=$?
  d=$(newest_run_dir); echo "exit code: $rc" >> "$logf"; [ -n "$d" ] && a7_artifacts "$d" "$logf"
  if [ $rc -ne 0 ] || [ ! -f "$d/summary.json" ]; then
    record A7-2 FAIL "$cmd2" "$logf" "exit $rc (harness=$h chosen: cc needs a logged-in claude CLI, mini needs no key with api=fake)"
    return 1
  fi
  record A7-2 PASS "$cmd2" "$logf" "exit 0; run dir $d"

  # (3) one real episode — costs tokens; gated
  logf=$LOGDIR/A7-3.log
  if [ "$RUN_PAID" != 1 ] || [ -z "$MODEL_A7" ]; then
    record A7-3 SKIP "python runner.py std_r2r_es_bareES harness=<cc|mini> model=<MODEL_A7> run.episodes=0" - "gated: rerun with RUN_PAID=1 MODEL_A7=<row> HARNESS_A7=<cc|mini> STEPS=A7 (A7-1/A7-2 rerun quickly)"
    return 0
  fi
  local h3=$HARNESS_A7; [ "$h3" = auto ] && h3=$h
  local cmd3="python runner.py std_r2r_es_bareES harness=$h3 model=$MODEL_A7 run.episodes=0"
  record A7-3 PLANNED "$cmd3" - "about to execute: 1 episode, real model (token cost)"
  run_mip "$logf" std_r2r_es_bareES harness="$h3" model="$MODEL_A7" run.episodes=0; rc=$?
  d=$(newest_run_dir); echo "exit code: $rc" >> "$logf"; [ -n "$d" ] && a7_artifacts "$d" "$logf"
  if [ $rc -eq 0 ] && [ -f "$d/summary.json" ]; then record A7-3 PASS "$cmd3" "$logf" "exit 0; run dir $d"; echo "$d" > "$WORKDIR/a7_3_ok"; else record A7-3 FAIL "$cmd3" "$logf" "exit $rc"; return 1; fi
}

# ── B: data id checks ──────────────────────────────────────────────────────
run_check() { # <step> <script> <cmdtext> <conclusion-on-pass>
  local step=$1 script=$2 cmd=$3 ok=$4 logf=$LOGDIR/$1.log
  py_ok || { record "$step" SKIP "$cmd" - "no env (A2)"; return 1; }
  "$PY" "$CHECKS/$script" > "$logf" 2>&1; local rc=$?
  case $rc in
    0) record "$step" PASS "$cmd" "$logf" "$ok" ;;
    2) record "$step" SKIP "$cmd" "$logf" "inputs missing (see log)" ;;
    *) record "$step" FAIL "$cmd" "$logf" "exit $rc" ;;
  esac
  return $rc
}

step_B1() {
  load_paths
  if [ -n "${VLNCE_DIR:-}" ] && [ -f "$VLNCE_DIR/val_unseen/val_unseen.json.gz" ]; then
    VLNCE_EPISODES=$VLNCE_DIR/val_unseen/val_unseen.json.gz VLNCE_GT=$VLNCE_DIR/val_unseen/val_unseen_gt.json.gz run_check B1 b1.py "VLNCE_EPISODES=\$VLNCE_DIR/val_unseen/val_unseen.json.gz python checks/b1.py" "full val_unseen inspected (expect 1839 episodes)"
  else
    VLNCE_EPISODES=$MIP_DIR/splits/r2r/rand100/rand100.json.gz VLNCE_GT=$MIP_DIR/splits/r2r/rand100/rand100_gt.json.gz run_check B1 b1.py "VLNCE_EPISODES=MIP/splits/r2r/rand100/rand100.json.gz python checks/b1.py" "full val_unseen not on this host; same schema inspected on rand100 (100 episodes) — the 1839 count stays unverified"
  fi
}
step_B2() { RAND100_DIR=$MIP_DIR/splits/r2r/rand100 run_check B2 b2.py "RAND100_DIR=MIP/splits/r2r/rand100 python checks/b2.py" "rand100 is a self-contained episode file; ids written to reports/rand100_ids.json"; }
step_B3() {
  local logf=$LOGDIR/B3-clone.log
  if [ ! -d "$FGR2R_DIR/.git" ]; then retry 3 git clone https://github.com/YicongHong/Fine-Grained-R2R.git "$FGR2R_DIR" > "$logf" 2>&1 || { record B3 FAIL "git clone https://github.com/YicongHong/Fine-Grained-R2R.git" "$logf" "clone failed"; return 1; }; fi
  FGR2R_DIR=$FGR2R_DIR run_check B3 b3.py "git clone Fine-Grained-R2R; FGR2R_DIR=... python checks/b3.py" "FGR2R fields/chunk_view structure and rand100 coverage in log"
}
step_B4() {
  load_paths
  [ -n "${R2R_DISC:-}" ] || R2R_DISC=$FGR2R_DIR/source/R2R-original/R2R_val_unseen.json
  local eps=$MIP_DIR/splits/r2r/rand100/rand100.json.gz
  [ -n "${VLNCE_DIR:-}" ] && [ -f "$VLNCE_DIR/val_unseen/val_unseen.json.gz" ] && eps=$VLNCE_DIR/val_unseen/val_unseen.json.gz
  CONN=${CONN:-$REPO_ROOT/precompute_img_features/connectivity} R2R_DISC=$R2R_DISC VLNCE_EPISODES=$eps run_check B4 b4.py "CONN=$CONN R2R_DISC=$R2R_DISC VLNCE_EPISODES=$eps python checks/b4.py" "best mapping mp3d(x,y,z)=(hab_x,-hab_z,hab_y); 3D and horizontal criteria in log (MP3D pose is the camera, ~1.4 m above the floor point)"
}

# ── C: API and cost ────────────────────────────────────────────────────────
step_C1() {
  local logf=$LOGDIR/C1.log
  {
    for v in ANTHROPIC_API_KEY OPENAI_API_KEY DASHSCOPE_API_KEY DASHSCOPE_INTL_API_KEY OPENROUTER_API_KEY OLLAMA_URL; do
      if [ -n "${!v:-}" ]; then echo "$v: set (len ${#v})"; else echo "$v: unset"; fi
    done
    grep -h -o -E "^(export )?(ANTHROPIC|OPENAI|DASHSCOPE|OPENROUTER)_API_KEY" ~/.bashrc ~/.zshrc ~/.profile 2>/dev/null | sort -u
    echo "claude: $(command -v claude || echo none) $(claude --version 2>/dev/null)"; claude auth status 2>/dev/null | head -5
    echo "codex: $(command -v codex || echo none) $(codex --version 2>/dev/null)"; echo "ollama: $(command -v ollama || echo none)"
  } > "$logf" 2>&1
  py_ok || { record C1 SKIP "-" "$logf" "no env"; return 1; }
  "$PY" "$CHECKS/c1.py" >> "$logf" 2>&1; local rc=$?
  case $rc in
    0) record C1 PASS "python checks/c1.py (text + 64x64 image probe per provider)" "$logf" "see per-provider OK/FAIL, latency, and VISION lines" ;;
    2) record C1 SKIP "python checks/c1.py" "$logf" "no provider keys in the shell; claude CLI login status recorded" ;;
    *) record C1 FAIL "python checks/c1.py" "$logf" "exit $rc" ;;
  esac
}

step_C2() {
  local logf=$LOGDIR/C2.log
  if [ "$RUN_PAID" != 1 ] || [ -z "$MODEL_A7" ] || [ ! -f "$WORKDIR/a7_3_ok" ]; then
    record C2 SKIP "python runner.py std_r2r_es_bareES harness=<h> model=<MODEL_A7> run.episodes=0-19" - "gated: needs RUN_PAID=1, MODEL_A7, and a PASS in A7-3"
    return 0
  fi
  local h=$HARNESS_A7; if [ "$h" = auto ]; then if claude_logged_in; then h=cc; else h=mini; fi; fi
  local cmd="python runner.py std_r2r_es_bareES harness=$h model=$MODEL_A7 run.episodes=0-19"
  record C2 PLANNED "$cmd" - "about to execute: 20 episodes, real model (token cost)"
  run_mip "$logf" std_r2r_es_bareES harness="$h" model="$MODEL_A7" run.episodes=0-19; local rc=$?
  local d; d=$(newest_run_dir); echo "exit code: $rc" >> "$logf"; [ -n "$d" ] && a7_artifacts "$d" "$logf"
  SUMMARY_JSON=$d/summary.json "$PY" "$CHECKS/c2_summary.py" > "$LOGDIR/C2-summary.log" 2>&1
  cat "$LOGDIR/C2-summary.log" >> "$logf"
  if [ $rc -eq 0 ]; then record C2 PASS "$cmd" "$logf" "aggregate + medians + 1839-episode extrapolation in log"; else record C2 FAIL "$cmd" "$logf" "exit $rc"; fi
}

step_C3() {
  local logf=$LOGDIR/C3.log
  {
    nvidia-smi --query-gpu=name,memory.total,count --format=csv 2>&1 | head -10
    echo "== pip"; py_ok && "$PY" -m pip list 2>/dev/null | grep -i -E "vllm|sglang|transformers|torch"; pip list 2>/dev/null | grep -i -E "vllm|sglang|transformers|torch"
    echo "== hf cache qwen"; ls ~/.cache/huggingface/hub 2>/dev/null | grep -i qwen || echo none
    echo "== ollama"; command -v ollama && ollama list 2>/dev/null | head
  } > "$logf" 2>&1
  record C3 INFO "nvidia-smi; pip list | grep -iE 'vllm|sglang|transformers'; ls ~/.cache/huggingface/hub | grep -i qwen" "$logf" "GPU and local weights recorded"
}

# ── main ───────────────────────────────────────────────────────────────────
log "WORKDIR=$WORKDIR STEPS=$STEPS RUN_PAID=$RUN_PAID"
for s in A0 A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 C1 C2 C3; do
  if want "$s"; then "step_$s" || log "$s did not pass; continuing"; fi
done
log "done. report: $REPORT — now run: bash $TOOL_DIR/collect.sh"
