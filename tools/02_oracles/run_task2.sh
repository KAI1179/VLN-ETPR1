#!/usr/bin/env bash
# run_task2.sh — task book #2: O-progress oracle and contradiction-rule calibration (no model API).
# Written here, run by a human on the server; results return through git (collect.sh).
#
#   bash tools/02_oracles/run_task2.sh                         # all steps (T2.0 briefly uses one GPU, see below)
#   STEPS=T2.1,T2.2,T2.3,T2.4,T2.5 bash tools/02_oracles/run_task2.sh   # CPU only
#
# Every step is a separate script and can be rerun alone; outputs go to $OUT, logs to $LOGDIR,
# the report $OUT/task2_report.md is rebuilt after every step. Exit codes of a step:
# 0 PASS, 1 FAIL (criterion not met, outputs still written), 2 SKIP, other = ERROR.
#
# Knobs (environment): WORKDIR PY OUT LOGDIR CE_ORIG FGR2R R2R_DISC CONN RAND100 (defaults = task book),
#   STEPS, CLAUSE_TOL_M (0.5; the task book's strict rule is 0 — T2.3 reports both), OFFTRACK_M (3.0),
#   EMBODIEDSCORE_GPU_ID (T2.0 only: the GPU the simulator renders on — pick the one with most free memory),
#   T26_ROOTS (extra dirs searched by T2.6).
set -u -o pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORKDIR=${WORKDIR:-/home/xukai/code/agentic-nav}
export PY=${PY:-$WORKDIR/MIP/envs/mip/bin/python}
export OUT=${OUT:-$WORKDIR/oracles}
export LOGDIR=${LOGDIR:-$WORKDIR/logs/task2}
export CE_ORIG=${CE_ORIG:-/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed}
export FGR2R=${FGR2R:-$WORKDIR/Fine-Grained-R2R/data}
export R2R_DISC=${R2R_DISC:-$WORKDIR/Fine-Grained-R2R/source/R2R-original}
export CONN=${CONN:-/data/xukai/VLN-GOAT/datasets/R2R/connectivity}
export RAND100=${RAND100:-$WORKDIR/MIP/splits/r2r/rand100}
export CLAUSE_TOL_M=${CLAUSE_TOL_M:-0.5}
export OFFTRACK_M=${OFFTRACK_M:-3.0}
STEPS=${STEPS:-T2.0,T2.1,T2.2,T2.3,T2.4,T2.5,T2.6}
MIP_DIR=$WORKDIR/MIP
R2R_LINKS=$MIP_DIR/data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed
mkdir -p "$OUT/md" "$LOGDIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >&2; }

set_status() { # <step> <status>
  local f=$OUT/md/status.txt
  touch "$f"
  grep -v -P "^\Q$1\E\t" "$f" > "$f.tmp" || true
  printf '%s\t%s\t%s\n' "$1" "$2" "$(ts)" >> "$f.tmp"
  sort -V "$f.tmp" > "$f" && rm -f "$f.tmp"
}

run_py() { # <step> <script>
  local step=$1 script=$2 logf=$LOGDIR/$1.log rc st
  (cd "$TOOL_DIR/py" && "$PY" "$script") > "$logf" 2>&1
  rc=$?
  case $rc in 0) st=PASS ;; 1) st=FAIL ;; 2) st=SKIP ;; *) st=ERROR ;; esac
  if [ "$st" = ERROR ]; then
    { echo "$(grep -m1 '^## ' "$OUT/md/$step.md" 2>/dev/null || echo "## $step")"; echo; echo "ERROR (exit $rc), log \`$logf\`:"; echo '```'; tail -30 "$logf"; echo '```'; } > "$OUT/md/$step.md"
  fi
  set_status "$step" "$st"
  (cd "$TOOL_DIR/py" && "$PY" assemble.py)
  log "$step -> $st (log $logf)"
}

# ── T2.0: switch the R2R-CE links to the original release, rerun A7-(1) ────
step_T20() {
  local logf=$LOGDIR/T2.0.log md=$OUT/md/T2.0.md st=PASS
  : > "$logf"
  {
    echo "== before"; ls -la "$R2R_LINKS"
    [ -d "$CE_ORIG" ] || { echo "CE_ORIG missing: $CE_ORIG"; exit 3; }
    for src in "$CE_ORIG"/*; do
      name=$(basename "$src")
      [ "$name" = rand100 ] && continue
      ln -sfn "$src" "$R2R_LINKS/$name"
    done
    # links left from the _xlmr release that the original release does not have
    for l in "$R2R_LINKS"/*; do
      name=$(basename "$l")
      [ "$name" = rand100 ] && continue
      if [ -L "$l" ] && [ ! -e "$CE_ORIG/$name" ]; then echo "removing stale link $name -> $(readlink "$l")"; rm -f "$l"; fi
    done
    echo "== after"; ls -la "$R2R_LINKS"
  } >> "$logf" 2>&1 || { set_status T2.0 ERROR; log "T2.0 -> ERROR (log $logf)"; return 1; }
  local bad
  bad=$(for l in "$R2R_LINKS"/*; do n=$(basename "$l"); [ "$n" = rand100 ] && continue; case "$(readlink "$l")" in "$CE_ORIG"/*) ;; *) echo "$n";; esac; done)
  [ "$(readlink "$R2R_LINKS/rand100")" = "$MIP_DIR/splits/r2r/rand100" ] || bad="$bad rand100"
  [ -n "$bad" ] && st=FAIL

  # A7-(1) rerun: keep task 1's run aside, run fresh, compare
  local run=$MIP_DIR/outputs/fake/fake_r2r_es_cc_fable-5_default_bareES prev=$OUT/t20/task1_summary.json
  mkdir -p "$OUT/t20"
  [ -f "$prev" ] || cp "$run/summary.json" "$prev" 2>/dev/null || cp "$WORKDIR/reports/fake_r2r_es_cc_fable-5_default_bareES_summary.json" "$prev" 2>/dev/null
  [ -d "$run" ] && mv "$run" "$run.before_t20_$(date +%Y%m%d-%H%M%S)"
  # shellcheck disable=SC1091
  [ -f "$WORKDIR/egl.env" ] && . "$WORKDIR/egl.env"
  local cmd="python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0"
  (cd "$MIP_DIR" && MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet timeout 1800 "$PY" runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0) >> "$logf" 2>&1
  local rc=$? cmp_out
  cmp_out=$( [ -f "$prev" ] && [ -f "$run/summary.json" ] && (cd "$TOOL_DIR/py" && "$PY" t20_compare.py "$prev" "$run/summary.json") 2>&1 ) || st=FAIL
  [ $rc -eq 0 ] || st=FAIL
  [ -f "$run/summary.json" ] && cp "$run/summary.json" "$OUT/t20/rerun_summary.json"
  {
    echo "## T2.0 数据切换"
    echo
    echo "软链目录 \`$R2R_LINKS\`：除 \`rand100\` 外全部改指向 \`\$CE_ORIG/<split>\`（\`$CE_ORIG\`），\`_xlmr\` 独有的链接已删除。"
    echo
    echo '```'; sed -n '/== after/,$p' "$logf" | grep -E '^l' | awk '{print $(NF-2), $(NF-1), $NF}'; echo '```'
    [ -n "$bad" ] && echo "链接异常：$bad"
    echo
    echo "A7-(1) 复跑：\`$cmd\`（GPU ${EMBODIEDSCORE_GPU_ID:-0}），exit $rc。任务 #1 的 run 目录已改名保留（\`*.before_t20_*\`）。"
    echo
    echo '```'; echo "${cmp_out:-no comparison (missing summary.json)}"; echo '```'
    echo
    echo "结论：$st"
  } > "$md"
  set_status T2.0 "$st"
  (cd "$TOOL_DIR/py" && "$PY" assemble.py)
  log "T2.0 -> $st (log $logf)"
}

want() { [[ ",$STEPS," == *",$1,"* ]]; }
log "WORKDIR=$WORKDIR OUT=$OUT STEPS=$STEPS CLAUSE_TOL_M=$CLAUSE_TOL_M"
[ -x "$PY" ] || { echo "no interpreter at $PY (task 1's install_env.sh builds it)"; exit 1; }
want T2.0 && step_T20
want T2.1 && run_py T2.1 t21_instr_index.py
want T2.2 && run_py T2.2 t22_oracle.py
want T2.3 && run_py T2.3 t23_replay.py
want T2.4 && run_py T2.4 t24_budget.py
want T2.5 && run_py T2.5 t25_rules.py
want T2.6 && run_py T2.6 t26_known_failures.py
log "done. report: $OUT/task2_report.md — then: bash $TOOL_DIR/collect.sh && git push"
