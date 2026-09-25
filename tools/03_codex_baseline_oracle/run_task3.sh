#!/usr/bin/env bash
# run_task3.sh — task book #3 (v1): codex baseline, oracle injection, contradiction-rule true-positive checks.
# Written here, run by a human on the server; results return through git (collect.sh).
#
#   bash tools/03_codex_baseline_oracle/run_task3.sh                      # every step, in the task-book order
#   STEPS=T3.0,T3.6,T3.7b bash tools/03_codex_baseline_oracle/run_task3.sh # zero-cost steps only (no codex, no GPU)
#
# Order (task book v1): T3.0 -> T3.4 -> T3.5 -> T3.1 -> T3.2 -> T3.3 -> T3.6 -> T3.7a -> T3.7b -> T3.7c
# T3.4 is delivered in-repo (docs/reports/03_codex_baseline_oracle/mip_code_map.md); the server step only records the
# MIP commit it was written against. T3.2 / T3.3 spend codex subscription quota and are gated by RUN_PAID=1.
# Exit codes of a python step: 0 PASS, 1 FAIL (criterion not met, outputs still written), 2 SKIP, other = ERROR.
#
# Knobs (environment): WORKDIR PY OUT OUT3 LOGDIR CE_ORIG FGR2R R2R_DISC CONN RAND100 (defaults = task book),
#   STEPS, RUN_PAID (0), CODEX_MODEL (gpt-5.5), CODEX_VERSION (pin; empty = keep what is installed),
#   EMBODIEDSCORE_GPU_ID (simulator GPU: pick the one with most free memory), HTTPS_PROXY / ALL_PROXY (codex needs both),
#   T33_EPISODES (0-19), T33_SMOKE_EPISODES (0-2).
set -u -o pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORKDIR=${WORKDIR:-/home/xukai/code/agentic-nav}
export PY=${PY:-$WORKDIR/MIP/envs/mip/bin/python}
export OUT=${OUT:-$WORKDIR/oracles}          # task 2 outputs (inputs here)
export OUT3=${OUT3:-$WORKDIR/task3}
export LOGDIR=${LOGDIR:-$WORKDIR/logs/task3}
export CE_ORIG=${CE_ORIG:-/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed}
export FGR2R=${FGR2R:-$WORKDIR/Fine-Grained-R2R/data}
export R2R_DISC=${R2R_DISC:-$WORKDIR/Fine-Grained-R2R/source/R2R-original}
export CONN=${CONN:-/data/xukai/VLN-GOAT/datasets/R2R/connectivity}
export RAND100=${RAND100:-$WORKDIR/MIP/splits/r2r/rand100}
export MIP_DIR=${MIP_DIR:-$WORKDIR/MIP}
export CODEX_MODEL=${CODEX_MODEL:-gpt-5.5}
export CODEX_VERSION=${CODEX_VERSION:-}
export RUN_PAID=${RUN_PAID:-0}
export T33_EPISODES=${T33_EPISODES:-0-19}
export T33_SMOKE_EPISODES=${T33_SMOKE_EPISODES:-0-2}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:37890}
export ALL_PROXY=${ALL_PROXY:-$HTTPS_PROXY}
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}
STEPS=${STEPS:-T3.0,T3.4,T3.5,T3.1,T3.2,T3.3,T3.6,T3.7a,T3.7b,T3.7c}
mkdir -p "$OUT3/md" "$OUT3/runs" "$OUT3/traj" "$LOGDIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >&2; }
set_status() { # <step> <status>
  local f=$OUT3/md/status.txt; touch "$f"
  grep -v -P "^\Q$1\E\t" "$f" > "$f.tmp" || true
  printf '%s\t%s\t%s\n' "$1" "$2" "$(ts)" >> "$f.tmp"; sort -V "$f.tmp" > "$f" && rm -f "$f.tmp"
}
assemble() { (cd "$TOOL_DIR/py" && "$PY" assemble.py); }
finish() { # <step> <status> <log>
  set_status "$1" "$2"; assemble; log "$1 -> $2 (log $3)"
}
run_py() { # <step> <script> [args...]
  local step=$1 script=$2 logf=$LOGDIR/$1.log rc st; shift 2
  (cd "$TOOL_DIR/py" && "$PY" "$script" "$@") > "$logf" 2>&1; rc=$?
  case $rc in 0) st=PASS ;; 1) st=FAIL ;; 2) st=SKIP ;; *) st=ERROR ;; esac
  grep -q "^Traceback" "$logf" && st=ERROR
  if [ "$st" = ERROR ]; then
    { echo "## $step"; echo; echo "ERROR (exit $rc), log \`$logf\`:"; echo '```'; tail -30 "$logf"; echo '```'; } > "$OUT3/md/$step.md"
  fi
  finish "$step" "$st" "$logf"
}
mip_run() { # <logfile> <runner args...>   (runs from MIP_DIR with the EGL fix; returns runner's exit code)
  local logf=$1; shift
  # shellcheck disable=SC1091
  [ -f "$WORKDIR/egl.env" ] && . "$WORKDIR/egl.env"
  (cd "$MIP_DIR" && MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet "$PY" runner.py "$@") >> "$logf" 2>&1
}
latest_run() { # <outputs subdir glob>  -> newest run dir with a summary.json
  ls -td "$MIP_DIR"/outputs/$1 2>/dev/null | while read -r d; do [ -f "$d/summary.json" ] && { echo "$d"; break; }; done
}
archive_run() { # <run dir> <name>  copies summary.json / stats.html / episode jsonl / run meta into $OUT3/runs/<name>
  local d=$1 name=$2 dest=$OUT3/runs/$2
  [ -d "$d" ] || return 1
  mkdir -p "$dest"; cp "$d"/summary.json "$dest/" 2>/dev/null; cp "$d"/stats.html "$dest/" 2>/dev/null
  cp "$d"/episode_*.jsonl "$dest/" 2>/dev/null; cp "$d"/*.yaml "$d"/*.json "$dest/" 2>/dev/null
  for l in "$d"/live_*/; do [ -d "$l" ] || continue; mkdir -p "$dest/$(basename "$l")"; cp "$l"/poses.jsonl "$l"/actions.log "$dest/$(basename "$l")/" 2>/dev/null; done
  { echo "run_dir $d"; echo "mip_commit $(cd "$MIP_DIR" && git rev-parse HEAD)"; echo "mip_branch $(cd "$MIP_DIR" && git rev-parse --abbrev-ref HEAD)";
    echo "codex_version $(codex --version 2>/dev/null | head -1)"; echo "model $CODEX_MODEL"; echo "date $(ts)"; } > "$dest/RUN_META.txt"
}
ratelimit_grep() { grep -i -c -E "rate.?limit|usage.?limit|too many requests|429" "$1" 2>/dev/null || echo 0; }

# ── T3.0: task-2 fixes (zero cost) ──────────────────────────────────────────
step_T30() { run_py T3.0 t30_fixes.py; }

# ── T3.4: code map is in-repo; record the MIP commit it describes ──────────
step_T34() {
  local md=$OUT3/md/T3.4.md st=PASS
  local head; head=$(cd "$MIP_DIR" && git rev-parse HEAD)
  { echo "## T3.4 代码地图"; echo; echo "已在仓库交付：\`docs/reports/03_codex_baseline_oracle/mip_code_map.md\`（针对 MIP commit 1da7f1d0）。"
    echo; echo "服务器上的 MIP：commit \`$head\`，分支 \`$(cd "$MIP_DIR" && git rev-parse --abbrev-ref HEAD)\`。"
    [ "${head:0:8}" = "1da7f1d0" ] || { echo; echo "注意：服务器 MIP 与代码地图的 commit 不同。"; st=FAIL; }
  } > "$md"
  finish T3.4 $st "-"
}

# ── T3.5: oracle configs + injector onto MIP fork branch e0-oracle, dry-runs ─
step_T35() {
  local logf=$LOGDIR/T3.5.log; : > "$logf"
  [ -f "$TOOL_DIR/t35_apply_oracle.sh" ] || { { echo "## T3.5"; echo; echo "t35_apply_oracle.sh 尚未提供（等待代码地图后补充）。"; } > "$OUT3/md/T3.5.md"; finish T3.5 SKIP "$logf"; return; }
  bash "$TOOL_DIR/t35_apply_oracle.sh" >> "$logf" 2>&1; local rc=$?
  case $rc in 0) finish T3.5 PASS "$logf" ;; 1) finish T3.5 FAIL "$logf" ;; 2) finish T3.5 SKIP "$logf" ;; *) finish T3.5 ERROR "$logf" ;; esac
}

# ── T3.1: codex CLI version pin, login check, codex + fake endpoint ─────────
step_T31() {
  local logf=$LOGDIR/T3.1.log md=$OUT3/md/T3.1.md st=PASS ver login_ok=0 fake_rc=-
  : > "$logf"
  if ! command -v codex > /dev/null 2>&1; then
    { echo "## T3.1 codex CLI"; echo; echo "未找到 \`codex\`。安装：\`npm install -g @openai/codex${CODEX_VERSION:+@$CODEX_VERSION}\`（node v24 已在 PATH），然后重跑本步。"; } > "$md"
    finish T3.1 FAIL "$logf"; return
  fi
  ver=$(codex --version 2>&1 | head -1); echo "installed: $ver" >> "$logf"
  if [ -n "$CODEX_VERSION" ] && ! echo "$ver" | grep -q "$CODEX_VERSION"; then
    echo "pinning to $CODEX_VERSION" >> "$logf"
    npm install -g "@openai/codex@$CODEX_VERSION" >> "$logf" 2>&1 || st=FAIL
    ver=$(codex --version 2>&1 | head -1)
  fi
  # login state: `codex login status` exits 0 when logged in (codex >= 0.20); fall back to a one-shot exec probe
  if codex login status >> "$logf" 2>&1; then login_ok=1; else
    if timeout 120 codex exec --skip-git-repo-check -s read-only "reply with the single word ok" >> "$logf" 2>&1; then login_ok=1; fi
  fi
  if [ $login_ok -eq 0 ]; then
    { echo "## T3.1 codex CLI"; echo; echo "版本：\`$ver\`。**未登录**。请在服务器上执行 \`codex login\`（无浏览器时试 \`codex login --device-auth\`），按提示在本机浏览器完成授权，然后 \`STEPS=T3.1,T3.2,T3.3 RUN_PAID=1 bash $0\`。不设置 OPENAI_API_KEY。"; } > "$md"
    finish T3.1 FAIL "$logf"; return
  fi
  # codex + fake endpoint (real CLI loop, scripted model): no quota
  local run
  mip_run "$logf" std_r2r_es_bareES harness=codex model="$CODEX_MODEL" api=fake run.episodes=0; fake_rc=$?
  run=$(latest_run "codex/fakeapi_r2r_es_codex_${CODEX_MODEL}_*"); [ -n "$run" ] && archive_run "$run" t31_fakeapi
  [ "$fake_rc" -eq 0 ] || st=FAIL
  {
    echo "## T3.1 codex CLI 安装与登录"; echo
    echo "- 版本：\`$ver\`（钉死：${CODEX_VERSION:-未指定，保持现状；下次请通过 CODEX_VERSION 固定}）；npm 全局包不自动更新，但请勿手动 \`npm update\`。"
    echo "- 登录：已登录（订阅）。未设置 OPENAI_API_KEY：$([ -z "${OPENAI_API_KEY:-}" ] && echo 确认 || echo '**已设置，请 unset**')"
    echo "- codex + api=fake：\`python runner.py std_r2r_es_bareES harness=codex model=$CODEX_MODEL api=fake run.episodes=0\` exit $fake_rc；run：\`${run:-not found}\`"
    [ -n "$run" ] && [ -f "$run/summary.json" ] && { echo; echo '```'; "$PY" - "$run/summary.json" <<'PYEOF'
import json,sys; s=json.load(open(sys.argv[1])); eps=s.get("episodes") or []
print({k:v for k,v in s.items() if k!="episodes"}); print("episode0:", {k:eps[0].get(k) for k in ("episode_id","success","distance_to_goal","steps","tool_calls","end_reason")} if eps else None)
PYEOF
    echo '```'; }
    echo; echo "结论：$st"
  } > "$md"
  finish T3.1 $st "$logf"
}

# ── T3.2: one real episode ──────────────────────────────────────────────────
step_T32() {
  local logf=$LOGDIR/T3.2.log md=$OUT3/md/T3.2.md st=PASS rc run
  [ "$RUN_PAID" = 1 ] || { { echo "## T3.2 单集"; echo; echo "SKIP：需要 RUN_PAID=1（消耗 codex 额度）。命令：\`python runner.py std_r2r_es_bareES harness=codex model=$CODEX_MODEL run.episodes=0\`"; } > "$md"; finish T3.2 SKIP "$logf"; return; }
  : > "$logf"
  echo "cmd: python runner.py std_r2r_es_bareES harness=codex model=$CODEX_MODEL run.episodes=0 (GPU ${EMBODIEDSCORE_GPU_ID:-0})" >> "$logf"
  mip_run "$logf" std_r2r_es_bareES harness=codex model="$CODEX_MODEL" run.episodes=0; rc=$?
  run=$(latest_run "codex/*_r2r_es_codex_${CODEX_MODEL}_*bareES*"); [ -n "$run" ] && archive_run "$run" t32_single
  [ $rc -eq 0 ] && [ -n "$run" ] || st=FAIL
  (cd "$TOOL_DIR/py" && "$PY" t3x_run_report.py T3.2 "$OUT3/runs/t32_single" "$logf" "$rc") > "$md" 2>> "$logf" || st=FAIL
  finish T3.2 $st "$logf"
}

# ── T3.3: 20-episode baseline + 3-episode oracle smoke, trajectories exported ─
step_T33() {
  local logf=$LOGDIR/T3.3.log md=$OUT3/md/T3.3.md st=PASS rc1 rc2=- run
  [ "$RUN_PAID" = 1 ] || { { echo "## T3.3 20 集基线 + 3 集冒烟"; echo; echo "SKIP：需要 RUN_PAID=1。命令见任务书。"; } > "$md"; finish T3.3 SKIP "$logf"; return; }
  : > "$logf"
  # baseline on the oracleES arm with oracle=none: byte-identical tool outputs to bareES (T3.5 proves it) and poses logged
  local cfg=std_r2r_es_oracle; [ -f "$MIP_DIR/exp_workspace/oracleES/configs/std_r2r_es_oracle.yaml" ] || { cfg=std_r2r_es_bareES; echo "oracleES missing: baseline on bareES (no poses)" >> "$logf"; }
  echo "cmd1: BAREES_ORACLE=none python runner.py $cfg harness=codex model=$CODEX_MODEL oracle=none run.episodes=$T33_EPISODES" >> "$logf"
  date '+start %F %T' >> "$logf"
  if [ "$cfg" = std_r2r_es_oracle ]; then BAREES_ORACLE=none mip_run "$logf" $cfg harness=codex model="$CODEX_MODEL" oracle=none run.episodes="$T33_EPISODES"; else mip_run "$logf" $cfg harness=codex model="$CODEX_MODEL" run.episodes="$T33_EPISODES"; fi; rc1=$?
  date '+end %F %T' >> "$logf"
  run=$(latest_run "codex/*_r2r_es_codex_${CODEX_MODEL}_*"); [ -n "$run" ] && archive_run "$run" t33_baseline
  [ $rc1 -eq 0 ] && [ -n "$run" ] || st=FAIL
  if [ -f "$MIP_DIR/exp_workspace/oracleES/configs/std_r2r_es_oracle.yaml" ]; then
    echo "cmd2: python runner.py std_r2r_es_oracle harness=codex model=$CODEX_MODEL oracle=all run.episodes=$T33_SMOKE_EPISODES" >> "$logf"
    BAREES_ORACLE=all mip_run "$logf" std_r2r_es_oracle harness=codex model="$CODEX_MODEL" oracle=all run.episodes="$T33_SMOKE_EPISODES"; rc2=$?
    run=$(latest_run "codex/*_r2r_es_codex_${CODEX_MODEL}_*oracle-all*"); [ -n "$run" ] && archive_run "$run" t33_smoke_all
  else
    echo "smoke skipped: std_r2r_es_oracle.yaml missing (T3.5 not applied)" >> "$logf"
  fi
  # trajectories for T3.7a (needs the pose fields T3.5's injector logs; falls back to whatever the jsonl carries)
  (cd "$TOOL_DIR/py" && "$PY" t33_export_traj.py "$OUT3/runs/t33_baseline" "$OUT3/traj") >> "$logf" 2>&1 || echo "traj export failed" >> "$logf"
  (cd "$TOOL_DIR/py" && "$PY" t3x_run_report.py T3.3 "$OUT3/runs/t33_baseline" "$logf" "$rc1" "$OUT3/runs/t33_smoke_all" "$rc2") > "$md" 2>> "$logf" || st=FAIL
  finish T3.3 $st "$logf"
}

# ── T3.6 / T3.7 ─────────────────────────────────────────────────────────────
step_T33S() { # smoke only (3 episodes, oracle=all), after T3.5 has been applied
  local logf=$LOGDIR/T3.3s.log md=$OUT3/md/T3.3s.md rc run
  [ "$RUN_PAID" = 1 ] || { { echo "## T3.3s 冒烟"; echo; echo "SKIP：需要 RUN_PAID=1。"; } > "$md"; finish T3.3s SKIP "$logf"; return; }
  : > "$logf"
  BAREES_ORACLE=all mip_run "$logf" std_r2r_es_oracle harness=codex model="$CODEX_MODEL" oracle=all run.episodes="$T33_SMOKE_EPISODES"; rc=$?
  run=$(latest_run "codex/*_r2r_es_codex_${CODEX_MODEL}_*oracle-all*"); [ -n "$run" ] && archive_run "$run" t33_smoke_all
  (cd "$TOOL_DIR/py" && "$PY" t3x_run_report.py T3.3s "$OUT3/runs/t33_smoke_all" "$logf" "$rc" "$OUT3/runs/t33_smoke_all" "$rc") > "$md" 2>> "$logf"
  [ $rc -eq 0 ] && finish T3.3s PASS "$logf" || finish T3.3s FAIL "$logf"
}
step_T33R() { # re-run the baseline episodes that died on a provider-side error (capacity / stream), filling the same run
  local logf=$LOGDIR/T3.3r.log md=$OUT3/md/T3.3r.md rc idx run name
  idx=$(cd "$TOOL_DIR/py" && "$PY" t33_failed_indices.py "$OUT3/runs/t33_baseline")
  [ -n "$idx" ] || { { echo "## T3.3r 重跑"; echo; echo "没有因供应商错误中止的 episode。"; } > "$md"; finish T3.3r SKIP "$logf"; return; }
  [ "$RUN_PAID" = 1 ] || { { echo "## T3.3r 重跑"; echo; echo "SKIP：需要 RUN_PAID=1。待重跑索引：$idx"; } > "$md"; finish T3.3r SKIP "$logf"; return; }
  name=$(sed -n 's/^run_dir .*\/\([^/]*\)$/\1/p' "$OUT3/runs/t33_baseline/RUN_META.txt")
  : > "$logf"; echo "re-run indices $idx into run $name (run.resume=true keeps the other records)" >> "$logf"
  BAREES_ORACLE=none mip_run "$logf" std_r2r_es_oracle harness=codex model="$CODEX_MODEL" oracle=none run.name="$name" run.resume=true run.episodes="$idx"; rc=$?
  run=$MIP_DIR/outputs/codex/$name; [ -d "$run" ] && archive_run "$run" t33_baseline
  (cd "$TOOL_DIR/py" && "$PY" t33_export_traj.py "$OUT3/runs/t33_baseline" "$OUT3/traj") >> "$logf" 2>&1
  (cd "$TOOL_DIR/py" && "$PY" t3x_run_report.py T3.3r "$OUT3/runs/t33_baseline" "$logf" "$rc") > "$md" 2>> "$logf"
  sed -i "1s/.*/## T3.3r 供应商错误重跑（索引 $idx）后的 20 集基线/" "$md"
  [ $rc -eq 0 ] && finish T3.3r PASS "$logf" || finish T3.3r FAIL "$logf"
}
step_T36()  { run_py T3.6 t36_commit_oracle.py; }
step_T37a() { run_py T3.7a t37a_rule_roc.py "$OUT3/traj"; }
step_T37b() { run_py T3.7b t37b_synthetic_branch.py; }
step_T37c() {
  [ -f "$TOOL_DIR/py/t37c_backtrack.py" ] || { { echo "## T3.7c"; echo; echo "t37c_backtrack.py 尚未提供。"; } > "$OUT3/md/T3.7c.md"; finish T3.7c SKIP -; return; }
  # shellcheck disable=SC1091
  [ -f "$WORKDIR/egl.env" ] && . "$WORKDIR/egl.env"
  run_py T3.7c t37c_backtrack.py
}

want() { [[ ",$STEPS," == *",$1,"* ]]; }
log "WORKDIR=$WORKDIR OUT3=$OUT3 STEPS=$STEPS CODEX_MODEL=$CODEX_MODEL RUN_PAID=$RUN_PAID"
[ -x "$PY" ] || { echo "no interpreter at $PY (task 1's install_env.sh builds it)"; exit 1; }
[ -f "$OUT/oracle_progress_rand100.json" ] || { echo "task 2 outputs missing in $OUT (run tools/02_oracles first)"; exit 1; }
want T3.0  && step_T30
want T3.4  && step_T34
want T3.5  && step_T35
want T3.1  && step_T31
want T3.2  && step_T32
want T3.3  && step_T33
want T3.3s && step_T33S
want T3.3r && step_T33R
want T3.6  && step_T36
want T3.7a && step_T37a
want T3.7b && step_T37b
want T3.7c && step_T37c
log "done. report: $OUT3/task3_report.md — then: bash $TOOL_DIR/collect.sh && git push"
