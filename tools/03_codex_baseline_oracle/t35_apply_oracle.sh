#!/usr/bin/env bash
# t35_apply_oracle.sh — T3.5 on the server: build the oracleES arm on MIP branch e0-oracle, dry-run every oracle
# condition with the scripted agent (+run.fake=true) and with the real codex loop against the fake endpoint (api=fake),
# check the [ORACLE] lines / spatial image are in the tool results, prove oracle=none == bareES, run the GT unit test.
# Called by run_task3.sh (env: WORKDIR PY OUT OUT3 LOGDIR MIP_DIR RAND100 CODEX_MODEL). Exit 0 PASS / 1 FAIL / 2 SKIP.
set -u -o pipefail
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
md=$OUT3/md/T3.5.md; st=PASS; notes=()
say() { notes+=("$*"); echo "$*"; }
cd "$MIP_DIR" || exit 3
[ -f "$WORKDIR/egl.env" ] && . "$WORKDIR/egl.env"   # shellcheck disable=SC1091
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# ── branch e0-oracle (never touches bareES) ──
base=$(git rev-parse --short HEAD)
if git rev-parse --verify -q e0-oracle > /dev/null; then git checkout -q e0-oracle; else git checkout -q -B e0-oracle; fi
say "MIP base commit $base, branch $(git rev-parse --abbrev-ref HEAD)"
params=""; [ -f "$OUT3/rules_v0.json" ] && params="--params $OUT3/rules_v0.json"
# shellcheck disable=SC2086
"$PY" "$TOOL_DIR/py/t35_make_arm.py" --mip "$MIP_DIR" --out3 "$OUT3" --oracle-json "$OUT/oracle_progress_rand100.json" --rand100 "$RAND100/rand100.json.gz" $params || { echo "## T3.5"; echo; echo "arm build failed"; } > "$md"
[ -d exp_workspace/oracleES ] || exit 1
git add exp_workspace/oracleES && git commit -q -m "oracleES arm: GT oracle hints in tool results (task 3, T3.5)" 2>/dev/null || true
say "e0-oracle commit $(git rev-parse --short HEAD); bareES untouched: $(git diff --quiet "$base" -- exp_workspace/bareES && echo yes || echo NO)"

# ── GT unit test (no simulator) ──
if "$PY" "$TOOL_DIR/py/t35_unittest.py" exp_workspace/oracleES/mcp "$OUT3/oracle_by_index_rand100.json" "$RAND100/rand100_gt.json.gz" "$RAND100/rand100.json.gz" > "$OUT3/t35_unittest.txt" 2>&1; then
  say "unit test: $(head -1 "$OUT3/t35_unittest.txt")"
else say "unit test FAILED: $(head -3 "$OUT3/t35_unittest.txt" | tr '\n' ' ')"; st=FAIL; fi

# ── dry-runs ──
check_run() { # <label> <run glob> -> prints jsonl/actions.log evidence; returns 1 when nothing found
  local label=$1 glob=$2 d n_lines n_img
  d=$(ls -td $glob 2>/dev/null | head -1)
  [ -n "$d" ] && [ -f "$d/summary.json" ] || { say "$label: no run dir for $glob"; return 1; }
  n_lines=$(cat "$d"/episode_0.jsonl "$d"/live_0/actions.log 2>/dev/null | grep -c '\[ORACLE\]')
  n_img=$(ls "$d"/live_0/ 2>/dev/null | grep -c 'obs_')
  say "$label: $d — [ORACLE] lines in jsonl+actions.log: $n_lines; obs frames: $n_img; poses.jsonl lines: $(wc -l < "$d/live_0/poses.jsonl" 2>/dev/null || echo 0)"
  echo "$d"
}
declare -A FAKE CODEX
for cond in none progress spatial offtrack commit all; do
  export BAREES_ORACLE=$cond
  logf=$LOGDIR/T3.5_fake_$cond.log
  "$PY" runner.py std_r2r_es_oracle harness=codex model="$CODEX_MODEL" oracle=$cond +run.fake=true run.episodes=0 > "$logf" 2>&1; rc=$?
  FAKE[$cond]=$(check_run "fake $cond (exit $rc)" "outputs/fake/*oracle-$cond*" | tail -1)
  [ $rc -eq 0 ] || st=FAIL
done
if command -v codex > /dev/null 2>&1; then
  for cond in none all; do
    export BAREES_ORACLE=$cond
    logf=$LOGDIR/T3.5_codexfake_$cond.log
    "$PY" runner.py std_r2r_es_oracle harness=codex model="$CODEX_MODEL" oracle=$cond api=fake run.episodes=0 > "$logf" 2>&1; rc=$?
    CODEX[$cond]=$(check_run "codex+api=fake $cond (exit $rc)" "outputs/codex/*oracle-$cond*" | tail -1)
    [ $rc -eq 0 ] || say "codex api=fake $cond exit $rc (see $logf)"
  done
else say "codex CLI not installed: codex+api=fake dry-runs skipped (T3.1 installs it; rerun STEPS=T3.5 after)"; fi
unset BAREES_ORACLE

# ── none == bareES: same scripted walk on bareES vs oracleES/none, compare summaries and the tool-result texts ──
logf=$LOGDIR/T3.5_bare.log
"$PY" runner.py std_r2r_es_bareES harness=codex model="$CODEX_MODEL" +run.fake=true run.episodes=0 > "$logf" 2>&1
bare=$(ls -td outputs/fake/*bareES* 2>/dev/null | head -1)
"$PY" - "$bare" "${FAKE[none]}" <<'PYEOF' > "$OUT3/t35_none_vs_bare.txt" 2>&1
import json, sys
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
sa, sb = json.load(open(a / "summary.json")), json.load(open(b / "summary.json"))
ea, eb = sa["episodes"][0], sb["episodes"][0]
same_metrics = ea["metrics"] == eb["metrics"] and ea["agent"].get("tool_calls") == eb["agent"].get("tool_calls")
def results(p):
    out = []
    for ln in (p / "episode_0.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("kind") == "tool_result":
            r = {k: v for k, v in r.items() if k != "t"}  # wall-clock stamp differs run to run
            out.append(json.dumps(r, sort_keys=True))
    return out
ra, rb = results(a), results(b)
print("metrics+tool_calls identical:", same_metrics)
print("tool_result records identical:", ra == rb, f"({len(ra)} vs {len(rb)})")
if ra != rb:
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x != y:
            print("first diff at record", i); print(" bare  :", x[:300]); print(" oracle:", y[:300]); break
PYEOF
say "none vs bareES: $(tr '\n' ' ' < "$OUT3/t35_none_vs_bare.txt")"
grep -q "identical: True" "$OUT3/t35_none_vs_bare.txt" || st=FAIL

{
  echo "## T3.5 oracle 注入实现与 dry-run"; echo
  echo "- 实现：新 arm 目录 \`exp_workspace/oracleES\`（bareES 的副本 + \`mcp/oracle_inject.py\`、\`mcp/oracles.py\`，bridge.py 只加 5 处钩子，prompts.py 加 briefing 句）；配置 \`exp_workspace/oracleES/configs/std_r2r_es_oracle.yaml\`，座位 \`oracle=\`；索引表 \`$OUT3/oracle_by_index_rand100.json\`。"
  echo "- 注入点：step() 结果 JSON 末尾字段 \`ORACLE\`（进度 / 偏离 / 分支各一行），observe() 帧后附轨迹俯视图（spatial）与同样的行；每个原语的真实位姿写 \`live_i/poses.jsonl\`。"
  echo; for n in "${notes[@]}"; do echo "- $n"; done
  echo; echo "证据片段（fake all）："; echo '```'
  d=${FAKE[all]}; [ -n "$d" ] && grep -h '\[ORACLE\]' "$d"/episode_0.jsonl "$d"/live_0/actions.log 2>/dev/null | head -4 | cut -c1-400
  echo '```'
  echo; echo "结论：$st"
} > "$md"
[ "$st" = PASS ] && exit 0 || exit 1
