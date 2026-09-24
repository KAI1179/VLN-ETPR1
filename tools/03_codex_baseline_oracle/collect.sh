#!/usr/bin/env bash
# collect.sh — copy task 3's outputs into this checkout and commit them (push by hand or AUTO_PUSH=1).
# Copies the report, per-step fragments, rules/oracle JSON, run archives (summary.json, stats.html, RUN_META, episode jsonl
# up to 5 MB each), exported trajectories, and logs (head/tail 200 lines when long).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR=${WORKDIR:-/home/xukai/code/agentic-nav}
OUT3=${OUT3:-$WORKDIR/task3}
LOGDIR=${LOGDIR:-$WORKDIR/logs/task3}
TAG=${TAG:-$(hostname)_$(date +%Y%m%d-%H%M)}
DEST=$REPO_ROOT/docs/reports/03_codex_baseline_oracle/server/$TAG
mkdir -p "$DEST/logs" "$DEST/md" "$DEST/runs" "$DEST/traj"
cp "$OUT3/task3_report.md" "$DEST/" 2>/dev/null || echo "no task3_report.md yet"
cp "$OUT3"/md/* "$DEST/md/" 2>/dev/null || true
cp "$OUT3"/*.json "$OUT3"/*.md "$DEST/" 2>/dev/null || true
: > "$DEST/MANIFEST.txt"
for d in "$OUT3"/runs/*/; do
  [ -d "$d" ] || continue; name=$(basename "$d"); mkdir -p "$DEST/runs/$name"
  for f in "$d"/*; do
    [ -f "$f" ] || continue; size=$(stat -c %s "$f"); rel=runs/$name/$(basename "$f")
    if [ "$size" -le 5000000 ]; then cp "$f" "$DEST/$rel"; echo "$rel $size" >> "$DEST/MANIFEST.txt"
    else gzip -c -9 "$f" > "$DEST/$rel.gz"; echo "$rel $size gzipped" >> "$DEST/MANIFEST.txt"; fi
  done
done
cp "$OUT3"/traj/*.json "$DEST/traj/" 2>/dev/null || true
for f in "$LOGDIR"/*.log; do
  [ -f "$f" ] || continue
  if [ "$(wc -l < "$f")" -gt 400 ]; then { head -200 "$f"; echo "... [truncated, $(wc -l < "$f") lines] ..."; tail -200 "$f"; } > "$DEST/logs/$(basename "$f")"; else cp "$f" "$DEST/logs/"; fi
done
cd "$REPO_ROOT"
git add "docs/reports/03_codex_baseline_oracle/server/$TAG"
git commit -q -m "docs(reports): task 3 codex/oracle run $TAG" || { echo "commit failed or nothing to commit"; exit 1; }
if [ "${AUTO_PUSH:-0}" = 1 ]; then git push -u origin "$(git rev-parse --abbrev-ref HEAD)"; else echo "now: git push -u origin $(git rev-parse --abbrev-ref HEAD)"; fi
echo "collected into $DEST"
