#!/usr/bin/env bash
# collect.sh — copy the task-1 results from WORKDIR into this checkout and commit them, so
# they travel back to Claude through git. Push is explicit: AUTO_PUSH=1 or do it by hand.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR=${WORKDIR:-$HOME/agentic-nav}
TAG=${TAG:-$(hostname)_$(date +%Y%m%d-%H%M)}
DEST=$REPO_ROOT/docs/reports/mip_task1/server/$TAG
mkdir -p "$DEST/logs"
cp "$WORKDIR/reports/task1_report.md" "$DEST/"
cp "$WORKDIR/reports/"*.json "$DEST/" 2>/dev/null || true
cp "$WORKDIR/reports/"*.html "$DEST/" 2>/dev/null || true
cp "$WORKDIR/data_paths.env" "$DEST/" 2>/dev/null || true
for f in "$WORKDIR"/logs/*.log; do
  # keep logs small: first/last 200 lines
  if [ "$(wc -l < "$f")" -gt 400 ]; then { head -200 "$f"; echo "... [truncated, $(wc -l < "$f") lines] ..."; tail -200 "$f"; } > "$DEST/logs/$(basename "$f")"; else cp "$f" "$DEST/logs/"; fi
done
# the env server log of the newest MIP run, useful when A7 failed
newest=$(find "$WORKDIR/MIP/outputs" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "${newest:-}" ] && [ -f "$newest/env_server.log" ] && tail -200 "$newest/env_server.log" > "$DEST/logs/last_env_server.log"
cd "$REPO_ROOT"
git add "docs/reports/mip_task1/server/$TAG"
git commit -q -m "docs(reports): MIP task 1 server run $TAG" || echo "nothing to commit"
if [ "${AUTO_PUSH:-0}" = 1 ]; then git push -u origin "$(git rev-parse --abbrev-ref HEAD)"; else echo "now: git push -u origin $(git rev-parse --abbrev-ref HEAD)"; fi
echo "collected into $DEST"
