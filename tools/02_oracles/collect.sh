#!/usr/bin/env bash
# collect.sh — copy task 2's outputs into this checkout and commit them (push by hand or AUTO_PUSH=1).
# JSON outputs larger than 5 MB are gzipped; a gzip larger than 20 MB is left out and listed with its sha256.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR=${WORKDIR:-/home/xukai/code/agentic-nav}
OUT=${OUT:-$WORKDIR/oracles}
LOGDIR=${LOGDIR:-$WORKDIR/logs/task2}
TAG=${TAG:-$(hostname)_$(date +%Y%m%d-%H%M)}
DEST=$REPO_ROOT/docs/reports/02_oracles/server/$TAG
mkdir -p "$DEST/logs" "$DEST/md"
cp "$OUT/task2_report.md" "$DEST/"
cp "$OUT"/md/* "$DEST/md/" 2>/dev/null || true
: > "$DEST/MANIFEST.txt"
for f in "$OUT"/*.json "$OUT"/t20/*.json; do
  [ -f "$f" ] || continue
  size=$(stat -c %s "$f"); sum=$(sha256sum "$f" | cut -d' ' -f1); rel=${f#"$OUT"/}
  if [ "$size" -le 5000000 ]; then
    mkdir -p "$DEST/$(dirname "$rel")"; cp "$f" "$DEST/$rel"; echo "$rel $size $sum" >> "$DEST/MANIFEST.txt"
  else
    gzip -c -9 "$f" > "$DEST/$rel.gz"
    if [ "$(stat -c %s "$DEST/$rel.gz")" -gt 20000000 ]; then rm -f "$DEST/$rel.gz"; echo "$rel $size $sum NOT-COLLECTED(too large)" >> "$DEST/MANIFEST.txt"; else echo "$rel $size $sum gzipped" >> "$DEST/MANIFEST.txt"; fi
  fi
done
for f in "$LOGDIR"/*.log; do
  [ -f "$f" ] || continue
  if [ "$(wc -l < "$f")" -gt 400 ]; then { head -200 "$f"; echo "... [truncated, $(wc -l < "$f") lines] ..."; tail -200 "$f"; } > "$DEST/logs/$(basename "$f")"; else cp "$f" "$DEST/logs/"; fi
done
cd "$REPO_ROOT"
git add "docs/reports/02_oracles/server/$TAG"
git commit -q -m "docs(reports): task 2 oracle run $TAG" || { echo "commit failed or nothing to commit (git user.name/user.email?)"; exit 1; }
if [ "${AUTO_PUSH:-0}" = 1 ]; then git push -u origin "$(git rev-parse --abbrev-ref HEAD)"; else echo "now: git push -u origin $(git rev-parse --abbrev-ref HEAD)"; fi
echo "collected into $DEST"
