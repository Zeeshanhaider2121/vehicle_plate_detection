#!/usr/bin/env bash
# Reclaim disk on the EC2 host. Safe to run anytime (including via cron) — it never
# touches the running container's image, the SQLite DB, the models, or recent output.
#
#   bash cleanup.sh                 # default: age out output media/logs older than 7 days
#   KEEP_DAYS=3 bash cleanup.sh     # more aggressive
#   KEEP_DAYS=0 bash cleanup.sh     # wipe ALL crops/media/logs (keeps the DB + models)
set -euo pipefail

KEEP_DAYS="${KEEP_DAYS:-7}"
OUT=/opt/plateflow/outputs

echo "== BEFORE =="; df -h / | tail -1; docker system df 2>/dev/null || true

echo "== Docker: stopped containers, unused images, build cache =="
docker container prune -f
# -a removes every image not used by a RUNNING container (old :latest versions etc.).
# Safe here: the box only pulls the final image from GHCR, so anything removed is re-pullable.
docker image prune -af
docker builder prune -af

echo "== App outputs: crops / media / logs older than ${KEEP_DAYS} day(s) =="
if [ -d "$OUT" ]; then
  find "$OUT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -mtime +"$KEEP_DAYS" -delete
  find "$OUT" -type f -iname '*.log' -mtime +"$KEEP_DAYS" -delete
  find "$OUT" -type d -empty -delete
fi

echo "== AFTER =="; df -h / | tail -1; docker system df 2>/dev/null || true
