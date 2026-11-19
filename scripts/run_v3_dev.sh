#!/usr/bin/env bash
# v3 development runs (train/dev/validation samples only; never test). Each task holds the
# shared Jev rate-limit lock for its own run only, so other lock users interleave.
#   scripts/run_v3_dev.sh dev_v3/contempo_matres dev_v3/smarter_hatexplain ...
set -euo pipefail
cd "$(dirname "$0")/.."
for task in "$@"; do
  echo "[$(date -Is)] waiting for lock: $task"
  ( flock 9; echo "[$(date -Is)] start $task"; uv run gauntlet run "$task" > "/tmp/v3dev_${task//\//_}.json" ) 9>/tmp/jev-run.lock
  echo "[$(date -Is)] done $task"
done
