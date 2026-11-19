#!/usr/bin/env bash
# jev-tailored dev study (Jev-adapted GEPA): VALIDATION ONLY. This script never calls /jev-evaluate-final.
#   phase main   : jev-tailored from the paper seed and the best jev-native seed (task_tailored*.yaml)
#                                                                              -> tailored-<task>
#   phase ablate : ablations from the paper seed only, no rulebook (task_tailored_noT1*.yaml) and GEPA's
#                  default acceptance (task_tailored_noT2*.yaml); run only after main, if budget allows
#                                                                              -> tailored-<task>-noT1, -noT2
# Usage: run_tailored_dev.sh [main|ablate] (default main). Finished runs are skipped; interrupted ones resumed.
# Every run holds the project-wide Jev lock (/tmp/jev-run.lock), so it queues behind the v2/v3 runs.
set -uo pipefail
cd "$(dirname "$0")/../.."
EXT=$PWD/jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/tailored-dev.log
mkdir -p "$ROOT"

# experiment dir : run-name stem : task-file suffix
tasks=(
  contempo_tbdense:contempo-tbdense:
  ezstance_subtaskA:ezstance-subtaskA:
  smarter_implicit_hate:smarter-implicit-hate:
  edit_intent:edit-intent:
  inli:inli:
  dress_new:dress-new-content:_content
  dress_new:dress-new-organization:_organization
  dress_new:dress-new-language:_language
)

omp_cmd() { ( flock 9; omp -p --no-session --no-extensions -e "$EXT" "$1" </dev/null ) 9>/tmp/jev-run.lock; }
finished() { grep -q '"state": "finished"' "$ROOT/$1/run.json" 2>/dev/null; }

# run <run-id> <task.yaml>
run() {
  local id=$1 cfg=$2
  finished "$id" && return 0
  if [[ -d $ROOT/$id ]]; then
    echo "$(date -Is) resume $id" | tee -a "$LOG"
    omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
  else
    echo "$(date -Is) optimize $id ($cfg; jev-tailored)" | tee -a "$LOG"
    omp_cmd "/jev-optimize $cfg --conditions jev-tailored --run-id $id" >>"$LOG" 2>&1
  fi
  finished "$id" || { echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; return 1; }
}

phase=${1:-main}
status=0
for spec in "${tasks[@]}"; do
  IFS=: read -r dir stem suffix <<<"$spec"
  if [[ $phase == main ]]; then
    run "tailored-$stem" "jev-opt/experiments/$dir/task_tailored$suffix.yaml" || status=1
  elif [[ $phase == ablate ]]; then
    for ab in noT1 noT2; do
      run "tailored-$stem-$ab" "jev-opt/experiments/$dir/task_tailored_$ab$suffix.yaml" || status=1
    done
  else
    echo "unknown phase $phase" >&2
    exit 2
  fi
done
echo "$(date -Is) tailored dev phase $phase done (status $status)" | tee -a "$LOG"
exit $status
