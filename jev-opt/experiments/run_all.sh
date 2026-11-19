#!/usr/bin/env bash
# Run the first optimization study (GEPA; its settings were fixed before any run): 3 single-condition runs per task, then one
# top-quality test evaluation per run. Serialized on the project-wide Jev rate-limit lock.
# Already-finished runs are skipped, so the script is safe to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")/../.."
EXT=./jev-opt/extension/index.ts
LOG=runs/jev-opt/study.log
mkdir -p runs/jev-opt

configs=(
  contempo_tbdense/task.yaml
  ezstance_subtaskA/task.yaml
  smarter_implicit_hate/task.yaml
  edit_intent/task.yaml
  inli/task.yaml
  dress_new/task_content.yaml
  dress_new/task_organization.yaml
  dress_new/task_language.yaml
)

# Serialize on the project-wide Jev lock (flock cannot exec shell functions, so lock an fd).
omp_cmd() { ( flock 9; omp -p --no-session --no-extensions -e "$EXT" "$1" ) 9>/tmp/jev-run.lock; }

for cfg in "${configs[@]}"; do
  base=$(dirname "$cfg"); f=$(basename "$cfg" .yaml); name=${base}${f#task}   # e.g. dress_new_content
  for cond in zero text program; do
    case $cond in zero) condition=jev-zero;; text) condition=jev-text-opt;; program) condition=jev-program-opt;; esac
    run="study-${name//_/-}-$cond"
    dir=runs/jev-opt/$run
    if [[ ! -f $dir/test_consumed.json ]]; then
      if ! grep -q '"state": "finished"' "$dir/run.json" 2>/dev/null; then
        verb=optimize; [[ -d $dir ]] && verb=resume
        echo "$(date -Is) $verb $run" | tee -a "$LOG"
        if [[ $verb == optimize ]]; then
          omp_cmd "/jev-optimize jev-opt/experiments/$cfg --conditions $condition --run-id $run" >>"$LOG" 2>&1
        else
          omp_cmd "/jev-resume $run" >>"$LOG" 2>&1
        fi
      fi
      grep -q '"state": "finished"' "$dir/run.json" 2>/dev/null || { echo "$(date -Is) ERROR: $run did not finish; stopping" | tee -a "$LOG"; exit 1; }
      echo "$(date -Is) evaluate-final $run" | tee -a "$LOG"
      omp_cmd "/jev-evaluate-final $run top-quality --yes" >>"$LOG" 2>&1
    fi
    [[ -d $dir ]] || { echo "$(date -Is) ERROR: $run produced no run dir; stopping" | tee -a "$LOG"; exit 1; }
  done
done
echo "$(date -Is) study done" | tee -a "$LOG"
