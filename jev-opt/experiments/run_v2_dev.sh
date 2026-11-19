#!/usr/bin/env bash
# v2 dev study (the second optimization study): VALIDATION ONLY. This script never calls /jev-evaluate-final.
#   phase ab    : A = seed + class calibration (task_v2*_nodemo.yaml, jev-zero)
#                 B = seed + kNN demos (k=8, class-balanced) + calibration (task_v2*.yaml, jev-zero)
#   phase cd    : one run per task with C = jev-text-opt and D = jev-program-opt (task_v2*.yaml);
#                 E = its ensemble-3 / ensemble-5 representatives (top candidates of C and D)
#   phase seeds : optimizer seeds 1 and 2 for whichever of C/D has the higher val primary metric
# Usage: run_v2_dev.sh [ab|cd|seeds|all]. Finished runs are skipped; interrupted ones resumed.
set -uo pipefail
cd "$(dirname "$0")/../.."
EXT=./jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/v2-dev.log
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

# Serialize every Jev-calling command on the project-wide Jev lock.
omp_cmd() { ( flock 9; omp -p --no-session --no-extensions -e "$EXT" "$1" ) 9>/tmp/jev-run.lock; }
finished() { grep -q '"state": "finished"' "$ROOT/$1/run.json" 2>/dev/null; }

# run <run-id> <task.yaml> <conditions> [seed]
run() {
  local id=$1 cfg=$2 conds=$3 seed=${4:-}
  finished "$id" && return 0
  if [[ -d $ROOT/$id ]]; then
    echo "$(date -Is) resume $id" | tee -a "$LOG"
    omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
  else
    echo "$(date -Is) optimize $id ($cfg; $conds${seed:+; seed $seed})" | tee -a "$LOG"
    omp_cmd "/jev-optimize $cfg --conditions $conds --run-id $id${seed:+ --seed $seed}" >>"$LOG" 2>&1
  fi
  finished "$id" || { echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; return 1; }
}

# best_condition <cd-run-id>: jev-text-opt or jev-program-opt, by val primary (ties: text-opt, the simpler)
best_condition() {
  python3 - "$ROOT/$1/pareto.json" <<'EOF'
import json, sys
reps = json.load(open(sys.argv[1]))["representatives"]
t, p = (reps[f"top-jev-{c}-opt"]["val"]["primary"] for c in ("text", "program"))
print("jev-program-opt" if p > t else "jev-text-opt")
EOF
}

phase=${1:-all}
status=0
for spec in "${tasks[@]}"; do
  IFS=: read -r dir stem suffix <<<"$spec"
  cfg=jev-opt/experiments/$dir/task_v2$suffix.yaml
  nodemo=jev-opt/experiments/$dir/task_v2${suffix}_nodemo.yaml
  if [[ $phase == ab || $phase == all ]]; then
    run "v2-$stem-A" "$nodemo" jev-zero || status=1
    run "v2-$stem-B" "$cfg" jev-zero || status=1
  fi
done
for spec in "${tasks[@]}"; do
  IFS=: read -r dir stem suffix <<<"$spec"
  cfg=jev-opt/experiments/$dir/task_v2$suffix.yaml
  if [[ $phase == cd || $phase == all ]]; then
    run "v2-$stem-CD" "$cfg" jev-text-opt,jev-program-opt || status=1
  fi
done
for spec in "${tasks[@]}"; do
  IFS=: read -r dir stem suffix <<<"$spec"
  cfg=jev-opt/experiments/$dir/task_v2$suffix.yaml
  if [[ $phase == seeds || $phase == all ]]; then
    finished "v2-$stem-CD" || { echo "$(date -Is) skip seeds for $stem: CD run not finished" | tee -a "$LOG"; status=1; continue; }
    cond=$(best_condition "v2-$stem-CD")
    tag=${cond#jev-}; tag=${tag%-opt}   # text | program
    for seed in 1 2; do
      run "v2-$stem-$tag-s$seed" "$cfg" "$cond" "$seed" || status=1
    done
  fi
done
echo "$(date -Is) v2 dev phase $phase done (status $status)" | tee -a "$LOG"
exit $status
