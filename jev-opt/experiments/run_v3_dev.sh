#!/usr/bin/env bash
# v3 dev study (the third optimization study): VALIDATION ONLY. This script never calls /jev-evaluate-final.
#   phase seeds : 3 jev-native + 3 decomposed seed programs per task written by the omp model from the
#             task definition only (/jev-seeds; no Jev call, no lock)                 -> <task>/seeds_v3*/
#   phase a : A = seed + readout head (jev-zero, task_v3*.yaml)                      -> v3-<task>-A
#   phase p : P = jev-population-opt (task_v3*.yaml) from the paper seed + the generated seeds;
#             representatives top-single (P), top-stacked (P+head), top-zero-/head-/search-<kind>
#                                                                                     -> v3-<task>-P
#   phase n : jev-native-zero = the jev-native seed with the best zero-shot TRAIN score in v3-<task>-P,
#             run exactly as written (jev-zero, no fitted number; task_v3_native*.yaml) -> v3-<task>-N
# Usage: run_v3_dev.sh [seeds|a|p|n|all] (all = a, p, n; seeds must be generated first).
# Finished runs are skipped; interrupted ones resumed.
set -uo pipefail
cd "$(dirname "$0")/../.."
EXT=$PWD/jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/v3-dev.log
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
omp_free() { omp -p --no-session --no-extensions -e "$EXT" "$1"; }  # optimizer-LLM only, no Jev call
finished() { grep -q '"state": "finished"' "$ROOT/$1/run.json" 2>/dev/null; }

# run <run-id> <task.yaml> <conditions>
run() {
  local id=$1 cfg=$2 conds=$3
  finished "$id" && return 0
  if [[ -d $ROOT/$id ]]; then
    echo "$(date -Is) resume $id" | tee -a "$LOG"
    omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
  else
    echo "$(date -Is) optimize $id ($cfg; $conds)" | tee -a "$LOG"
    omp_cmd "/jev-optimize $cfg --conditions $conds --run-id $id" >>"$LOG" 2>&1
  fi
  finished "$id" || { echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; return 1; }
}

phase=${1:-all}
status=0
if [[ $phase == seeds ]]; then
  for spec in "${tasks[@]}"; do
    IFS=: read -r dir stem suffix <<<"$spec"
    out=jev-opt/experiments/$dir/seeds_v3$suffix
    [[ -f $out/generation.json ]] && continue
    echo "$(date -Is) seeds $stem -> $out" | tee -a "$LOG"
    omp_free "/jev-seeds jev-opt/experiments/$dir/task_v3$suffix.yaml --out $out --n 3" >>"$LOG" 2>&1
    [[ -f $out/generation.json ]] || { echo "$(date -Is) ERROR: no seeds for $stem" | tee -a "$LOG"; status=1; }
  done
  exit $status
fi
for want in a p n; do
  [[ $phase == "$want" || $phase == all ]] || continue
  for spec in "${tasks[@]}"; do
    IFS=: read -r dir stem suffix <<<"$spec"
    cfg=jev-opt/experiments/$dir/task_v3$suffix.yaml
    if [[ $want == a ]]; then
      run "v3-$stem-A" "$cfg" jev-zero || status=1
    elif [[ $want == p ]]; then
      run "v3-$stem-P" "$cfg" jev-population-opt || status=1
    else
      finished "v3-$stem-P" || { echo "$(date -Is) skip v3-$stem-N: P run not finished" | tee -a "$LOG"; status=1; continue; }
      native=jev-opt/experiments/$dir/task_v3_native$suffix.yaml
      if [[ ! -f $native ]]; then
        uv run --project jev-opt/python python jev-opt/experiments/native_task_v3.py "$ROOT/v3-$stem-P/run.json" "$cfg" "$native" "seeds_v3$suffix" >>"$LOG" 2>&1 || { status=1; continue; }
      fi
      run "v3-$stem-N" "$native" jev-zero || status=1
    fi
  done
done
echo "$(date -Is) v3 dev phase $phase done (status $status)" | tee -a "$LOG"
exit $status
