#!/usr/bin/env bash
# Seed variance of the headline configuration (pre-registered): the v3 population search + readout
# (run_v3_dev.sh phase p) rerun with optimizer seeds 1 and 2.
#   phase val  : /jev-optimize task_v3*.yaml --conditions jev-population-opt --seed <s>  -> v3-<task>-P-s<s>
#                VALIDATION ONLY; never calls /jev-evaluate-final.
#   phase test : /jev-evaluate-final v3-<task>-P-s<s> <representative> --yes, once per run, for the
#                representative chosen by the pre-registered rule (top-single, or top-stacked if its val
#                primary metric is at least 1.0 point higher; read from results/seeds.json, written by
#                analyze_seeds.py --select).
# Runs land in runs/jev-opt.
# Usage: run_seeds.sh val|test
set -uo pipefail
cd "$(dirname "$0")/../.."
CODE=.
EXT=$CODE/jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/seeds.log
SEEDS=(1 2)
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
record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}

phase=${1:?usage: run_seeds.sh val|test}
status=0
for seed in "${SEEDS[@]}"; do
  for spec in "${tasks[@]}"; do
    IFS=: read -r dir stem suffix <<<"$spec"
    id=v3-$stem-P-s$seed
    if [[ $phase == val ]]; then
      finished "$id" && continue
      if [[ -d $ROOT/$id ]]; then
        echo "$(date -Is) resume $id" | tee -a "$LOG"
        omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
      else
        cfg=$CODE/jev-opt/experiments/$dir/task_v3$suffix.yaml
        echo "$(date -Is) optimize $id ($cfg; seed $seed)" | tee -a "$LOG"
        omp_cmd "/jev-optimize $cfg --conditions jev-population-opt --run-id $id --seed $seed" >>"$LOG" 2>&1
      fi
      if finished "$id"; then
        record "run(seeds): $id validation (optimizer seed $seed)"
      else
        echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; status=1
      fi
    elif [[ $phase == test ]]; then
      [[ -f $ROOT/$id/test_consumed.json ]] && continue
      rep=$(python3 -c "import json,sys; print(json.load(open('results/seeds.json'))['selection']['$stem']['$seed']['representative'])") || { status=1; continue; }
      echo "$(date -Is) test $id $rep" | tee -a "$LOG"
      omp_cmd "/jev-evaluate-final $id $rep --yes" >>"$LOG" 2>&1
      if [[ -f $ROOT/$id/test_consumed.json ]]; then
        record "run(seeds): $id $rep test"
      else
        echo "$(date -Is) ERROR: $id test not recorded" | tee -a "$LOG"; status=1
      fi
    fi
  done
done
echo "$(date -Is) seeds phase $phase done (status $status)" | tee -a "$LOG"
exit $status
