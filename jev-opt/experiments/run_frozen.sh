#!/usr/bin/env bash
# Frozen-configuration round (final optimizer settings): the v3 configuration on every eligible shortfall set.
#   phase seeds : /jev-seeds <dir>/task_v3.yaml --out <dir>/seeds_v3 --n 3 (optimizer LLM only, no Jev call)
#   phase a     : fz-<stem>-A = jev-zero (paper question + readout); records the fit-slice/val answers that
#                 the calibration step uses. VALIDATION ONLY.
#   phase p     : fz-<stem>-P = jev-population-opt. VALIDATION ONLY.
#   phase test  : /jev-evaluate-final fz-<stem>-P <representative> --yes, once per set, representative read
#                 from results/frozen.json (written by analyze_frozen.py --select).
# Runs land in runs/jev-opt.
# Usage: run_frozen.sh seeds|a|p|test
set -uo pipefail
cd "$(dirname "$0")/../.."
CODE=.
EXT=$CODE/jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/frozen.log
EXP=jev-opt/experiments
tasks=(  # experiment dir : run-name stem
  contempo_matres:contempo-matres
  lifetox:lifetox
  sciprompt_s2orc:sciprompt-s2orc
  smarter_hatexplain:smarter-hatexplain
  smarter_latent_hate:smarter-latent-hate
)

omp_cmd() { ( flock 9; omp -p --no-session --no-extensions -e "$EXT" "$1" </dev/null ) 9>/tmp/jev-run.lock; }
omp_free() { omp -p --no-session --no-extensions -e "$EXT" "$1" </dev/null; }
finished() { grep -q '"state": "finished"' "$ROOT/$1/run.json" 2>/dev/null; }
record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}
run() {  # run <run-id> <task.yaml> <condition>
  local id=$1 cfg=$2 cond=$3
  finished "$id" && return 0
  if [[ -d $ROOT/$id ]]; then
    echo "$(date -Is) resume $id" | tee -a "$LOG"
    omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
  else
    echo "$(date -Is) optimize $id ($cfg; $cond)" | tee -a "$LOG"
    omp_cmd "/jev-optimize $PWD/$cfg --conditions $cond --run-id $id" >>"$LOG" 2>&1
  fi
  if finished "$id"; then
    record "run(frozen): $id validation"
  else
    echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; return 1
  fi
}

phase=${1:?usage: run_frozen.sh seeds|a|p|test}
status=0
for spec in "${tasks[@]}"; do
  IFS=: read -r dir stem <<<"$spec"
  cfg=$EXP/$dir/task_v3.yaml
  case $phase in
    seeds)
      out=$EXP/$dir/seeds_v3
      [[ -f $out/generation.json ]] && continue
      echo "$(date -Is) seeds $stem -> $out" | tee -a "$LOG"
      omp_free "/jev-seeds $PWD/$cfg --out $PWD/$out --n 3" >>"$LOG" 2>&1
      if [[ -f $out/generation.json ]]; then
        record "seeds(frozen): $stem jev-native + decomposed seeds (/jev-seeds, task definition only)"
      else
        echo "$(date -Is) ERROR: no seeds for $stem" | tee -a "$LOG"; status=1
      fi ;;
    a) run "fz-$stem-A" "$cfg" jev-zero || status=1 ;;
    p) run "fz-$stem-P" "$cfg" jev-population-opt || status=1 ;;
    test)
      id=fz-$stem-P
      [[ -f $ROOT/$id/test_consumed.json ]] && continue
      rep=$(python3 -c "import json; print(json.load(open('results/frozen.json'))['selection']['$stem']['representative'])") || { status=1; continue; }
      echo "$(date -Is) test $id $rep" | tee -a "$LOG"
      omp_cmd "/jev-evaluate-final $id $rep --yes" >>"$LOG" 2>&1
      if [[ -f $ROOT/$id/test_consumed.json ]]; then
        record "run(frozen): $id $rep test"
      else
        echo "$(date -Is) ERROR: $id test not recorded" | tee -a "$LOG"; status=1
      fi ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
done
echo "$(date -Is) frozen phase $phase done (status $status)" | tee -a "$LOG"
exit $status
