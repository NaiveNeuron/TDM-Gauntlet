#!/usr/bin/env bash
# H100 round 2 (pre-registered); driver: h100_round2.py.
#
#   run_h100_round2.sh --exp base [--exp native] --device cuda:N [--slot gpu0|gpu1 | --set <stem> ...] [--dry-run]
#   run_h100_round2.sh --exp base [--exp native] --device cuda:N --smoke <stem> [--n 20]
#
# Experiments given twice run one after the other (all sets of the first, then the second):
#   base    Clef's base model Qwen3.8-27B through option log-probabilities (calibration step + optimized questions)
#   native  Clef on the label-free jev-native questions (zero-shot + refitted readout)
# --slot   the sets that h100_round2_assignment.json gives to that GPU for that experiment (one command per GPU);
#          default: all 13 sets in pre-registered order.
# --dry-run  validates assignment, inputs, pins, weights, caches, template limits and the frozen-unit identity;
#            loads no model, makes no request; writes results/h100_round2/plan-<exp>-<slot|all>.json.
# Per set: GPU check (stops with status 3 if the GPU already holds >1 GiB), both steps, then a log line.
# A set whose two test evaluations are recorded is skipped; an interrupted set resumes from the
# response cache. A failed set stops the launcher with status 1.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # cuda:N == nvidia-smi index N

exps=() device="" slot="" dry=0 smoke="" n=20 sets=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --exp) exps+=("$2"); shift 2 ;;
    --device) device=$2; shift 2 ;;
    --slot) slot=$2; shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    --dry-run) dry=1; shift ;;
    --smoke) smoke=$2; shift 2 ;;
    --n) n=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ ${#exps[@]} -gt 0 ]] || { echo "--exp base|native (repeatable)" >&2; exit 2; }
for exp in "${exps[@]}"; do
  [[ $exp == base || $exp == native ]] || { echo "--exp base|native" >&2; exit 2; }
done
[[ $device == cuda:* ]] || { echo "--device cuda:N" >&2; exit 2; }
idx=${device#cuda:}
ROOT=runs/jev-opt
PY=(uv run --project jev-opt/python python jev-opt/experiments/h100_round2.py)
declare -A MODEL=([base]=qwen-base [native]=clef)
declare -A STEPS=([base]="cal opt" [native]="native native-lr")

record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}
gpu_free() {
  local used
  used=$(nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
  if (( used > 1024 )); then
    echo "$(date -Is) GPU $idx holds ${used} MiB before $1; pausing (status 3)" | tee -a "$LOG"
    return 1
  fi
}

status=0
for exp in "${exps[@]}"; do
  LOG=$ROOT/h100_round2-$exp.log
  model=${MODEL[$exp]}
  if [[ $dry == 1 ]]; then
    sel=(); [[ -n $slot ]] && sel=(--slot "$slot")
    for s in "${sets[@]}"; do sel+=(--set "$s"); done
    "${PY[@]}" plan --exp "$exp" --device "$device" "${sel[@]}" --out "results/h100_round2/plan-$exp-${slot:-all}.json" || status=1
    continue
  fi

  if [[ -n $smoke ]]; then
    gpu_free "smoke $smoke" || exit 3
    "${PY[@]}" smoke --exp "$exp" --device "$device" --set "$smoke" --n "$n" 2>>"$LOG.stderr" || exit 1
    record "run(h100-round2): $exp smoke on $n fit items of $smoke"
    continue
  fi

  run_sets=("${sets[@]}")
  if [[ -n $slot ]]; then
    mapfile -t run_sets < <(python3 -c "import json,sys; print('\n'.join(json.load(open('jev-opt/experiments/h100_round2_assignment.json'))[sys.argv[1]]['slots'][sys.argv[2]]))" "$exp" "$slot")
  elif [[ ${#sets[@]} -eq 0 ]]; then
    mapfile -t run_sets < <(python3 -c "import json; print('\n'.join(json.load(open('jev-opt/experiments/h100_round2_workload.json'))))")
  fi
  echo "$(date -Is) $exp ($model) on $device: ${run_sets[*]}" | tee -a "$LOG"
  for stem in "${run_sets[@]}"; do
    done=1
    for step in ${STEPS[$exp]}; do
      d=$ROOT/h2-$model-$stem-$step
      [[ -f $d/test_consumed.json ]] || done=0
    done
    if [[ $done == 1 ]]; then
      echo "$(date -Is) $stem already finished; skipped" | tee -a "$LOG"; continue
    fi
    gpu_free "$stem" || exit 3
    if "${PY[@]}" run --exp "$exp" --device "$device" --set "$stem" 2>>"$LOG.stderr"; then
      record "run(h100-round2): $exp $stem (${STEPS[$exp]}; val, one test evaluation each)"
    else
      echo "$(date -Is) ERROR: $stem failed (see $LOG.stderr); stopped" | tee -a "$LOG"
      exit 1
    fi
  done
done
exit $status
