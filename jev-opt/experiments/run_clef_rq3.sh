#!/usr/bin/env bash
# Optimization step (RQ3: optimization with training labels) on Clef / Clef-flash answers (pre-registered); driver: clef_rq3.py.
#
#   run_clef_rq3.sh --model clef [--model clef-flash] --device cuda:N [--slot gpu0|gpu1 | --set <stem> ...] [--dry-run]
#   run_clef_rq3.sh --model clef [--model clef-flash] --device cuda:N --smoke <stem> [--n 20]
#
# Models given twice run one after the other (all sets of the first model, then the second).
# --slot   the sets that clef_rq3_assignment.json gives to that GPU (H100 machine: one command per GPU);
#          default: all 13 sets in pre-registered order.
# --dry-run  validates assignment, inputs, pins, weights and cache path; loads no model, makes no request;
#            writes results/clef_rq3/plan-<model>-<slot|all>.json.
# Per set: GPU check (aborts with status 3 if the GPU already holds >1 GiB), then calibration step and transfer
# of the optimized questions, then a log line.
# A set whose two test evaluations are recorded is skipped; an interrupted set resumes from the response cache.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # cuda:N == nvidia-smi index N

models=() device="" slot="" dry=0 smoke="" n=20 sets=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) models+=("$2"); shift 2 ;;
    --device) device=$2; shift 2 ;;
    --slot) slot=$2; shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    --dry-run) dry=1; shift ;;
    --smoke) smoke=$2; shift 2 ;;
    --n) n=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ ${#models[@]} -gt 0 ]] || { echo "--model clef|clef-flash (repeatable)" >&2; exit 2; }
for model in "${models[@]}"; do
  [[ $model == clef || $model == clef-flash ]] || { echo "--model clef|clef-flash" >&2; exit 2; }
done
[[ $device == cuda:* ]] || { echo "--device cuda:N" >&2; exit 2; }
idx=${device#cuda:}
ROOT=runs/jev-opt
PY=(uv run --project jev-opt/python python jev-opt/experiments/clef_rq3.py)

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
for model in "${models[@]}"; do
  LOG=$ROOT/clef_rq3-$model.log
  if [[ $dry == 1 ]]; then
    sel=(); [[ -n $slot ]] && sel=(--slot "$slot")
    for s in "${sets[@]}"; do sel+=(--set "$s"); done
    "${PY[@]}" plan --model "$model" --device "$device" "${sel[@]}" --out "results/clef_rq3/plan-$model-${slot:-all}.json" || status=1
    continue
  fi

  if [[ -n $smoke ]]; then
    gpu_free "smoke $smoke" || exit 3
    "${PY[@]}" smoke --model "$model" --device "$device" --set "$smoke" --n "$n" 2>>"$LOG.stderr" || exit 1
    record "run(clef-rq3): $model smoke on $n fit items of $smoke"
    continue
  fi

  run_sets=("${sets[@]}")
  if [[ -n $slot ]]; then
    mapfile -t run_sets < <(python3 -c "import json,sys; print('\n'.join(json.load(open('jev-opt/experiments/clef_rq3_assignment.json'))['slots'][sys.argv[1]]))" "$slot")
  elif [[ ${#sets[@]} -eq 0 ]]; then
    mapfile -t run_sets < <(python3 -c "import json; print('\n'.join(json.load(open('jev-opt/experiments/clef_rq3_workload.json'))))")
  fi
  echo "$(date -Is) $model on $device: ${run_sets[*]}" | tee -a "$LOG"
  for stem in "${run_sets[@]}"; do
    cal=$ROOT/cl-$model-$stem-cal opt=$ROOT/cl-$model-$stem-opt
    if [[ -f $cal/test_consumed.json && -f $opt/test_consumed.json ]]; then
      echo "$(date -Is) $stem already finished; skipped" | tee -a "$LOG"; continue
    fi
    gpu_free "$stem" || exit 3
    if "${PY[@]}" run --model "$model" --device "$device" --set "$stem" 2>>"$LOG.stderr"; then
      record "run(clef-rq3): $model $stem calibration step + transferred questions (val, one test evaluation each)"
    else
      echo "$(date -Is) ERROR: $stem failed (see $LOG.stderr); stopped" | tee -a "$LOG"
      exit 1
    fi
  done
done
exit $status
