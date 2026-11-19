#!/usr/bin/env bash
# Optimization step (RQ3: optimization with training labels) on GPT-6 Luna answers or, with MODEL=msd1 (run_msd1_rq3.sh),
# on Microsoft-Decision-1 answers (both pre-registered); driver: luna_rq3.py --model $MODEL.
#
#   [MODEL=luna|msd1] run_luna_rq3.sh [--set <stem> ...] [--dry-run]
#   [MODEL=luna|msd1] run_luna_rq3.sh --smoke <stem> [--n 20]
#
# Default: all 13 sets in pre-registered order (clef_rq3.SETS).
# --dry-run  inputs, identity (where Luna's frozen run exists), embeddings, cache, key and spend; planned requests
#            and projected cost; makes no request; writes results/${MODEL}_rq3/plan.json.
# Per set: refuses to start unless Luna's frozen zero-shot run runs/$MODEL/<task>/scores.json exists, then the
# calibration step and the transfer of the optimized questions, then a log line.
# A set whose two test evaluations are recorded is skipped; an interrupted set resumes from the
# response cache. Any failure (budget cap, off-pin snapshot, API error) stops with status 1.
set -uo pipefail
cd "$(dirname "$0")/../.."

dry=0 smoke="" n=20 sets=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --set) sets+=("$2"); shift 2 ;;
    --dry-run) dry=1; shift ;;
    --smoke) smoke=$2; shift 2 ;;
    --n) n=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
MODEL=${MODEL:-luna}
ROOT=runs/jev-opt
LOG=$ROOT/${MODEL}_rq3.log
PY=(uv run --project jev-opt/python python jev-opt/experiments/luna_rq3.py)
MODEL_ARG=(--model "$MODEL")

record() {  # record <message>: note a finished step in the log
  uv run python -m gauntlet.luna summary "$MODEL" >/dev/null  # refresh the digest of the spend ledger
  echo "$(date -Is) done: $1" >>"$LOG"
}

if [[ $dry == 1 ]]; then
  sel=(); for s in "${sets[@]}"; do sel+=(--set "$s"); done
  "${PY[@]}" plan "${MODEL_ARG[@]}" "${sel[@]}" --out "results/${MODEL}_rq3/plan.json"
  exit $?
fi

if [[ -n $smoke ]]; then
  "${PY[@]}" smoke "${MODEL_ARG[@]}" --set "$smoke" --n "$n" 2>>"$LOG.stderr" || { echo "$(date -Is) ERROR: smoke $smoke failed (see $LOG.stderr)" | tee -a "$LOG"; exit 1; }
  record "run($MODEL-rq3): smoke on $n fit items of $smoke"
  exit 0
fi

run_sets=("${sets[@]}")
if [[ ${#sets[@]} -eq 0 ]]; then
  mapfile -t run_sets < <(python3 -c "import json; print('\n'.join(json.load(open('jev-opt/experiments/clef_rq3_workload.json'))))")
fi
frozen_task() {  # the gauntlet task of a set's frozen zero-shot run (clef_rq3.SETS)
  python3 - "$1" <<'EOF'
import ast, sys
src = open("jev-opt/experiments/clef_rq3.py").read()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "SETS")
print(ast.literal_eval(node.value)[sys.argv[1]][1])
EOF
}
echo "$(date -Is) $MODEL: ${run_sets[*]}" | tee -a "$LOG"
for stem in "${run_sets[@]}"; do
  cal=$ROOT/cl-$MODEL-$stem-cal opt=$ROOT/cl-$MODEL-$stem-opt
  if [[ -f $cal/test_consumed.json && -f $opt/test_consumed.json ]]; then
    echo "$(date -Is) $stem already finished; skipped" | tee -a "$LOG"; continue
  fi
  task=$(frozen_task "$stem") || { echo "$(date -Is) ERROR: unknown set $stem" | tee -a "$LOG"; exit 1; }
  if [[ -f runs/$MODEL/HOLD ]] && grep -qxF -- "$task" "runs/$MODEL/HOLD"; then
    echo "$(date -Is) $stem HELD (runs/$MODEL/HOLD): skipped, no request" | tee -a "$LOG"; continue
  fi
  if [[ ! -f runs/$MODEL/$task/scores.json ]]; then
    echo "$(date -Is) ERROR: $stem needs the frozen zero-shot run runs/$MODEL/$task (no scores.json); stopped" | tee -a "$LOG"
    exit 1
  fi
  if "${PY[@]}" run "${MODEL_ARG[@]}" --set "$stem" 2>>"$LOG.stderr"; then
    record "run($MODEL-rq3): $stem calibration step + transferred questions (val, one test evaluation each)"
  else
    echo "$(date -Is) ERROR: $stem failed (see $LOG.stderr); stopped" | tee -a "$LOG"
    exit 1
  fi
done
echo "$(date -Is) all sets done" | tee -a "$LOG"
