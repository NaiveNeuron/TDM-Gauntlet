#!/usr/bin/env bash
# Zero-shot sweep of a Decisions-endpoint model on the frozen gauntlet cohorts: GPT-6 Luna or, with
# MODEL=msd1 (scripts/run_msd1.sh), Microsoft-Decision-1.
#
#   [MODEL=luna|msd1] scripts/run_luna.sh [--rq3-first] [task ...]
#
# Default: every frozen zero-shot cohort except Wiki-WD, the 11 cohorts the RQ3 transfer needs first
# (--rq3-first: only those); for msd1 the four TTP cohorts and MADE last, each after a projection check
# (`python -m gauntlet.luna project`: no request if the projected spend exceeds the zero-shot cap). One
# `gauntlet run --backend $MODEL` per cohort (resumable: responses are cached, a finished cohort is skipped), then
# a log line. A failure (budget cap, off-pin snapshot, API error,
# projection) stops the sweep with status 1.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-luna}
LOG=runs/$MODEL/zero_shot.log
mkdir -p "runs/$MODEL"
RQ3=(lifetox contempo/matres contempo/tbdense ezstance/subtaskA smarter/hatexplain smarter/implicit_hate
     smarter/latent_hate edit_intent inli dress_new sciprompt/s2orc)
LAST=()  # projection-checked cohorts, run last (msd1 only)
[[ $MODEL == msd1 ]] && LAST=(ttp/expert ttp/tram ttp/expert_full ttp/procedures_plus made)

if [[ ${1:-} == --rq3-first ]]; then
  tasks=("${RQ3[@]}")
elif [[ $# -gt 0 ]]; then
  tasks=("$@")
else
  mapfile -t rest < <(uv run python -c "
from gauntlet.clef_report import zero_shot_names
from gauntlet.luna import SPECS
skip = set('''${RQ3[*]} ${LAST[*]}'''.split()) | set(SPECS['$MODEL'].excluded)
print('\n'.join(n for n in zero_shot_names() if n not in skip and n != 'biorc800/dev'))")
  tasks=("${RQ3[@]}" "${rest[@]}" "${LAST[@]}")
fi

record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}

echo "$(date -Is) sweep (re)started: concurrency ${CONCURRENCY:-32}, ≤ 1,200 requests/minute" | tee -a "$LOG"
for task in "${tasks[@]}"; do
  if [[ -f runs/$MODEL/$task/scores.json ]]; then
    echo "$(date -Is) $task already finished; skipped" | tee -a "$LOG"; continue
  fi
  if [[ -f runs/$MODEL/HOLD ]] && grep -qxF -- "$task" "runs/$MODEL/HOLD"; then
    echo "$(date -Is) $task HELD (runs/$MODEL/HOLD): skipped, no request" | tee -a "$LOG"; continue
  fi
  if [[ " ${LAST[*]} " == *" $task "* ]]; then
    if ! proj=$(uv run python -m gauntlet.luna project "$MODEL" "$task" 2>>"$LOG.stderr"); then
      echo "$(date -Is) STOP: projection before $task: ${proj:-failed (see $LOG.stderr)}" | tee -a "$LOG"
      exit 1
    fi
    echo "$(date -Is) projection before $task: $proj" | tee -a "$LOG"
  fi
  echo "$(date -Is) $task start" | tee -a "$LOG"
  if uv run gauntlet run "$task" --backend "$MODEL" --concurrency "${CONCURRENCY:-32}" --rpm 1200 >>"$LOG.stdout" 2>>"$LOG.stderr"; then
    uv run python -m gauntlet.luna summary "$MODEL" >/dev/null  # spend digest; the ledger itself is git-ignored
    summary=$(uv run python -c "
import json,sys; m=json.load(open('runs/$MODEL/$task/manifest.json')); a=m['answer_audit']
print(f\"{m['n_requests']} requests, {m['input_tokens']} tokens, \${m['cost_usd']:.4f}, refused {a['n_refused_answers']}/{a['n_answers']}, rejected {a['n_rejected_requests']}, snapshots {m['model_snapshots']}\")")
    echo "$(date -Is) $task done: $summary" | tee -a "$LOG"
    record "run($MODEL): $task zero-shot on the frozen requests ($summary)"
  else
    echo "$(date -Is) ERROR: $task failed (see $LOG.stderr); sweep stopped" | tee -a "$LOG"
    exit 1
  fi
done
echo "$(date -Is) sweep finished" | tee -a "$LOG"
