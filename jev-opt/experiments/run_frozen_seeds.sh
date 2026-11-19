#!/usr/bin/env bash
# Seed variance of the frozen configuration on the 5 eligible sets (pre-registered): run_frozen.sh
# phase p (fz-<stem>-P) rerun with optimizer seeds 1 and 2, exactly as run_seeds.sh does for the v3 runs.
#   phase val  : /jev-optimize <dir>/task_v3.yaml --conditions jev-population-opt --seed <s> -> fz-<stem>-P-s<s>
#                VALIDATION ONLY; never calls /jev-evaluate-final.
#   phase test : /jev-evaluate-final fz-<stem>-P-s<s> <representative> --yes, once per run, representative read
#                from results/frozen_seeds.json (written by analyze_frozen_seeds.py --select).
# Cost cap: stops before any run / test call once the 10 runs' spend (fresh Jev from the response shards +
# optimizer from usage.jsonl + test cost) exceeds $CAP (default 60).
# Runs land in runs/jev-opt.
# Usage: run_frozen_seeds.sh val|test
set -uo pipefail
cd "$(dirname "$0")/../.."
CODE=.
EXT=$CODE/jev-opt/extension/index.ts
ROOT=runs/jev-opt
LOG=$ROOT/frozen_seeds.log
EXP=jev-opt/experiments
CAP=${CAP:-60}
SEEDS=(1 2)
tasks=(  # experiment dir : run-name stem
  contempo_matres:contempo-matres
  lifetox:lifetox
  sciprompt_s2orc:sciprompt-s2orc
  smarter_hatexplain:smarter-hatexplain
  smarter_latent_hate:smarter-latent-hate
)

omp_cmd() { ( flock 9; omp -p --no-session --no-extensions -e "$EXT" "$1" </dev/null ) 9>/tmp/jev-run.lock; }
finished() { grep -q '"state": "finished"' "$ROOT/$1/run.json" 2>/dev/null; }
record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}
spent() {
  python3 - "$ROOT" <<'PY'
import glob, gzip, json, sys
from pathlib import Path
tot = 0.0  # fresh + replayed Jev (upper bound), optimizer, test
for d in map(Path, glob.glob(f"{sys.argv[1]}/fz-*-P-s[12]")):
    for f in sorted(d.glob("jev_responses*.jsonl*")):
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tot += float((json.loads(line).get("response", {}).get("usage") or {}).get("cost") or 0.0)
                except ValueError:
                    pass
    if (d / "usage.jsonl").exists():
        for line in (d / "usage.jsonl").read_text().splitlines():
            if line.strip() and json.loads(line).get("source") == "omp":
                tot += float(json.loads(line).get("cost_total") or 0.0)
    if (d / "test_consumed.json").exists():
        tot += json.loads((d / "test_consumed.json").read_text())["metrics"].get("total_cost") or 0.0
print(f"{tot:.4f}")
PY
}
check_cap() {
  local s; s=$(spent) || s=0
  echo "$(date -Is) spend so far \$$s (cap \$$CAP)" | tee -a "$LOG"
  python3 -c "import sys; sys.exit(0 if float('$s') <= float('$CAP') else 1)" \
    || { echo "$(date -Is) STOP: cost cap exceeded" | tee -a "$LOG"; exit 3; }
}

phase=${1:?usage: run_frozen_seeds.sh val|test}
status=0
for seed in "${SEEDS[@]}"; do
  for spec in "${tasks[@]}"; do
    IFS=: read -r dir stem <<<"$spec"
    id=fz-$stem-P-s$seed
    if [[ $phase == val ]]; then
      finished "$id" && continue
      check_cap
      if [[ -d $ROOT/$id ]]; then
        echo "$(date -Is) resume $id" | tee -a "$LOG"
        omp_cmd "/jev-resume $id" >>"$LOG" 2>&1
      else
        cfg=$PWD/$EXP/$dir/task_v3.yaml
        echo "$(date -Is) optimize $id ($cfg; seed $seed)" | tee -a "$LOG"
        omp_cmd "/jev-optimize $cfg --conditions jev-population-opt --run-id $id --seed $seed" >>"$LOG" 2>&1
      fi
      if finished "$id"; then
        record "run(frozen-seeds): $id validation (optimizer seed $seed)"
      else
        echo "$(date -Is) ERROR: $id did not finish" | tee -a "$LOG"; status=1
      fi
    elif [[ $phase == test ]]; then
      [[ -f $ROOT/$id/test_consumed.json ]] && continue
      check_cap
      rep=$(python3 -c "import json; print(json.load(open('results/frozen_seeds.json'))['selection']['$stem']['$seed']['representative'])") || { status=1; continue; }
      echo "$(date -Is) test $id $rep" | tee -a "$LOG"
      omp_cmd "/jev-evaluate-final $id $rep --yes" >>"$LOG" 2>&1
      if [[ -f $ROOT/$id/test_consumed.json ]]; then
        record "run(frozen-seeds): $id $rep test"
      else
        echo "$(date -Is) ERROR: $id test not recorded" | tee -a "$LOG"; status=1
      fi
    else
      echo "unknown phase $phase" >&2; exit 2
    fi
  done
done
check_cap
echo "$(date -Is) frozen-seeds phase $phase done (status $status)" | tee -a "$LOG"
exit $status
