#!/usr/bin/env bash
# H100 LoRA (pre-registered); driver: h100_lora.py, trainer: gauntlet.lora_train (`lora` extra).
#
#   run_h100_lora.sh --device cuda:N [--slot gpu0|gpu1 | --set <stem> ...] [--dry-run]
#   run_h100_lora.sh --device cuda:N --preflight                (compiler/driver/backward checks; writes a report)
#   run_h100_lora.sh --device cuda:N --smoke [<stem>]        (default smarter-latent-hate; must run before any set)
#
# --slot     the sets that h100_lora_assignment.json gives to that GPU (one command per GPU); default: all 13 sets.
# --dry-run  validates assignment, weights, trainer environment (incl. the static backward checks), data and training
#            targets; loads no model; writes results/h100_lora/plan-<slot|all>.json.
# --device cuda:0+cuda:1 is the pre-registered OOM fallback (model split over both GPUs).
# Preflight: before smoke and each slot, on every GPU of --device:
# pinned TileLang and standalone CUDA 12.8.1 NVCC 12.8.93 (trainer pins compiler/cache), driver >= CUDA 12.8,
# actual TileLang backward and output/all gradients within 0.05 relative L2 of FP32 reference; failure exits 4
# (the report is written to runs/jev-opt/lora-preflight/cudaN.json).
# Per set: GPU check (stops with status 3 if a GPU already holds >1 GiB), 2 learning rates × selected seeds (a finished run is
# skipped, an interrupted one restarts), val decoding, selection, test once per seed, then a log line.
# A finished set is skipped. A failed set stops with status 1.
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # cuda:N == nvidia-smi index N

device="" slot="" dry=0 smoke="" pre=0 sets=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --device) device=$2; shift 2 ;;
    --slot) slot=$2; shift 2 ;;
    --set) sets+=("$2"); shift 2 ;;
    --dry-run) dry=1; shift ;;
    --preflight) pre=1; shift ;;
    --smoke) if [[ $# -gt 1 && $2 != --* ]]; then smoke=$2; shift 2; else smoke=smarter-latent-hate; shift; fi ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ $device == cuda:* ]] || { echo "--device cuda:N (or cuda:0+cuda:1)" >&2; exit 2; }
IFS=+ read -r -a devs <<< "$device"
ROOT=${H100_LORA_RUNS:-runs/jev-opt}  # tests point this at a temporary directory
LOG=$ROOT/h100_lora.log
PY=(uv run --project jev-opt/python python jev-opt/experiments/h100_lora.py)
TRAINER=(uv run --no-sync --extra lora python -m gauntlet.lora_train)
PRE=$ROOT/lora-preflight

record() {  # record <message>: note a finished step in the log
  echo "$(date -Is) done: $1" >>"$LOG"
}
gpu_free() {
  local used d
  for d in "${devs[@]}"; do
    used=$(nvidia-smi -i "${d#cuda:}" --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
    if (( used > 1024 )); then
      echo "$(date -Is) GPU ${d#cuda:} holds ${used} MiB before $1; pausing (status 3)" | tee -a "$LOG"
      return 1
    fi
  done
}
finished() {  # every tested seed of the set's selection has its test evaluation recorded
  python3 - "$1" <<'EOF'
import json, sys
from pathlib import Path
root = Path("runs/jev-opt")
sel = root / f"lora-{sys.argv[1]}" / "selection.json"
ok = sel.exists() and all((root / f"lora-{sys.argv[1]}-s{t['seed']}" / "test_consumed.json").exists()
                          for t in json.loads(sel.read_text())["tested"])
sys.exit(0 if ok else 1)
EOF
}
preflight() {  # On every GPU of --device; status 4 on any compiler/driver/reference problem
  local d
  mkdir -p "$PRE"
  for d in "${devs[@]}"; do
    echo "$(date -Is) preflight on $d" | tee -a "$LOG"
    if ! "${TRAINER[@]}" preflight --device "$d" --out "$PRE/cuda${d#cuda:}.json" 2>>"$LOG.stderr" | tee -a "$LOG"; then
      echo "$(date -Is) ERROR: preflight failed on $d (see $PRE/cuda${d#cuda:}.json); stopped (status 4)" | tee -a "$LOG"
      exit 4
    fi
  done
}

if [[ $dry == 1 ]]; then
  sel=(); [[ -n $slot ]] && sel=(--slot "$slot")
  for s in "${sets[@]}"; do sel+=(--set "$s"); done
  mkdir -p results/h100_lora
  "${PY[@]}" plan --device "$device" "${sel[@]}" --out "results/h100_lora/plan-${slot:-all}.json"
  exit $?
fi

gpu_free "preflight" || exit 3
preflight
if [[ $pre == 1 ]]; then
  record "run(h100-lora): preflight on $device (pinned compiler, driver, TileLang backward vs reference)"
  exit 0
fi

if [[ -n $smoke ]]; then
  gpu_free "smoke $smoke" || exit 3
  "${PY[@]}" smoke --device "$device" --set "$smoke" 2>>"$LOG.stderr" || exit 1
  record "run(h100-lora): smoke on $smoke (untrained adapter vs Exp. 1, 2 steps, reload; seed count)"
  exit 0
fi

run_sets=("${sets[@]}")
if [[ -n $slot ]]; then
  mapfile -t run_sets < <(python3 -c "import json,sys; print('\n'.join(json.load(open('jev-opt/experiments/h100_lora_assignment.json'))['slots'][sys.argv[1]]))" "$slot")
elif [[ ${#sets[@]} -eq 0 ]]; then
  mapfile -t run_sets < <(python3 -c "import json; print('\n'.join(json.load(open('jev-opt/experiments/h100_lora_workload.json'))))")
fi
echo "$(date -Is) LoRA on $device: ${run_sets[*]}" | tee -a "$LOG"
for stem in "${run_sets[@]}"; do
  if finished "$stem"; then
    echo "$(date -Is) $stem already finished; skipped" | tee -a "$LOG"; continue
  fi
  gpu_free "$stem" || exit 3
  if "${PY[@]}" run --device "$device" --set "$stem" 2>>"$LOG.stderr"; then
    record "run(h100-lora): $stem (training runs, val selection, one test evaluation per seed)"
  else
    echo "$(date -Is) ERROR: $stem failed (see $LOG.stderr); stopped" | tee -a "$LOG"
    exit 1
  fi
done
exit 0
