#!/usr/bin/env bash
# Budget-matched baselines (pre-registered) on 2 GPUs. Each worker claims (family, set) jobs from a shared queue
# (mkdir claims under $QUEUE), so both GPUs share the B3 sets. Run from the repository root:
#   bash jev-opt/experiments/run_budget.sh            (GPU 0: B1, B2, then B3; GPU 1: B3)
set -euo pipefail
QUEUE=${QUEUE:-/tmp/budget_baseline_queue}
LOG=${LOG:-runs/budget_baseline}
mkdir -p "$QUEUE" "$LOG"
# longest B3 sets first
SETS="dress-new-content dress-new-organization dress-new-language lifetox sciprompt-s2orc ezstance-subtaskA inli
edit-intent contempo-tbdense smarter-implicit-hate contempo-matres smarter-hatexplain smarter-latent-hate"
run() {
  CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project jev-opt/python --with setfit==1.2.0 --with sentencepiece --with protobuf \
    python jev-opt/experiments/budget_baseline.py "$2" --sets "$3" --device cuda
}
worker() {
  local gpu=$1; shift
  for fam in "$@"; do
    for s in $SETS; do
      mkdir "$QUEUE/$fam-$s" 2>/dev/null || continue
      run "$gpu" "$fam" "$s" >>"$LOG/$fam-$s.log" 2>&1 || echo "FAILED $fam $s" >>"$LOG/failures.txt"
    done
  done
}
worker 0 emb setfit ft &
worker 1 ft &
wait
