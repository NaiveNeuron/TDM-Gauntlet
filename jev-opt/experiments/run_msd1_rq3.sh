#!/usr/bin/env bash
# Optimization step (RQ3) on Microsoft-Decision-1 answers (pre-registered): run_luna_rq3.sh with MODEL=msd1.
MODEL=msd1 exec "$(dirname "$0")/run_luna_rq3.sh" "$@"
