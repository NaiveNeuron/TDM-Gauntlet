#!/usr/bin/env bash
# Microsoft-Decision-1 zero-shot sweep: scripts/run_luna.sh with MODEL=msd1.
MODEL=msd1 exec "$(dirname "$0")/run_luna.sh" "$@"
