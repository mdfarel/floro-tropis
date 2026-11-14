#!/usr/bin/env bash
# Extra lane started 2026-11-13 03:5x to use the 6 cores freed when lane A finished.
# Runs the floro-s2only head (last item of lane B) early; lane B skips it if this lane's .done marker exists by then.
set -u
cd "$(dirname "$0")/.."
L=logs/run
echo "$(date '+%F %T') start head_floro-s2only (lane D)" >> $L/STATUS
if .venv/bin/python -I src/train_head.py --model floro-s2only --seeds 0 1 2 --threads 5 > $L/head_floro-s2only.log 2>&1; then
  touch $L/head_floro-s2only.done; echo "$(date '+%F %T') done head_floro-s2only (lane D)" >> $L/STATUS
else
  touch $L/head_floro-s2only.failed; echo "$(date '+%F %T') FAILED head_floro-s2only (lane D)" >> $L/STATUS
fi
