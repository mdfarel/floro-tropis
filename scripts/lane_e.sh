#!/usr/bin/env bash
# Audit remedies (2026-11-13): GBM re-run with per-patch confusion, radar-only GBM reference, 2 extra U-Net seeds.
set -u
cd "$(dirname "$0")/.."
L=logs/run; PY=".venv/bin/python -I"
st() { echo "$(date '+%F %T') $*" >> $L/STATUS; }
run() { local n=$1; shift; [ -f $L/$n.done ] && return 0; st "start $n (lane E)"
  if "$@" > $L/$n.log 2>&1; then touch $L/$n.done; st "done $n (lane E)"; else touch $L/$n.failed; st "FAILED $n (lane E)"; fi; }
run gbm_perpatch $PY src/gbm_baseline.py --seeds 0 1 2 --threads 4
run gbm_sar $PY src/gbm_baseline.py --seeds 0 1 2 --threads 4 --sar-only
# wait until one of the long head jobs has finished so the cores are free
while [ ! -f $L/head_floro-s2only.done ] && [ ! -f $L/head_terramind-l.done ] && [ ! -f $L/head_prithvi2-300m.done ]; do sleep 60; done
run unet_seeds34 $PY src/unet_baseline.py --seeds 3 4 --threads 5 --results-dir results/unet_seeds34
st "lane E finished"
