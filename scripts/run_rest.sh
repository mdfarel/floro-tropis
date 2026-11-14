#!/usr/bin/env bash
# Remaining runs after the 2026-11-13 10:17 machine restart. Safe to re-run: both trainers checkpoint every finished
# seed in results/_ckpt/ (or results/unet_seeds34/_ckpt/) and skip it on the next start.
#   setsid nohup bash scripts/run_rest.sh > logs/run/rest.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
L=logs/run; PY=".venv/bin/python -I"; FR=data/features_dataset_raw
st() { echo "$(date '+%F %T') $*" >> $L/STATUS; }
run() { local n=$1; shift; [ -f $L/$n.done ] && return 0; rm -f $L/$n.failed; st "start $n (rest)"
  if "$@" >> $L/$n.log 2>&1; then touch $L/$n.done; st "done $n (rest)"; else touch $L/$n.failed; st "FAILED $n (rest)"; fi; }
run head_prithvi2-300m $PY src/train_head.py --model prithvi2-300m --seeds 0 1 2 --threads 4 &
run head_floro-s2only $PY src/train_head.py --model floro-s2only --seeds 0 1 2 --threads 4 &
run unet_seeds34 $PY src/unet_baseline.py --seeds 3 4 --threads 4 --results-dir results/unet_seeds34 &
( run head_raw_seed0 $PY src/train_head.py --model floro-s2s1 --seeds 0 --threads 4 --features-dir $FR --out results/floro-s2s1.dataset_raw.json
  # seeds 1-2 of the ablation once the main runs are done (seed 0 is reused from the checkpoint)
  while pgrep -f "model prithvi2-300m|model floro-s2only|unet_baseline.py --seeds 3 4" > /dev/null; do sleep 60; done
  run head_raw_floro-s2s1 $PY src/train_head.py --model floro-s2s1 --seeds 0 1 2 --threads 8 --features-dir $FR --out results/floro-s2s1.dataset_raw.json ) &
wait
st "REST FINISHED"
