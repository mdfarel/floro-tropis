#!/usr/bin/env bash
# Full FLORO-Tropis pilot run. Resumable: a step whose logs/run/<name>.done marker exists is skipped and
# extractors resume partial outputs. On a fresh clone every step runs and overwrites the published
# results/*.json. Logs in logs/run/, status in logs/run/STATUS.
#   setsid nohup bash scripts/run_all.sh > logs/run/master.log 2>&1 < /dev/null &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python -I"
PYTM="$ROOT/.venv-tm/bin/python -I"
L="$ROOT/logs/run"
mkdir -p "$L" results
SITES="mahakam sembilang bintuni langsa segara_anakan kubu_raya mimika kendari"

status() { echo "$(date '+%F %T') $*" | tee -a "$L/STATUS"; }

step() {  # step <name> <command...>; skips when logs/run/<name>.done exists
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then status "skip $name (done)"; return 0; fi
  status "start $name"
  if "$@" > "$L/$name.log" 2>&1; then touch "$L/$name.done"; status "done $name"; return 0; fi
  status "FAILED $name (see logs/run/$name.log)"; touch "$L/$name.failed"; return 1
}

wait_files() {  # wait_files <failed-marker-glob> <file...>; returns 1 if a marker appears
  local marker=$1; shift
  while true; do
    local missing=0
    for f in "$@"; do [ -f "$f" ] || missing=1; done
    [ $missing -eq 0 ] && return 0
    compgen -G "$marker" > /dev/null && return 1
    sleep 30
  done
}

F=data/features
side() { echo "$F/$1_$2_$3.json"; }  # model cond split
need_s2s1="$(side floro-s2s1 clean train) $(side floro-s2s1 clean val) $(side floro-s2s1 clean test) $(side floro-s2s1 cloud50 test) $(side floro-s2s1 cloud100 test)"
need_s2only="$(side floro-s2only clean train) $(side floro-s2only clean val) $(side floro-s2only clean test) $(side floro-s2only cloud50 test)"
need_tm="$(side terramind-l clean train) $(side terramind-l clean val) $(side terramind-l clean test) $(side terramind-l cloud50 test) $(side terramind-l cloud100 test)"
need_pr="$(side prithvi2-300m clean train) $(side prithvi2-300m clean val) $(side prithvi2-300m clean test) $(side prithvi2-300m cloud50 test) $(side prithvi2-300m cloud100 test)"
FR=data/features_dataset_raw
need_raw="$FR/floro-s2s1_clean_train.json $FR/floro-s2s1_clean_val.json $FR/floro-s2s1_clean_test.json $FR/floro-s2s1_cloud50_test.json $FR/floro-s2s1_cloud100_test.json"

# ---------- 0. wait for all site downloads
status "waiting for site downloads"
while pgrep -f "src/download.py" > /dev/null; do sleep 30; done
for s in $SITES; do
  [ -f "data/sites/$s/label.tif" ] || { status "FAILED: data/sites/$s/label.tif missing after downloads ended"; exit 1; }
done
status "all 8 sites downloaded"

# ---------- 1. patches
step patches $PY src/patches.py || exit 1

# ---------- extra S2 bands for TerraMind (network-bound, runs beside the CPU lanes)
(
  pids=()
  for grp in "mahakam langsa" "sembilang kubu_raya" "bintuni mimika kendari"; do
    todo=""
    for s in $grp; do [ -f "data/sites/$s/s2extra.tif" ] || todo="$todo $s"; done
    [ -n "$todo" ] && { $PY src/download_extra.py $todo > "$L/extra_${todo// /_}.log" 2>&1 & pids+=($!); }
  done
  ok=0; for p in "${pids[@]}"; do wait "$p" || ok=1; done
  for s in $SITES; do [ -f "data/sites/$s/s2extra.tif" ] || ok=1; done
  if [ $ok -eq 0 ]; then touch "$L/extra.done"; status "done extra bands"; else touch "$L/extra.failed"; status "FAILED extra bands"; fi
) &

# ---------- lane A: encoder feature extraction (6 threads), in priority order
(
  T=6
  for sp in "train clean" "val clean" "test clean" "test cloud50" "test cloud100"; do
    set -- $sp; step "feat_floro-s2s1_$2_$1" $PY src/extract_floro.py --variant s2s1 --split $1 --cond $2 --threads $T || true
  done
  if wait_files "$L/extra.failed" "$L/extra.done"; then
    for sp in "train clean" "val clean" "test clean" "test cloud50" "test cloud100"; do
      set -- $sp; step "feat_terramind-l_$2_$1" $PYTM src/extract_terramind.py --split $1 --cond $2 --threads $T || true
    done
  else status "skip TerraMind features: extra bands failed"; fi
  for sp in "train clean" "val clean" "test clean" "test cloud50" "test cloud100"; do
    set -- $sp; step "feat_prithvi2-300m_$2_$1" $PYTM src/extract_prithvi.py --split $1 --cond $2 --threads $T || true
  done
  for sp in "train clean" "val clean" "test clean" "test cloud50"; do
    set -- $sp; step "feat_floro-s2only_$2_$1" $PY src/extract_floro.py --variant s2only --split $1 --cond $2 --threads $T || true
  done
  for sp in "train clean" "val clean" "test clean" "test cloud50" "test cloud100"; do
    set -- $sp; step "feat_raw_floro-s2s1_$2_$1" $PY src/extract_floro.py --variant s2s1 --split $1 --cond $2 --threads $T --input-scale dataset_raw || true
  done
  status "lane A finished"
) &

# ---------- lane B (5 threads): GBM, radar-only GBM reference, then heads for FLORO s2s1, Prithvi, FLORO s2only
(
  T=5
  step gbm $PY src/gbm_baseline.py --seeds 0 1 2 --threads $T || true
  step gbm_sar $PY src/gbm_baseline.py --seeds 0 1 2 --threads $T --sar-only || true
  wait_files "$L/feat_floro-s2s1_*.failed" $need_s2s1 && step head_floro-s2s1 $PY src/train_head.py --model floro-s2s1 --seeds 0 1 2 --threads $T || true
  wait_files "$L/feat_prithvi2-300m_*.failed" $need_pr && step head_prithvi2-300m $PY src/train_head.py --model prithvi2-300m --seeds 0 1 2 --threads $T || true
  wait_files "$L/feat_floro-s2only_*.failed" $need_s2only && step head_floro-s2only $PY src/train_head.py --model floro-s2only --seeds 0 1 2 --threads $T || true
  status "lane B finished"
) &

# ---------- lane C (5 threads): U-Net seeds 0-2 and 3-4, then heads for TerraMind and the FLORO input-scale ablation
(
  T=5
  step unet $PY src/unet_baseline.py --seeds 0 1 2 --threads $T || true
  step unet_seeds34 $PY src/unet_baseline.py --seeds 3 4 --threads $T --results-dir results/unet_seeds34 || true
  wait_files "$L/feat_terramind-l_*.failed" $need_tm && step head_terramind-l $PY src/train_head.py --model terramind-l --seeds 0 1 2 --threads $T || true
  wait_files "$L/feat_raw_floro-s2s1_*.failed" $need_raw && step head_raw_floro-s2s1 $PY src/train_head.py --model floro-s2s1 --seeds 0 1 2 --threads $T --features-dir $FR --out results/floro-s2s1.dataset_raw.json || true
  status "lane C finished"
) &

wait
status "ALL LANES FINISHED"
