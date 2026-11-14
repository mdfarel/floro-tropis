# FLORO-Tropis pilot: data and code contract

Goal: frozen-encoder test of FLORO (KAUST, MIT) on Indonesian mangrove segmentation, against a U-Net trained
from scratch and a per-pixel gradient-boosting model, on 4 train sites and 4 unseen test sites. CPU only
(16 cores, 17 GB RAM, no GPU). Python env: `.venv` in the repository root (torch 2.15 CPU); TerraMind and
Prithvi run in `.venv-tm`. Run scripts with `python -I src/<script>.py` from the repo root; every script inserts
its own folder on `sys.path`.

## Site rasters (written by `src/download.py`)
`data/sites/<site>/s2.tif` 8 bands float32 surface reflectance 0..1, order B2 B3 B4 B5 B8 B8A B11 B12, NaN = never clear.
`data/sites/<site>/s1.tif` 2 bands float32 VV, VH in dB, NaN = missing.
`data/sites/<site>/label.tif` uint8, 1 = mangrove (10 m pixel overlaps any Global Mangrove Watch v3 2020 mangrove cell, ~25 m source), 0 = other.
All three share one 10 m UTM grid per site (about 2780 x 2780 px). Sites and split live in `src/sites.py`.

## Patches (written by `src/patches.py`)
`data/patches/optical.npy` float32 (N, 8, 256, 256), NaN where invalid.
`data/patches/sar.npy` float32 (N, 2, 256, 256), NaN where invalid.
`data/patches/label.npy` uint8 (N, 256, 256), values 0/1, 255 = ignore (no label).
`data/patches/meta.csv` columns: idx, site, split (train | val | test), row, col, x0, y0, mangrove_frac, s2_valid_frac.
Tiling: non-overlapping 256 x 256 windows. Keep a patch when s2_valid_frac >= 0.8 and s1 valid >= 0.8.
Split: test sites -> `test`. Train sites -> `train`, except the bottom two patch rows of each train site,
which form a spatially contiguous `val` block (used only for early stopping and model selection).
`data/patches/stats.json`: {"optical": {"B2": {"mean", "std", "min", "max"}, ...}, "sar": {"VV": {...}, "VH": {...}}}
mean/std computed on the `train` split only (never val/test). min/max are PHYSICAL bounds, not empirical:
optical 0.0 and 1.0, sar -60.0 and 20.0. (FLORO marks values outside min/max as invalid, so empirical
train min/max would wrongly invalidate bright test pixels.)

## Shared modules (used by every model)
`src/metrics.py`: `ConfusionAccumulator`, `iou_report(...)`; pooled and per-site IoU, mIoU, F1, accuracy.
`src/clouds.py`: `apply_clouds(optical, cover, seed)` -> copy with NaN cloud blobs covering `cover` of each patch;
cover 0.0 = clean, 0.5 = half the patch, 1.0 = optical entirely missing. Deterministic per (seed, patch index).

## Test conditions (identical for every model)
`clean`  optical + SAR as observed.
`cloud50` optical NaN over 50% of each test patch (apply_clouds seed 2027), SAR untouched.
`cloud100` optical entirely missing, SAR only.
Models are trained on `clean` train patches only. No cloud augmentation in training for any model, so the
cloud rows measure out-of-the-box robustness.

## Results
Each model writes `results/<model>.json`:
{"model": str, "seeds": [int], "train_seconds": float, "conditions": {"clean": report, "cloud50": report, "cloud100": report}}
where report = output of `metrics.iou_report` averaged over seeds (mean and sd per number) plus per-seed values.
Main metric: IoU of the mangrove class, pooled over all test pixels. Also report mIoU (2 classes) and per-site IoU.
