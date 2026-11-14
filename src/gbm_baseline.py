"""Per-pixel gradient-boosting baseline (model name "pixel-gbm").

Features per pixel (15, raw values, no normalisation; trees are invariant to monotone scaling):
  B2 B3 B4 B5 B8 B8A B11 B12 (S2 reflectance), VV VH (S1 dB),
  NDVI = (B8-B4)/(B8+B4), NDWI = (B3-B8)/(B3+B8), MNDWI = (B3-B11)/(B3+B11), NDMI = (B8-B11)/(B8+B11), VH-VV.
Missing inputs stay NaN and propagate into the indices (a zero denominator also gives NaN);
sklearn HistGradientBoostingClassifier handles NaN natively, so cloud50 / cloud100 need no imputation.

Training: per seed, 300k labelled pixels from split "train" (clean), stratified so the class shares match the
true mangrove share of all labelled train pixels, drawn with numpy default_rng(seed). HGB max_iter 500,
learning_rate 0.1, early stopping on an internal 10% split (random_state = seed), other settings sklearn defaults.
The final model is validated on every labelled pixel of split "val" (val mIoU reported).
Test: split "test" under clean / cloud50 / cloud100 (clouds.apply_clouds, seed 2027, start_index = position of
the patch within the test split), predicted in chunks. Writes results/pixel-gbm.json (CONTRACT.md).

Usage:
    python -I src/gbm_baseline.py --seeds 0 1 2 [--threads 16] [--limit N] [--train-pixels 300000]
        [--patches-dir data/patches] [--results-dir results]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from clouds import apply_clouds  # noqa: E402
from metrics import IGNORE, ConfusionAccumulator, summarize_seeds  # noqa: E402
from sites import FLORO_OPTICAL  # noqa: E402

MODEL_NAME = "pixel-gbm"
CLOUD_SEED = 2027
CONDITIONS = {"clean": 0.0, "cloud50": 0.5, "cloud100": 1.0}
FEATURES = FLORO_OPTICAL + ["VV", "VH", "NDVI", "NDWI", "MNDWI", "NDMI", "VH_minus_VV"]
B3, B4, B8, B11 = (FLORO_OPTICAL.index(b) for b in ("B3", "B4", "B8", "B11"))
FEATURE_COLS = None  # --sar-only sets [8, 9, 14] (VV, VH, VH-VV): a radar-only reference for the cloud100 condition
INPUT_DESC = ("per-pixel 15 features: raw S2 B2 B3 B4 B5 B8 B8A B11 B12 reflectance, S1 VV VH dB, "
              "NDVI, NDWI(B3,B8), MNDWI(B3,B11), NDMI(B8,B11), VH-VV; NaN kept (HGB native missing-value handling)")


# ----------------------------------------------------------------------------------------------- data
def read_meta(patches_dir):
    with open(os.path.join(patches_dir, "meta.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["idx"] = int(r["idx"])
    rows.sort(key=lambda r: r["idx"])
    return rows


def split_rows(meta, split, limit=None):
    rows = [r for r in meta if r["split"] == split]
    return rows[:limit] if limit else rows


def load_arrays(patches_dir):
    opt = np.load(os.path.join(patches_dir, "optical.npy"), mmap_mode="r")
    sar = np.load(os.path.join(patches_dir, "sar.npy"), mmap_mode="r")
    lab = np.load(os.path.join(patches_dir, "label.npy"), mmap_mode="r")
    if not (opt.shape[0] == sar.shape[0] == lab.shape[0]):
        raise ValueError(f"patch arrays disagree on N: {opt.shape} {sar.shape} {lab.shape}")
    return opt, sar, lab


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        den = a + b
        return np.where(den != 0, (a - b) / den, np.nan).astype(np.float32)


def pixel_features(opt_px, sar_px):
    """opt_px (n, 8), sar_px (n, 2) float32 with NaN -> (n, 15) float32, NaN propagated."""
    opt_px = np.asarray(opt_px, np.float32)
    sar_px = np.asarray(sar_px, np.float32)
    f = np.empty((opt_px.shape[0], len(FEATURES)), np.float32)
    f[:, 0:8] = opt_px
    f[:, 8:10] = sar_px
    f[:, 10] = _nd(opt_px[:, B8], opt_px[:, B4])    # NDVI
    f[:, 11] = _nd(opt_px[:, B3], opt_px[:, B8])    # NDWI (McFeeters)
    f[:, 12] = _nd(opt_px[:, B3], opt_px[:, B11])   # MNDWI
    f[:, 13] = _nd(opt_px[:, B8], opt_px[:, B11])   # NDMI
    f[:, 14] = sar_px[:, 1] - sar_px[:, 0]          # VH - VV (dB)
    return f if FEATURE_COLS is None else f[:, FEATURE_COLS]


def patch_pixels(opt, sar, mask):
    """opt (C, H, W), sar (2, H, W), mask (H, W) bool -> features of the masked pixels in row-major order."""
    return pixel_features(np.asarray(opt, np.float32)[:, mask].T, np.asarray(sar, np.float32)[:, mask].T)


def sample_train(arrays, rows, n_total, seed):
    """Stratified pixel sample: per class, draw round(n_total * class share) labelled pixels without replacement."""
    opt, sar, lab = arrays
    idx = np.array([r["idx"] for r in rows], np.int64)
    labels = np.asarray(lab[idx])  # (n, H, W) uint8
    hw = labels.shape[1] * labels.shape[2]
    flat = labels.reshape(-1)
    counts = np.array([(flat == 0).sum(), (flat == 1).sum()], np.int64)
    n_lab = int(counts.sum())
    n_total = min(n_total, n_lab)
    share1 = counts[1] / n_lab if n_lab else 0.0
    want = np.array([n_total - int(round(n_total * share1)), int(round(n_total * share1))])
    want = np.minimum(want, counts)
    rng = np.random.default_rng(seed)
    picks = []
    for c in (0, 1):
        pos = np.flatnonzero(flat == c)
        picks.append(rng.choice(pos, size=int(want[c]), replace=False) if want[c] else pos[:0])
    pick = np.sort(np.concatenate(picks))
    p_of = pick // hw
    X = np.empty((len(pick), len(FEATURES) if FEATURE_COLS is None else len(FEATURE_COLS)), np.float32)
    y = flat[pick].astype(np.int64)
    bounds = np.searchsorted(p_of, np.arange(len(idx) + 1))
    for k in range(len(idx)):
        a, b = bounds[k], bounds[k + 1]
        if a == b:
            continue
        m = np.zeros(hw, bool)
        m[pick[a:b] - k * hw] = True
        m = m.reshape(labels.shape[1:])
        X[a:b] = patch_pixels(opt[idx[k]], sar[idx[k]], m)  # row-major order == sorted pick order
    info = {"pixels": int(len(pick)), "mangrove_share_sample": float(y.mean()) if len(y) else float("nan"),
            "mangrove_share_train": float(share1), "labelled_train_pixels": n_lab}
    return X, y, info


def predict_split(model, arrays, rows, cover, chunk=8):
    """Confusion over the labelled pixels of `rows`; clouds (cover > 0) use the position within the split."""
    opt, sar, lab = arrays
    idx = np.array([r["idx"] for r in rows], np.int64)
    acc = ConfusionAccumulator()
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        o = np.asarray(opt[ii], np.float32)
        if cover > 0:
            o = apply_clouds(o, cover, seed=CLOUD_SEED, start_index=s)
        sa = np.asarray(sar[ii], np.float32)
        lb = np.asarray(lab[ii])
        masks = lb != IGNORE
        feats = [patch_pixels(o[j], sa[j], masks[j]) for j in range(len(ii))]
        sizes = [len(f) for f in feats]
        if sum(sizes) == 0:
            continue
        pred = model.predict(np.concatenate(feats)).astype(np.uint8)
        off = 0
        for j, n in enumerate(sizes):
            acc.update(pred[off:off + n], lb[j][masks[j]], rows[s + j]["site"])
            off += n
    return acc.report()


# ------------------------------------------------------------------------------------------------ io
def json_clean(o):
    if isinstance(o, dict):
        return {k: json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def scalar_summary(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], np.float64)
    return {"mean": float(v.mean()) if v.size else float("nan"),
            "sd": float(v.std(ddof=1)) if v.size > 1 else (0.0 if v.size else float("nan")),
            "per_seed": [float(x) for x in values]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="use at most N patches per split (smoke tests)")
    ap.add_argument("--train-pixels", type=int, default=300_000)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--patches-dir", default=os.path.join(ROOT, "data", "patches"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--sar-only", action="store_true",
                    help="train and test on VV, VH, VH-VV only (model name pixel-gbm-sar); a radar-only reference")
    args = ap.parse_args()
    global MODEL_NAME, FEATURE_COLS, INPUT_DESC
    if args.sar_only:
        MODEL_NAME, FEATURE_COLS = "pixel-gbm-sar", [8, 9, 14]
        INPUT_DESC = "per-pixel 3 features: S1 VV, VH (dB), VH-VV; optical ignored (radar-only reference)"

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    meta = read_meta(args.patches_dir)
    arrays = load_arrays(args.patches_dir)
    if meta and max(r["idx"] for r in meta) >= arrays[0].shape[0]:
        raise ValueError("meta.csv idx exceeds patch array length")
    tr_rows = split_rows(meta, "train", args.limit)
    va_rows = split_rows(meta, "val", args.limit)
    te_rows = split_rows(meta, "test", args.limit)
    if not (tr_rows and va_rows and te_rows):
        raise SystemExit(f"empty split: train {len(tr_rows)} val {len(va_rows)} test {len(te_rows)}")
    log(f"patches train {len(tr_rows)} val {len(va_rows)} test {len(te_rows)}; threads {args.threads}")

    reports = {c: [] for c in CONDITIONS}
    seed_info = []
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            t = time.time()
            X, y, sinfo = sample_train(arrays, tr_rows, args.train_pixels, seed)
            t_sample = time.time() - t
            log(f"seed {seed} sampled {sinfo['pixels']:,} px (mangrove share {sinfo['mangrove_share_sample']:.4f} "
                f"vs train {sinfo['mangrove_share_train']:.4f}) in {t_sample:.1f}s")
            model = HistGradientBoostingClassifier(
                max_iter=args.max_iter, learning_rate=args.learning_rate, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=10, random_state=seed)
            t = time.time()
            model.fit(X, y)
            t_fit = time.time() - t
            del X, y
            t = time.time()
            val = predict_split(model, arrays, va_rows, 0.0)
            t_val = time.time() - t
            log(f"seed {seed} fit {t_fit:.1f}s n_iter {model.n_iter_} val_miou {val['pooled']['miou']:.4f} "
                f"val_iou_m {val['pooled']['iou_mangrove']:.4f} ({t_val:.1f}s)")
            t = time.time()
            for cond, cover in CONDITIONS.items():
                rep = predict_split(model, arrays, te_rows, cover)
                reports[cond].append(rep)
                log(f"seed {seed} test {cond} iou_m {rep['pooled']['iou_mangrove']:.4f} "
                    f"miou {rep['pooled']['miou']:.4f}")
            t_test = time.time() - t
            seed_info.append({"seed": seed, "train_seconds": t_sample + t_fit, "sample_seconds": t_sample,
                              "fit_seconds": t_fit, "n_iter": int(model.n_iter_), "val_seconds": t_val,
                              "test_seconds": t_test, "val_miou": val["pooled"]["miou"],
                              "val_iou_mangrove": val["pooled"]["iou_mangrove"], "sample": sinfo})

    result = {
        "model": MODEL_NAME,
        "seeds": list(args.seeds),
        "train_seconds": float(np.mean([i["train_seconds"] for i in seed_info])),
        "train_seconds_note": "mean wall-clock seconds per seed for pixel sampling + HGB fit; per seed below",
        "conditions": {c: summarize_seeds(reports[c]) for c in CONDITIONS},
        "val": {"miou": scalar_summary([i["val_miou"] for i in seed_info]),
                "iou_mangrove": scalar_summary([i["val_iou_mangrove"] for i in seed_info])},
        "per_seed": seed_info,
        "features": FEATURES if FEATURE_COLS is None else [FEATURES[i] for i in FEATURE_COLS],
        "input": INPUT_DESC,
        "config": {"estimator": "sklearn HistGradientBoostingClassifier", "max_iter": args.max_iter,
                   "learning_rate": args.learning_rate, "early_stopping": True, "validation_fraction": 0.1,
                   "n_iter_no_change": 10, "train_pixels": args.train_pixels,
                   "sampling": "stratified to the true class share of labelled train pixels, default_rng(seed)",
                   "threads": args.threads, "limit": args.limit, "cloud_seed": CLOUD_SEED,
                   "n_patches": {"train": len(tr_rows), "val": len(va_rows), "test": len(te_rows)}},
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{MODEL_NAME}.json")
    with open(path, "w") as f:
        json.dump(json_clean(result), f, indent=1, allow_nan=False)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
