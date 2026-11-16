"""Train and evaluate the UPerNet head (src/head.py) on frozen, cached encoder features.

Protocol (PANGAEA frozen-encoder evaluation, sized for CPU):
  features  data/features/<model>_clean_train.npy  (A=4, N, L=4, C, G, G) float16, augmentations in the fixed order
            0 identity, 1 hflip (W axis), 2 vflip (H axis), 3 rot180 (both); the label of pair (a, i) is label i with
            the same transform. Val/test arrays are (N, L, C, G, G). N follows meta.csv order (split rows sorted by idx).
  labels    data/patches/label.npy (rows indexed by meta.csv idx), 255 = ignore.
  training  every epoch visits all A*N (aug, patch) pairs once in a seeded shuffle; cross-entropy (ignore 255);
            AdamW (wd 0.01) with cosine LR decay per step from --lr to 0 over --epochs (--schedule constant keeps it
            flat); batches stream from the memmap with a one-thread prefetch, so the train array is never loaded whole.
  selection val mIoU (metrics.py, pooled over val pixels, 2 classes) after every epoch; best weights are kept and
            training stops after --patience epochs without improvement. Epoch 0 (untrained head) and an all-"other"
            prediction are logged as chance references; the restored best weights are re-scored on val as a check.
  test      best weights on every data/features/<model>_<cond>_test.npy that exists (clean, cloud50, cloud100);
            missing conditions are skipped and listed under "conditions_missing".
  output    results/<model>.json per CONTRACT "Results" (metrics.summarize_seeds per condition) plus train_seconds,
            best epoch, val mIoU per seed and the head's parameter count. NaN scores are written as null.
  checks    before any training, every feature file must have its sidecar, whose model/cond/split/shape match the
            file; the sidecar idx list (FLORO) must equal the meta.csv split rows, or else its n_split (TerraMind,
            Prithvi) their count; sidecar nonfinite_float16 > 0 is refused; every value used is scanned once and
            any NaN/inf stops the run (batches are checked again as they stream). A patches_dir other than
            --patches-dir, or a meta.csv/label.npy newer than the features, is a warning in the log and under
            "input_warnings" in the results.

Usage (from the repo root):
  python -I src/train_head.py --model floro-s2s1 --seeds 0 1 2 [--epochs 60] [--patience 10] [--batch 16]
         [--lr 1e-4] [--threads 16] [--limit N] [--features-dir D] [--patches-dir D] [--results-dir D]
--limit N keeps the first N patches of every split (smoke tests); such a run writes results/<model>.limitN.json so it
can never overwrite a real result.
"""
import argparse
import copy
import csv
import json
import math
import os
import queue
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from head import UPerNetHead
from metrics import IGNORE, ConfusionAccumulator, summarize_seeds

ROOT = Path(__file__).resolve().parent.parent
CONDS = ("clean", "cloud50", "cloud100")
AUGS = ("identity", "hflip", "vflip", "rot180")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------------- data
def aug_label(lab, a):
    """Same transform the feature extractor applied to the input image: 0 id, 1 flip W, 2 flip H, 3 both."""
    if a == 0:
        return lab
    if a == 1:
        return lab[..., :, ::-1]
    if a == 2:
        return lab[..., ::-1, :]
    if a == 3:
        return lab[..., ::-1, ::-1]
    raise ValueError(f"unknown augmentation {a}")


def read_meta(path):
    """{split: (idx array sorted ascending, site list)} from meta.csv."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["split"], []).append((int(r["idx"]), r["site"]))
    out = {}
    for split, items in rows.items():
        items.sort()
        out[split] = (np.array([i for i, _ in items], dtype=np.int64), [s for _, s in items])
    return out


def open_features(path):
    """Memmap of a feature file plus its sidecar .json, which is required (CONTRACT): without it nothing ties the
    rows to meta.csv beyond their count."""
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.float16:
        log(f"  warning: {path.name} is {arr.dtype}, contract says float16 (used as is)")
    side = path.with_suffix(".json")
    if not side.exists():
        raise SystemExit(f"{path.name}: sidecar {side.name} missing (extraction unfinished?); refusing to pair "
                         f"unverified features with labels")
    try:
        sidecar = json.loads(side.read_text())
    except ValueError as e:
        raise SystemExit(f"{path.name}: unreadable sidecar {side.name}: {e}")
    if not isinstance(sidecar, dict):
        raise SystemExit(f"{path.name}: sidecar {side.name} is not a JSON object")
    return arr, sidecar


def nonfinite(a):
    """Number of NaN/inf values (a float16 overflow, |x| > 65504, is stored as inf)."""
    return int(a.size - np.count_nonzero(np.isfinite(a)))


class Split:
    """Features (memmap), labels (in RAM, uint8), meta idx and site names of one split, in meta.csv idx order."""

    def __init__(self, name, feats, labels, idx, sites, sidecar, path, checks):
        self.name, self.feats, self.labels, self.idx, self.sites = name, feats, labels, idx, sites
        self.sidecar, self.path, self.checks = sidecar, path, checks


def check_sidecar(path, side, expect, shape, name, idx, sites, patches_dir, meta_files):
    """Ties the feature rows to the current meta.csv. Hard errors for anything that proves a mismatch; warnings
    (returned, logged and written to the results) for signs that cannot be proved. idx/sites are the FULL split."""
    for k, v in expect.items():  # model, cond, split as encoded in the file name
        if side.get(k) != v:
            raise SystemExit(f"{path.name}: sidecar {k} = {side.get(k)!r}, file name says {v!r}")
    if list(side.get("shape") or ()) != list(shape):
        raise SystemExit(f"{path.name}: sidecar shape {side.get('shape')} but the array is {list(shape)} "
                         f"(sidecar left over from another extraction?)")
    if (side.get("nonfinite_float16") or 0) > 0:
        raise SystemExit(f"{path.name}: sidecar reports {side['nonfinite_float16']} non-finite values "
                         f"(float16 overflow or NaN in the encoder output); fix the extraction first")
    warnings = []
    if side.get("idx") is not None:
        s_idx = np.asarray(side["idx"], dtype=np.int64)
        if s_idx.shape != idx.shape or not np.array_equal(s_idx, idx):
            diff = (int(np.argmax(s_idx != idx)) if s_idx.shape == idx.shape
                    else min(len(s_idx), len(idx)))
            raise SystemExit(f"{path.name}: sidecar idx list ({len(s_idx)} rows) is not meta.csv's '{name}' idx list "
                             f"({len(idx)} rows; first difference at position {diff}, "
                             f"{len(np.setdiff1d(idx, s_idx))} meta rows not in the features): the features were "
                             f"extracted from another patch set, re-extract them")
        membership = "idx list matches meta.csv"
    elif side.get("n_split") is not None:
        if int(side["n_split"]) != len(idx):
            raise SystemExit(f"{path.name}: sidecar n_split {side['n_split']} but meta.csv has {len(idx)} '{name}' rows")
        membership = "count only (sidecar has n_split, no idx)"
    else:
        membership = "count only (sidecar has neither idx nor n_split)"
    if isinstance(side.get("sites"), dict):
        want = {}
        for s in sites:
            want[s] = want.get(s, 0) + 1
        if {str(k): int(v) for k, v in side["sites"].items()} != want:
            raise SystemExit(f"{path.name}: sidecar sites {side['sites']} but meta.csv '{name}' rows give {want}")
    if side.get("patches_dir") and (ROOT / side["patches_dir"]).resolve() != Path(patches_dir).resolve():
        warnings.append(f"extracted from patches_dir {side['patches_dir']}, labels read from {_rel(patches_dir)}")
    f_mtime = path.stat().st_mtime
    newer = [p.name for p in meta_files if p.stat().st_mtime > f_mtime]
    if newer:
        warnings.append(f"{' and '.join(newer)} newer than the features (patches rebuilt after extraction?)")
    for w in warnings:
        log(f"  WARNING {path.name}: {w}")
    return {"membership": membership, "warnings": warnings}


def scan_finite(path, feats, idx, train, chunk=8):
    """Read every value the trainer will use once, before any training, and refuse non-finite ones."""
    t0, bad, total = time.perf_counter(), [], 0
    for a in range(feats.shape[0] if train else 1):
        arr = feats[a] if train else feats
        for s in range(0, arr.shape[0], chunk):
            x = np.asarray(arr[s:s + chunk])
            if nonfinite(x):
                for k in range(x.shape[0]):
                    c = nonfinite(x[k])
                    if c:
                        total += c
                        bad.append(f"{'aug %d ' % a if train else ''}position {s + k} (meta idx {idx[s + k]}): {c}")
    if bad:
        raise SystemExit(f"{path.name}: {total} non-finite values in {len(bad)} patch(es): " + "; ".join(bad[:5])
                         + (" ..." if len(bad) > 5 else ""))
    return round(time.perf_counter() - t0, 2)


def load_split(name, feat_path, meta, label_mm, limit, model, cond, patches_dir, train=False):
    if name not in meta:
        raise SystemExit(f"meta.csv has no '{name}' rows")
    idx, sites = meta[name]
    feats, sidecar = open_features(feat_path)
    n_axis = 1 if train else 0
    want_ndim = 6 if train else 5
    if feats.ndim != want_ndim:
        raise SystemExit(f"{feat_path.name}: shape {feats.shape}, expected {want_ndim} dims "
                         f"({'(A, N, L, C, G, G)' if train else '(N, L, C, G, G)'})")
    if train and feats.shape[0] != len(AUGS):
        raise SystemExit(f"{feat_path.name}: A = {feats.shape[0]}, expected {len(AUGS)} augmentations {AUGS}")
    if feats.shape[n_axis] != len(idx):
        raise SystemExit(f"{feat_path.name}: {feats.shape[n_axis]} patches but meta.csv has {len(idx)} '{name}' rows")
    checks = check_sidecar(feat_path, sidecar, {"model": model, "cond": cond, "split": name}, feats.shape, name,
                           idx, sites, patches_dir, [patches_dir / "meta.csv", patches_dir / "label.npy"])
    if limit:
        idx, sites = idx[:limit], sites[:limit]
        feats = feats[:, :limit] if train else feats[:limit]
    checks["finite_scan_seconds"] = scan_finite(feat_path, feats, idx, train)
    if idx.max(initial=-1) >= label_mm.shape[0]:
        raise SystemExit(f"meta.csv idx {idx.max()} outside label.npy ({label_mm.shape[0]} rows)")
    labels = np.ascontiguousarray(label_mm[idx])
    return Split(name, feats, labels, idx, sites, sidecar, feat_path, checks)


def _prefetch(it, depth=2):
    """Run a batch generator in a background thread (memmap reads overlap with torch compute)."""
    q = queue.Queue(maxsize=depth)
    done = object()
    err = []

    def run():
        try:
            for item in it:
                q.put(item)
        except BaseException as e:  # re-raised in the consumer
            err.append(e)
        finally:
            q.put(done)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is done:
            break
        yield item
    t.join()
    if err:
        raise err[0]


def train_batches(split, pairs, batch):
    """pairs: (M, 2) int array of (aug, patch). Yields (x float16 tensor, y int64 tensor)."""
    feats, labels = split.feats, split.labels
    shape = feats.shape[2:]
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        x = np.empty((len(chunk),) + shape, dtype=feats.dtype)
        y = np.empty((len(chunk),) + labels.shape[1:], dtype=np.int64)
        for k, (a, i) in enumerate(chunk):
            x[k] = feats[a, i]
            y[k] = aug_label(labels[i], a)
        if nonfinite(x):  # already scanned at load; this guards the bytes actually fed to the head
            k = next(k for k in range(len(chunk)) if nonfinite(x[k]))
            raise SystemExit(f"{split.path.name}: non-finite features at aug {chunk[k][0]} position {chunk[k][1]} "
                             f"(meta idx {split.idx[chunk[k][1]]})")
        yield torch.from_numpy(x), torch.from_numpy(y)


def eval_batches(split, batch):
    n = split.feats.shape[0]
    for s in range(0, n, batch):
        x = np.array(split.feats[s:s + batch])  # copy: memmap is read-only
        if nonfinite(x):
            k = next(k for k in range(x.shape[0]) if nonfinite(x[k]))
            raise SystemExit(f"{split.path.name}: non-finite features at position {s + k} "
                             f"(meta idx {split.idx[s + k]})")
        yield s, torch.from_numpy(x)


# --------------------------------------------------------------------------------------------------- train/eval
DEVICE = "cpu"  # set from --device in main(); "cuda" runs the head on a GPU


@torch.inference_mode()
def evaluate(model, split, batch):
    model.eval()
    acc = ConfusionAccumulator()
    for s, x in _prefetch(eval_batches(split, batch)):
        pred = model(x.to(DEVICE)).argmax(1).to(torch.uint8).cpu().numpy()
        for k in range(pred.shape[0]):
            acc.update(pred[k], split.labels[s + k], split.sites[s + k])
    return acc.report()


def make_epoch_pairs(n_aug, n, batch, rng):
    """All (aug, patch) pairs in a seeded shuffle. A trailing batch of 1 is dropped (BatchNorm needs >= 2)."""
    pairs = np.stack(np.meshgrid(np.arange(n_aug), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    pairs = pairs[rng.permutation(len(pairs))]
    if len(pairs) % batch == 1 and len(pairs) > 1:
        pairs = pairs[:-1]
    return pairs


def lr_factor(step, total, schedule):
    if schedule == "constant":
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_seed(seed, args, train, val, in_ch):
    set_seed(seed)
    out_size = tuple(train.labels.shape[1:])
    model = UPerNetHead(in_ch, channels=args.channels, neck=args.neck, out_size=out_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    n_aug, n = train.feats.shape[:2]
    steps_per_epoch = math.ceil(len(make_epoch_pairs(n_aug, n, args.batch, np.random.default_rng(0))) / args.batch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    best = {"miou": -1.0, "epoch": 0, "state": None, "report": None}
    history, step, since_best = [], 0, 0
    t_start = time.perf_counter()
    rep0 = evaluate(model, val, args.eval_batch)  # untrained head: the chance reference for the val curve
    history.append({"epoch": 0, "train_loss": None, "val_miou": float(rep0["pooled"]["miou"]),
                    "val_iou_mangrove": float(rep0["pooled"]["iou_mangrove"]), "lr_end": 0.0,
                    "train_seconds": 0.0, "epoch_seconds": 0.0})
    log(f"  seed {seed} epoch   0  (untrained)    val mIoU {rep0['pooled']['miou']:.4f}  "
        f"IoU mangrove {rep0['pooled']['iou_mangrove']:.4f}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        pairs = make_epoch_pairs(n_aug, n, args.batch, rng)
        loss_sum, loss_n = 0.0, 0
        for x, y in _prefetch(train_batches(train, pairs, args.batch)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor(step, total_steps, args.schedule)
            step += 1
            valid = int((y != IGNORE).sum())
            if valid == 0:  # whole batch unlabelled: cross-entropy would be NaN
                continue
            loss = F.cross_entropy(model(x), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.shape[0]
            loss_n += x.shape[0]
        t_train = time.perf_counter() - t0
        rep = evaluate(model, val, args.eval_batch)
        miou, iou_m = rep["pooled"]["miou"], rep["pooled"]["iou_mangrove"]
        improved = np.isfinite(miou) and miou > best["miou"]
        if improved:
            best.update(miou=float(miou), epoch=epoch, state=copy.deepcopy(model.state_dict()), report=rep)
            since_best = 0
        else:
            since_best += 1
        sec = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": loss_sum / max(loss_n, 1), "val_miou": float(miou),
                        "val_iou_mangrove": float(iou_m), "lr_end": opt.param_groups[0]["lr"],
                        "train_seconds": round(t_train, 2), "epoch_seconds": round(sec, 2)})
        log(f"  seed {seed} epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  val mIoU {miou:.4f}  "
            f"IoU mangrove {iou_m:.4f}  {'*' if improved else ' '}  {t_train:.1f}+{sec - t_train:.1f} s")
        if since_best >= args.patience:
            log(f"  seed {seed}: early stop after epoch {epoch} (no val mIoU gain for {args.patience} epochs); "
                f"best epoch {best['epoch']} mIoU {best['miou']:.4f}")
            break
    train_seconds = time.perf_counter() - t_start
    if best["state"] is None:
        raise SystemExit(f"seed {seed}: val mIoU never finite; check val labels/features")
    model.load_state_dict(best["state"])
    recheck = evaluate(model, val, args.eval_batch)["pooled"]["miou"]  # restored weights reproduce the best score
    if not abs(recheck - best["miou"]) <= 1e-9:
        log(f"  warning: seed {seed} restored weights give val mIoU {recheck:.6f}, best epoch had {best['miou']:.6f}")
    epochs_run = len(history) - 1
    trained = [h["epoch_seconds"] for h in history[1:]]
    return model, {"seed": seed, "best_epoch": best["epoch"], "epochs_run": epochs_run,
                   "stopped_early": epochs_run < args.epochs, "val_miou": best["miou"], "val_miou_recheck": recheck,
                   "val_miou_untrained": history[0]["val_miou"],
                   "seconds_per_epoch": float(np.mean(trained)) if trained else None,
                   "val_iou_mangrove": float(best["report"]["pooled"]["iou_mangrove"]),
                   "train_seconds": round(train_seconds, 2), "history": history}


# ----------------------------------------------------------------------------------------------------------- io
def _clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else None
    return o


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1, allow_nan=False))
    os.replace(tmp, path)


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="feature prefix, e.g. floro-s2s1, floro-s2only, terramind-l")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--channels", type=int, default=128, help="UPerNet decoder channels (PANGAEA 512; CPU 128)")
    ap.add_argument("--neck", choices=("fused", "pangaea"), default="fused", help="see src/head.py")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="torch device for the head: cpu (default), cuda, or auto")
    ap.add_argument("--limit", type=int, default=0, help="first N patches of each split (testing only)")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "data" / "features")
    ap.add_argument("--patches-dir", type=Path, default=ROOT / "data" / "patches")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=None, help="results file (default <results-dir>/<model>.json)")
    args = ap.parse_args()
    if args.batch < 2:
        ap.error("--batch must be >= 2 (BatchNorm)")

    global DEVICE
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if DEVICE.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)

    fd, pd = args.features_dir, args.patches_dir
    meta = read_meta(pd / "meta.csv")
    label_mm = np.load(pd / "label.npy", mmap_mode="r")
    paths = {s: fd / f"{args.model}_clean_{s}.npy" for s in ("train", "val")}
    for s, p in paths.items():
        if not p.exists():
            raise SystemExit(f"missing {_rel(p)}")
    train = load_split("train", paths["train"], meta, label_mm, args.limit, args.model, "clean", pd, train=True)
    val = load_split("val", paths["val"], meta, label_mm, args.limit, args.model, "clean", pd)
    L, C, G = train.feats.shape[2:5]
    if L != 4:
        raise SystemExit(f"expected 4 encoder layers, got L = {L}")
    if val.feats.shape[1:] != train.feats.shape[2:]:
        raise SystemExit(f"val features {val.feats.shape[1:]} do not match train {train.feats.shape[2:]}")
    tests, missing = {}, []
    for cond in CONDS:
        p = fd / f"{args.model}_{cond}_test.npy"
        if not p.exists():
            missing.append(cond)
            log(f"test condition '{cond}': {_rel(p)} missing, skipped")
            continue
        t = load_split("test", p, meta, label_mm, args.limit, args.model, cond, pd)
        if t.feats.shape[1:] != train.feats.shape[2:]:
            raise SystemExit(f"{p.name}: features {t.feats.shape[1:]} do not match train {train.feats.shape[2:]}")
        tests[cond] = t

    for s in [train, val, *tests.values()]:
        log(f"input {s.path.name}: rows vs meta.csv: {s.checks['membership']}; all finite "
            f"(scanned in {s.checks['finite_scan_seconds']:.1f} s)")
    acc0 = ConfusionAccumulator()
    for lab_i in val.labels:  # trivial reference: predict "other" everywhere
        acc0.update(np.zeros_like(lab_i), lab_i)
    val_all_other = acc0.report()["pooled"]["miou"]
    n_params = UPerNetHead(C, channels=args.channels, neck=args.neck, out_size=train.labels.shape[1:]).n_params()
    log(f"model {args.model}: train {train.feats.shape[0]}x{train.feats.shape[1]} pairs, val {val.feats.shape[0]}, "
        f"test {', '.join(f'{c} {t.feats.shape[0]}' for c, t in tests.items()) or 'none'}; L={L} C={C} G={G}; "
        f"head params {n_params:,}; threads {args.threads}")

    per_seed, cond_reports, test_seconds = [], {c: [] for c in tests}, 0.0
    description = None
    # Per-seed checkpoint (added 2026-11-13 after two machine restarts lost multi-hour runs): every finished seed
    # is stored in results/_ckpt/<out-name>.json with a key of everything that determines its numbers; a rerun with
    # the same key reuses finished seeds instead of retraining them. Seeds are independent, so reuse is exact.
    out = args.out or args.results_dir / (f"{args.model}.limit{args.limit}.json" if args.limit else f"{args.model}.json")
    ckpt_path = args.results_dir / "_ckpt" / out.name
    ckpt_key = {"model": args.model, "features_dir": str(fd.resolve()), "patches_dir": str(pd.resolve()),
                "epochs": args.epochs, "patience": args.patience, "batch": args.batch, "eval_batch": args.eval_batch,
                "lr": args.lr, "weight_decay": args.weight_decay, "schedule": args.schedule,
                "channels": args.channels, "neck": args.neck, "limit": args.limit,
                **({"device": DEVICE} if DEVICE != "cpu" else {}),
                "inputs": {k: (s.sidecar or {}).get("created") for k, s in
                           [("train", train), ("val", val)] + [(c, t) for c, t in tests.items()]}}
    done = {}
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text())
        if ck.get("key") == ckpt_key:
            done = ck.get("seeds", {})
            log(f"checkpoint {_rel(ckpt_path)}: reusing finished seeds {sorted(int(k) for k in done)}")
        else:
            log(f"checkpoint {_rel(ckpt_path)} has a different key; ignoring it")
    for seed in args.seeds:
        if str(seed) in done and set(done[str(seed)]["reports"]) == set(tests):
            info, reps = done[str(seed)]["info"], done[str(seed)]["reports"]
            for cond in tests:
                cond_reports[cond].append(reps[cond])
            test_seconds += info["test_seconds"]
            per_seed.append(info)
            log(f"  seed {seed}: reused from checkpoint")
            continue
        model, info = train_seed(seed, args, train, val, C)
        description = model.describe()
        t0 = time.perf_counter()
        info["test"] = {}
        reps = {}
        for cond, t in tests.items():
            rep = evaluate(model, t, args.eval_batch)
            cond_reports[cond].append(rep)
            reps[cond] = rep
            info["test"][cond] = {"iou_mangrove": rep["pooled"]["iou_mangrove"], "miou": rep["pooled"]["miou"]}
            log(f"  seed {seed} test {cond:8s}  IoU mangrove {rep['pooled']['iou_mangrove']:.4f}  "
                f"mIoU {rep['pooled']['miou']:.4f}  per site "
                + ", ".join(f"{s} {r['iou_mangrove']:.3f}" for s, r in rep["per_site"].items()))
        info["test_seconds"] = round(time.perf_counter() - t0, 2)
        test_seconds += info["test_seconds"]
        per_seed.append(info)
        done[str(seed)] = {"info": info, "reports": reps}
        write_json(ckpt_path, {"key": ckpt_key, "seeds": done})
        log(f"  seed {seed}: checkpoint written to {_rel(ckpt_path)}")
    if description is None:
        description = UPerNetHead(C, channels=args.channels, neck=args.neck,
                                  out_size=train.labels.shape[1:]).describe()

    val_mious = [p["val_miou"] for p in per_seed]
    result = {
        "model": args.model,
        "seeds": list(args.seeds),
        "train_seconds": round(sum(p["train_seconds"] for p in per_seed), 2),
        "conditions": {c: summarize_seeds(r) for c, r in cond_reports.items()},
        "conditions_missing": missing,
        "best_epoch_per_seed": [p["best_epoch"] for p in per_seed],
        "val_miou_per_seed": val_mious,
        "val_miou": {"mean": float(np.mean(val_mious)),
                     "sd": float(np.std(val_mious, ddof=1)) if len(val_mious) > 1 else 0.0},
        "val_miou_references": {"all_other": val_all_other,
                                "untrained_head_per_seed": [p["val_miou_untrained"] for p in per_seed]},
        "n_params_head": n_params,
        "test_seconds": round(test_seconds, 2),
        "head": {
            "description": description,
            "decoder": "UPerNet (PPM 1,2,3,6 + FPN) on a Feature2Pyramid neck (x4 deconv, x2 deconv, identity, maxpool /2)",
            "decoder_channels": args.channels,
            "neck": args.neck,
            "encoder_frozen": True,
            "features": {"layers": int(L), "channels": int(C), "grid": int(G)},
            "output": f"2-class logits, bilinear upsample to {train.labels.shape[1]}x{train.labels.shape[2]}",
        },
        "training": {
            "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "schedule": ("cosine decay per step from lr to 0 over max epochs" if args.schedule == "cosine"
                         else "constant"),
            "loss": "cross-entropy, ignore_index 255, no class weights",
            "batch": args.batch, "epochs_max": args.epochs, "patience": args.patience,
            "early_stopping": "pooled val mIoU (metrics.py, 2 classes); best-epoch weights restored",
            "augmentations": list(AUGS), "epoch": "all A x N (aug, patch) pairs once, seeded shuffle",
            "threads": args.threads, "limit": args.limit or None, "device": DEVICE,
        },
        "inputs": {
            k: {"path": _rel(s.path), "shape": list(s.feats.shape), "checks": s.checks, "sidecar": s.sidecar}
            for k, s in [("train", train), ("val", val)] + [(f"test_{c}", t) for c, t in tests.items()]
        },
        "input_warnings": [f"{s.path.name}: {w}" for s in [train, val, *tests.values()] for w in s.checks["warnings"]],
        "per_seed": per_seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(out, result)
    log(f"wrote {_rel(out)}  (train {result['train_seconds']:.0f} s over {len(args.seeds)} seeds; "
        f"val mIoU {result['val_miou']['mean']:.4f}; best epochs {result['best_epoch_per_seed']})")


if __name__ == "__main__":
    main()
