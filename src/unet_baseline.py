"""U-Net baseline trained from scratch (model name "unet").

Input: 12 channels per pixel
  0-7   Sentinel-2 B2 B3 B4 B5 B8 B8A B11 B12, z-scored with train-split mean/std from stats.json
  8-9   Sentinel-1 VV, VH (dB), z-scored with train-split mean/std from stats.json
  10    optical validity mask (1 where all 8 optical bands are finite)
  11    SAR validity mask (1 where both SAR bands are finite)
NaN is replaced by 0 after z-scoring (0 = the train mean), so a cloud reads as "mean value, mask 0".

Model: classic U-Net (Ronneberger et al. 2015) with padding, 4 down levels, base width 32 (32-64-128-256-512),
two 3x3 conv + BatchNorm + ReLU per block, transposed-conv upsampling, 1x1 head to 2 classes.
7,765,666 trainable parameters (12 input channels).

Training: split "train" (clean) only, one random view per batch from {identity, hflip, vflip, rot180}
(the same 4 views the frozen-encoder heads see), cross-entropy with label 255 ignored, AdamW lr 1e-3 wd 0.01,
batch 8, up to 60 epochs, early stopping with patience 10 on val-split mIoU (best epoch restored).
Test: split "test" under clean / cloud50 / cloud100 (clouds.apply_clouds, seed 2027, start_index = position of
the patch within the test split). Writes results/unet.json (CONTRACT.md).

Usage:
    python -I src/unet_baseline.py --seeds 0 1 2 [--threads 16] [--limit N] [--epochs 60]
        [--patches-dir data/patches] [--results-dir results]
"""
import argparse
import copy
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from clouds import apply_clouds  # noqa: E402
from metrics import IGNORE, ConfusionAccumulator, summarize_seeds  # noqa: E402
from sites import FLORO_OPTICAL  # noqa: E402

MODEL_NAME = "unet"
CLOUD_SEED = 2027
CONDITIONS = {"clean": 0.0, "cloud50": 0.5, "cloud100": 1.0}
SAR_NAMES = ["VV", "VH"]
IN_CHANNELS = 12
INPUT_DESC = ("12 channels: S2 B2 B3 B4 B5 B8 B8A B11 B12 reflectance and S1 VV VH dB, each z-scored with "
              "train-split mean/std (stats.json); NaN -> 0 after z-scoring; plus optical validity mask "
              "(all 8 bands finite) and SAR validity mask (VV and VH finite)")


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


def load_norm(patches_dir):
    with open(os.path.join(patches_dir, "stats.json")) as f:
        st = json.load(f)
    om = np.array([st["optical"][b]["mean"] for b in FLORO_OPTICAL], np.float32)
    osd = np.array([st["optical"][b]["std"] for b in FLORO_OPTICAL], np.float32)
    sm = np.array([st["sar"][b]["mean"] for b in SAR_NAMES], np.float32)
    ssd = np.array([st["sar"][b]["std"] for b in SAR_NAMES], np.float32)
    return om, osd, sm, ssd


def make_input(opt, sar, norm):
    """opt (n, 8, H, W), sar (n, 2, H, W) with NaN -> float32 (n, 12, H, W)."""
    om, osd, sm, ssd = norm
    opt = np.asarray(opt, np.float32)
    sar = np.asarray(sar, np.float32)
    n, _, h, w = opt.shape
    x = np.empty((n, IN_CHANNELS, h, w), np.float32)
    x[:, 0:8] = (opt - om[None, :, None, None]) / osd[None, :, None, None]
    x[:, 8:10] = (sar - sm[None, :, None, None]) / ssd[None, :, None, None]
    x[:, 10] = np.isfinite(opt).all(axis=1)
    x[:, 11] = np.isfinite(sar).all(axis=1)
    np.nan_to_num(x[:, 0:10], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def build_split(arrays, rows, norm, chunk=16):
    opt, sar, lab = arrays
    idx = np.array([r["idx"] for r in rows], np.int64)
    h, w = lab.shape[1:]
    x = np.empty((len(idx), IN_CHANNELS, h, w), np.float32)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        x[s:s + len(ii)] = make_input(opt[ii], sar[ii], norm)
    y = np.asarray(lab[idx]).astype(np.int64)
    return x, y


# ---------------------------------------------------------------------------------------------- model
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.conv = DoubleConv(cin // 2 + cout, cout)

    def forward(self, x, skip):
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch=IN_CHANNELS, n_classes=2, base=32, depth=4):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]  # 32 64 128 256 512
        self.inc = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(widths[i], widths[i + 1])) for i in range(depth))
        self.ups = nn.ModuleList(Up(widths[i + 1], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        x = skips.pop()
        for u in self.ups:
            x = u(x, skips.pop())
        return self.head(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


AUG_DIMS = {0: None, 1: [-1], 2: [-2], 3: [-2, -1]}  # identity, hflip (W), vflip (H), rot180


def augment(x, y, k):
    dims = AUG_DIMS[k]
    if dims is None:
        return x, y
    return torch.flip(x, dims), torch.flip(y, dims)


# ------------------------------------------------------------------------------------------ train/eval
@torch.no_grad()
def predict(model, x, batch=8):
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[s:s + batch]))
        out.append(model(xb).argmax(1).to(torch.uint8).numpy())
    return np.concatenate(out)


def evaluate(model, x, y, sites, batch=8):
    pred = predict(model, x, batch)
    acc = ConfusionAccumulator()
    for p, l, s in zip(pred, y, sites):
        acc.update(p, l, s)
    return acc


def train_one(seed, data, args, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr, xva, yva, va_sites = data
    model = UNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best = {"miou": -math.inf, "epoch": -1, "state": None, "val": None}
    epoch_secs, train_secs = [], []
    t0 = time.time()
    since_best = 0
    epochs_run = 0
    for ep in range(1, args.epochs + 1):
        te = time.time()
        model.train()
        perm = rng.permutation(len(xtr))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), args.batch):
            bi = np.sort(perm[s:s + args.batch])
            xb = torch.from_numpy(xtr[bi])
            yb = torch.from_numpy(ytr[bi])
            xb, yb = augment(xb, yb, int(rng.integers(4)))
            if not (yb != IGNORE).any():
                continue
            loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        t_train = time.time() - te
        train_secs.append(t_train)
        val = evaluate(model, xva, yva, va_sites, args.batch).report()
        vm = val["pooled"]["miou"]
        epochs_run = ep
        epoch_secs.append(time.time() - te)
        improved = np.isfinite(vm) and vm > best["miou"]
        if improved:
            best.update(miou=vm, epoch=ep, state=copy.deepcopy(model.state_dict()), val=val)
            since_best = 0
        else:
            since_best += 1
        log(f"seed {seed} epoch {ep:3d} loss {tot / max(nb, 1):.4f} val_miou {vm:.4f} "
            f"val_iou_m {val['pooled']['iou_mangrove']:.4f} train {t_train:.1f}s total {epoch_secs[-1]:.1f}s"
            f"{' *' if improved else ''}")
        if since_best >= args.patience:
            log(f"seed {seed} early stop at epoch {ep} (best epoch {best['epoch']})")
            break
    train_seconds = time.time() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    info = {
        "seed": seed, "train_seconds": train_seconds, "epochs_run": epochs_run, "best_epoch": best["epoch"],
        "seconds_per_epoch": float(np.mean(epoch_secs)) if epoch_secs else float("nan"),
        "train_only_seconds_per_epoch": float(np.mean(train_secs)) if train_secs else float("nan"),
        "val_miou": best["miou"] if best["val"] else float("nan"),
        "val_iou_mangrove": best["val"]["pooled"]["iou_mangrove"] if best["val"] else float("nan"),
    }
    return model, info


def test_conditions(model, arrays, rows, norm, batch=8, chunk=16):
    """Returns {cond: metrics report} on the test split. Clouds use the patch position within the test split."""
    opt, sar, lab = arrays
    idx = np.array([r["idx"] for r in rows], np.int64)
    sites = [r["site"] for r in rows]
    out = {}
    for cond, cover in CONDITIONS.items():
        acc = ConfusionAccumulator()
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            o = np.asarray(opt[ii], np.float32)
            if cover > 0:
                o = apply_clouds(o, cover, seed=CLOUD_SEED, start_index=s)
            x = make_input(o, sar[ii], norm)
            pred = predict(model, x, batch)
            for p, l, site in zip(pred, np.asarray(lab[ii]), sites[s:s + len(ii)]):
                acc.update(p, l, site)
        out[cond] = acc.report()
    return out


# ------------------------------------------------------------------------------------------------ io
def json_clean(o):
    """NaN/inf -> None so the file is strict JSON."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="use at most N patches per split (smoke tests)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patches-dir", default=os.path.join(ROOT, "data", "patches"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    meta = read_meta(args.patches_dir)
    arrays = load_arrays(args.patches_dir)
    if meta and max(r["idx"] for r in meta) >= arrays[0].shape[0]:
        raise ValueError("meta.csv idx exceeds patch array length")
    norm = load_norm(args.patches_dir)
    tr_rows = split_rows(meta, "train", args.limit)
    va_rows = split_rows(meta, "val", args.limit)
    te_rows = split_rows(meta, "test", args.limit)
    if not (tr_rows and va_rows and te_rows):
        raise SystemExit(f"empty split: train {len(tr_rows)} val {len(va_rows)} test {len(te_rows)}")
    log(f"patches train {len(tr_rows)} val {len(va_rows)} test {len(te_rows)}; threads {args.threads}")

    t = time.time()
    xtr, ytr = build_split(arrays, tr_rows, norm)
    xva, yva = build_split(arrays, va_rows, norm)
    log(f"loaded train+val into memory ({(xtr.nbytes + xva.nbytes) / 1e9:.2f} GB) in {time.time() - t:.1f}s")
    data = (xtr, ytr, xva, yva, [r["site"] for r in va_rows])

    params = n_params(UNet())
    log(f"U-Net parameters: {params:,}")

    reports = {c: [] for c in CONDITIONS}
    seed_info = []
    # Per-seed checkpoint (added 2026-11-13 after machine restarts lost runs): finished seeds are kept in
    # <results-dir>/_ckpt/unet.json and reused when the settings match.
    ckpt_path = os.path.join(args.results_dir, "_ckpt", f"{MODEL_NAME}.json")
    ckpt_key = {"epochs": args.epochs, "patience": args.patience, "batch": args.batch, "lr": args.lr, "wd": args.wd,
                "limit": args.limit, "patches_dir": os.path.abspath(args.patches_dir)}
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ck = json.load(f)
        if ck.get("key") == ckpt_key:
            done = ck.get("seeds", {})
            log(f"checkpoint {ckpt_path}: reusing finished seeds {sorted(int(k) for k in done)}")
    for seed in args.seeds:
        if str(seed) in done:
            info, rep = done[str(seed)]["info"], done[str(seed)]["reports"]
            for c in CONDITIONS:
                reports[c].append(rep[c])
            seed_info.append(info)
            log(f"seed {seed}: reused from checkpoint")
            continue
        model, info = train_one(seed, data, args, log)
        te = time.time()
        rep = test_conditions(model, arrays, te_rows, norm, args.batch)
        info["test_seconds"] = time.time() - te
        done[str(seed)] = {"info": info, "reports": rep}
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path + ".tmp", "w") as f:
            json.dump(json_clean({"key": ckpt_key, "seeds": done}), f, allow_nan=False)
        os.replace(ckpt_path + ".tmp", ckpt_path)
        for c in CONDITIONS:
            reports[c].append(rep[c])
        seed_info.append(info)
        log(f"seed {seed} test " + " ".join(
            f"{c} iou_m {rep[c]['pooled']['iou_mangrove']:.4f} miou {rep[c]['pooled']['miou']:.4f}"
            for c in CONDITIONS) + f" ({info['test_seconds']:.1f}s)")

    result = {
        "model": MODEL_NAME,
        "seeds": list(args.seeds),
        "train_seconds": float(np.mean([i["train_seconds"] for i in seed_info])),
        "train_seconds_note": "mean wall-clock seconds to train one seed (incl. per-epoch val); per seed below",
        "conditions": {c: summarize_seeds(reports[c]) for c in CONDITIONS},
        "val": {"miou": summarize_seeds_scalar([i["val_miou"] for i in seed_info]),
                "iou_mangrove": summarize_seeds_scalar([i["val_iou_mangrove"] for i in seed_info])},
        "per_seed": seed_info,
        "params": params,
        "input": INPUT_DESC,
        "architecture": "U-Net, 4 down levels, widths 32-64-128-256-512, 2x(conv3x3-BN-ReLU) per block, "
                        "maxpool down, ConvTranspose2d up, 1x1 head to 2 classes",
        "config": {"epochs_max": args.epochs, "patience": args.patience, "batch": args.batch, "lr": args.lr,
                   "weight_decay": args.wd, "optimizer": "AdamW", "loss": "cross-entropy, ignore 255",
                   "augmentation": "one of identity/hflip/vflip/rot180 per batch, uniform",
                   "selection": "best val-split pooled mIoU", "threads": args.threads, "limit": args.limit,
                   "cloud_seed": CLOUD_SEED,
                   "n_patches": {"train": len(tr_rows), "val": len(va_rows), "test": len(te_rows)}},
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{MODEL_NAME}.json")
    with open(path, "w") as f:
        json.dump(json_clean(result), f, indent=1, allow_nan=False)
    log(f"wrote {path}")


def summarize_seeds_scalar(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], np.float64)
    return {"mean": float(v.mean()) if v.size else float("nan"),
            "sd": float(v.std(ddof=1)) if v.size > 1 else (0.0 if v.size else float("nan")),
            "per_seed": [float(x) for x in values]}


if __name__ == "__main__":
    main()
