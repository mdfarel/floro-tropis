"""Frozen TerraMind-1.0-large features in the shared feature-file layout (same layout as the FLORO extractor).

Model    ibm-esa-geospatial/TerraMind-1.0-large (Apache-2.0), terratorch 1.2.14 backbone `terramind_v1_large`
         = TerraMindViT: 24 transformer blocks, dim 1024, patch 16, raw-pixel ("untok") modality embeddings.
         Encoder only (the checkpoint's decoder is not loaded). Weights: weights/terramind-l/TerraMind_v1_large.pt
         (HF revision pinned below; downloaded on first use if missing). Runs in .venv-tm, never in .venv.
Input    modalities S2L2A + S1RTC, 256 x 256 px at 10 m -> 16 x 16 tokens per modality. Positional embeddings are
         TerraMind's fixed 2D sin-cos, built for the 16 x 16 grid (img_size=256), so nothing is interpolated.
         S2L2A: 12 bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 (TerraMind's pretraining order).
                Eight come from data/patches/optical.npy, B01 B06 B07 B09 from data/sites/<site>/s2extra.tif
                (src/download_extra.py) read at the patch window (meta.csv row/col, cross-checked with x0/y0).
                Values: DN = reflectance x 10000 (L2A DN scale; our composites are pre-2022 so no +1000 offset),
                standardised with TerraMind v1 pretraining mean/std (terratorch v1_pretraining_mean/std).
         S1RTC: VV, VH in dB (as in sar.npy), standardised with TerraMind v1 S1RTC mean/std.
         TerraMindViT does no normalisation itself; it is done here.
Missing  TerraMind is not availability-aware. A pixel is invalid for S2L2A when ANY of the 12 bands is NaN
         (never-clear pixels, simulated clouds); all 12 bands of that pixel are set to the band mean, i.e. 0 after
         standardisation. Same for S1RTC (VV or VH NaN -> both 0). cloud100 therefore feeds a constant S2L2A image
         (all zeros after standardisation) plus the real S1RTC; the S2L2A modality is not dropped.
Output   merge_method="mean": per token, the mean of the S2L2A and S1RTC token embeddings (terratorch's default,
         keeps C = 1024 like FLORO and Prithvi). Layers = zero-based block indices 5, 11, 17, 23 (PANGAEA's choice
         for 24-block ViT-L encoders); blocks 5/11/17 are raw residual-stream outputs, block 23 is after the
         encoder's final LayerNorm (terratorch behaviour, same convention as the FLORO encoder).

Feature files (contract shared by all encoders)
  data/features/<model>_<cond>_<split>.npy  float16 open_memmap
      val/test: (N, L=4, C=1024, G=16, G=16), N = patches of the split in meta.csv idx order
      train:    (A=4, N, L, C, G, G), A = identity, hflip (W axis), vflip (H axis), rot180, applied to the input
                image (optical and SAR) before encoding
  data/features/<model>_<cond>_<split>.json sidecar {"model","cond","split","shape","layers","seconds_per_sample",
      "input", ...}
  cond in clean / cloud50 / cloud100 (train and val: clean only). Clouds: clouds.apply_clouds(optical, cover,
  seed=2027, start_index=<position of the batch's first patch within the split>) on the full optical stack, so
  every model gets the same cloud masks.

Usage (from the repo root, in .venv-tm):
  python -I src/extract_terramind.py --split test --cond cloud50 [--batch 4] [--threads 16] [--limit K]
         [--out-dir data/features] [--patches-dir data/patches] [--sites-dir data/sites] [--weights FILE]
         [--overwrite]
--limit K encodes only the first K patches of the split (shape N = K; recorded in the sidecar) for tests.
Work goes to <stem>.npy.part with <stem>.progress.json after every batch; an interrupted run resumes where it
stopped, and the .npy appears by atomic rename once complete. A finished .npy with the expected shape is skipped
unless --overwrite is given (same behaviour as extract_floro.py).

The generic driver below (meta reading, clouds, augmentations, memmap, sidecar) is shared with
src/extract_prithvi.py, which imports it from this file.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clouds import apply_clouds  # noqa: E402
from sites import S2_BANDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PATCH = 256
COVER = {"clean": 0.0, "cloud50": 0.5, "cloud100": 1.0}
CLOUD_SEED = 2027
AUG_NAMES = ["identity", "hflip", "vflip", "rot180"]


# ====================================================================== shared driver


def _rel(p):
    """Path relative to the repository root when it lies inside it (keeps local paths out of sidecars)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def augment(x, a):
    """x: (B, C, H, W). 0 identity, 1 hflip (W = last axis), 2 vflip (H axis), 3 rot180 (both)."""
    if a == 0:
        return x
    if a == 1:
        return x[..., ::-1]
    if a == 2:
        return x[..., ::-1, :]
    if a == 3:
        return x[..., ::-1, ::-1]
    raise ValueError(a)


def standardise_fill(x, mean, std):
    """x: (B, C, H, W) float with NaN. Returns float32 (x - mean) / std with every channel of a pixel set to 0
    (= the band mean) where ANY channel of that pixel is non-finite, plus the (B, H, W) invalid mask."""
    mean = np.asarray(mean, np.float32)[None, :, None, None]
    std = np.asarray(std, np.float32)[None, :, None, None]
    invalid = ~np.isfinite(x).all(axis=1)
    z = (np.asarray(x, np.float32) - mean) / std
    z = np.where(invalid[:, None], np.float32(0), z).astype(np.float32)
    return np.ascontiguousarray(z), invalid


def tokens_to_grid(t, n_prefix=0):
    """(B, n_prefix + G*G, C) tokens in row-major patch order -> (B, C, G, G)."""
    t = t[:, n_prefix:]
    b, n, c = t.shape
    g = int(round(n ** 0.5))
    if g * g != n:
        raise ValueError(f"{n} tokens is not a square grid")
    return t.reshape(b, g, g, c).permute(0, 3, 1, 2)


def read_meta(patches_dir):
    with open(patches_dir / "meta.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("idx", "row", "col"):
            r[k] = int(r[k])
        for k in ("x0", "y0"):
            r[k] = float(r[k])
    return rows


class PatchSource:
    """Reads optical.npy / sar.npy rows by meta idx (memory-mapped)."""

    def __init__(self, patches_dir, n_meta_max):
        self.optical = np.load(patches_dir / "optical.npy", mmap_mode="r")
        self.sar = np.load(patches_dir / "sar.npy", mmap_mode="r")
        if self.optical.shape[1:] != (8, PATCH, PATCH) or self.sar.shape[1:] != (2, PATCH, PATCH):
            raise SystemExit(f"unexpected patch arrays {self.optical.shape} {self.sar.shape}")
        if n_meta_max >= self.optical.shape[0] or self.optical.shape[0] != self.sar.shape[0]:
            raise SystemExit("meta.csv idx out of range of optical.npy / sar.npy")

    def get(self, rows):
        idx = [r["idx"] for r in rows]
        return np.asarray(self.optical[idx], np.float32), np.asarray(self.sar[idx], np.float32)


class Encoder:
    """Interface the driver needs. Subclasses: TerraMindEncoder (below), PrithviEncoder (extract_prithvi.py)."""
    name = ""
    layers = []
    channels = 1024
    patch_size = 16
    uses_sar = True
    weights_info = {}

    def build(self, args):  # load the frozen model
        raise NotImplementedError

    def optical(self, rows, opt8, args):  # (B, Copt, H, W) optical stack the clouds are applied to (NaN = invalid)
        raise NotImplementedError

    def encode(self, optical, sar):  # numpy inputs (already augmented) -> torch (B, L, C, G, G) float32
        raise NotImplementedError

    def describe(self):  # sidecar "input" text
        raise NotImplementedError


def parse_args(enc, argv=None):
    p = argparse.ArgumentParser(description=f"Frozen {enc.name} features (see module docstring).")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--cond", default="clean", choices=list(COVER))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="encode only the first K patches of the split")
    p.add_argument("--out-dir", default="data/features")
    p.add_argument("--patches-dir", default="data/patches")
    p.add_argument("--sites-dir", default="data/sites")
    p.add_argument("--weights", default=None, help="local checkpoint file (default: weights/<model>/...)")
    p.add_argument("--overwrite", action="store_true", help="redo a finished output instead of skipping it")
    a = p.parse_args(argv)
    if a.split in ("train", "val") and a.cond != "clean":
        p.error("train and val features use cond=clean only")
    for k in ("out_dir", "patches_dir", "sites_dir"):
        v = Path(getattr(a, k))
        setattr(a, k, v if v.is_absolute() else ROOT / v)
    return a


def _write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run(enc, argv=None):
    """Encode one (split, cond). Work goes to <stem>.npy.part with <stem>.progress.json after every batch, so an
    interrupted run resumes; the .npy appears by atomic rename when complete. A finished .npy of the expected
    shape is skipped unless --overwrite."""
    a = parse_args(enc, argv)
    import torch

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    all_rows = read_meta(a.patches_dir)
    rows = sorted((r for r in all_rows if r["split"] == a.split), key=lambda r: r["idx"])
    if not rows:
        raise SystemExit(f"no patches with split={a.split} in {a.patches_dir / 'meta.csv'}")
    n_total = len(rows)
    if a.limit is not None:
        rows = rows[: a.limit]
    src = PatchSource(a.patches_dir, max(r["idx"] for r in all_rows))
    n = len(rows)
    cover = COVER[a.cond]
    n_aug = 4 if a.split == "train" else 1
    g = PATCH // enc.patch_size
    shape = ((n_aug,) if a.split == "train" else ()) + (n, len(enc.layers), enc.channels, g, g)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{enc.name}_{a.cond}_{a.split}"
    npy_path, json_path = a.out_dir / f"{stem}.npy", a.out_dir / f"{stem}.json"
    part, prog_path = a.out_dir / f"{stem}.npy.part", a.out_dir / f"{stem}.progress.json"
    if npy_path.is_file() and not a.overwrite:
        if tuple(np.load(npy_path, mmap_mode="r").shape) == shape:
            print(f"[{enc.name}] {npy_path} exists with shape {list(shape)}: skipped (--overwrite to redo)")
            return None
        raise SystemExit(f"{npy_path} exists with another shape; use --overwrite")
    key = {"model": enc.name, "split": a.split, "cond": a.cond, "shape": list(shape), "layers": enc.layers,
           "idx": [r["idx"] for r in rows], "patches_dir": _rel(a.patches_dir), "weights": enc.weights_info}
    state = {"done": 0, "enc_seconds": 0.0, "enc_samples": 0, "nonfinite": 0, "max_abs": 0.0}
    out = None
    if part.is_file() and prog_path.is_file() and not a.overwrite:
        prog = json.loads(prog_path.read_text())
        if prog.get("key") == key:
            out = np.lib.format.open_memmap(part, mode="r+")
            state = prog["state"]
            print(f"[{enc.name}] resuming at patch {state['done']}/{n}", flush=True)
    if out is None:
        out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=shape)
    print(f"[{enc.name}] split={a.split} cond={a.cond} (cover {cover}) patches {n}/{n_total} augmentations "
          f"{n_aug} threads {a.threads} batch {a.batch} -> {list(shape)}", flush=True)

    t0 = time.time()
    enc.build(a)
    print(f"[{enc.name}] model ready in {time.time() - t0:.1f} s", flush=True)

    # Optical-only model with optical fully missing: every patch has the same (all-fill) input, so the features
    # are identical; encode one patch and reuse it (saves hours of encoding a constant).
    constant = (not enc.uses_sar) and cover >= 1.0
    const_feat = None
    t_start, first = time.time(), state["done"]
    for i0 in range(state["done"], n, a.batch):
        batch = rows[i0: i0 + a.batch]
        opt8, sar = src.get(batch)
        opt = enc.optical(batch, opt8, a)
        opt = apply_clouds(opt, cover, seed=CLOUD_SEED, start_index=i0)  # i0 = position within the split
        for aug in range(n_aug):
            o, s = augment(opt, aug), augment(sar, aug)
            if constant and const_feat is not None:
                f = const_feat
            else:
                t = time.time()
                f = enc.encode(np.ascontiguousarray(o[:1] if constant else o),
                               np.ascontiguousarray(s[:1] if constant else s))
                state["enc_seconds"] += time.time() - t
                state["enc_samples"] += f.shape[0]
                if constant:
                    const_feat = f
            if constant:
                f = f[:1].expand(len(batch), *f.shape[1:])
            if tuple(f.shape[1:]) != shape[-4:]:
                raise SystemExit(f"encoder returned {tuple(f.shape)}, expected (B, {shape[-4:]})")
            f16 = f.to(torch.float16).contiguous().numpy()
            state["nonfinite"] += int((~np.isfinite(f16)).sum())
            state["max_abs"] = max(state["max_abs"], float(f.abs().max()))
            if a.split == "train":
                out[aug, i0: i0 + len(batch)] = f16
            else:
                out[i0: i0 + len(batch)] = f16
        out.flush()
        state["done"] = min(i0 + a.batch, n)
        _write_json(prog_path, {"key": key, "state": state})
        el, done = time.time() - t_start, state["done"]
        if done == n or ((i0 - first) // a.batch) % 10 == 0:
            eta = el / max(done - first, 1) * (n - done)
            print(f"[{enc.name}] {done}/{n} patches, {el:.0f} s, ETA {eta:.0f} s", flush=True)
    del out
    os.replace(part, npy_path)
    prog_path.unlink(missing_ok=True)
    if state["nonfinite"]:
        print(f"[{enc.name}] WARNING: {state['nonfinite']} non-finite values after float16 cast", flush=True)
    spp = state["enc_seconds"] / max(state["enc_samples"], 1)
    side = {
        "model": enc.name, "cond": a.cond, "split": a.split, "shape": list(shape), "layers": enc.layers,
        "seconds_per_sample": round(spp, 4), "input": enc.describe(),
        "layer_index": "zero-based transformer block index; the last one is after the encoder's final norm",
        "augmentations": AUG_NAMES if a.split == "train" else ["identity"],
        "cloud_cover": cover, "cloud_seed": CLOUD_SEED, "n_split": n_total, "limit": a.limit,
        "threads": a.threads, "batch": a.batch, "weights": enc.weights_info,
        "constant_input_reused": bool(constant), "nonfinite_float16": state["nonfinite"],
        "max_abs": round(state["max_abs"], 3), "patches_dir": _rel(a.patches_dir),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if a.limit is None:
        side["idx"] = [r["idx"] for r in rows]  # lets train_head.py verify membership, not just the count
    _write_json(json_path, side)
    print(f"[{enc.name}] wrote {npy_path} {list(shape)} float16, {spp:.3f} s/sample "
          f"({a.threads} threads), total {time.time() - t0:.0f} s", flush=True)
    return side


def load_checkpoint(path, repo, filename, revision):
    """Local checkpoint path, downloaded from the HF hub (pinned revision) into its folder if missing."""
    path = Path(path)
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        print(f"downloading {repo}/{filename}@{revision[:8]} -> {path.parent}", flush=True)
        hf_hub_download(repo, filename, revision=revision, local_dir=path.parent)
    return path


# ====================================================================== TerraMind

TM_REPO = "ibm-esa-geospatial/TerraMind-1.0-large"
TM_REVISION = "3ca567c5494310528269402a96ba4f13ceefc87c"
TM_FILE = "TerraMind_v1_large.pt"
S2L2A_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
EXTRA_BANDS = ["B01", "B06", "B07", "B09"]  # s2extra.tif band order (src/download_extra.py)


class ExtraBands:
    """Reads B01 B06 B07 B09 for a patch from data/sites/<site>/s2extra.tif. Checks once per site that s2extra.tif
    is on the s2.tif grid and that the patch window really is where optical.npy came from."""

    def __init__(self, sites_dir):
        self.sites_dir = Path(sites_dir)
        self.ds = {}

    def _site(self, site):
        if site not in self.ds:
            import rasterio

            p = self.sites_dir / site / "s2extra.tif"
            if not p.is_file():
                raise SystemExit(f"{p} missing: run `python -I src/download_extra.py {site}` (main .venv) first")
            ds, ref = rasterio.open(p), rasterio.open(self.sites_dir / site / "s2.tif")
            if (ds.count, ds.crs, ds.transform, ds.shape) != (4, ref.crs, ref.transform, ref.shape):
                raise SystemExit(f"{p} is not on the s2.tif grid of {site}")
            self.ds[site] = (ds, ref, [False])
        return self.ds[site]

    @staticmethod
    def offsets(row, transform):
        """Pixel offsets (row0, col0) of a patch; meta x0/y0 must agree (map metres, or pixel offsets)."""
        r0, c0 = row["row"] * PATCH, row["col"] * PATCH
        cf, rf = ~transform * (row["x0"], row["y0"])
        if not ((abs(cf - c0) < 1e-3 and abs(rf - r0) < 1e-3) or (row["x0"] == c0 and row["y0"] == r0)):
            raise SystemExit(f"meta.csv idx {row['idx']}: x0/y0 ({row['x0']}, {row['y0']}) disagree with "
                             f"row/col window ({r0}, {c0})")
        return r0, c0

    def read(self, row, opt8):
        from rasterio.windows import Window

        ds, ref, checked = self._site(row["site"])
        r0, c0 = self.offsets(row, ds.transform)
        win = Window(c0, r0, PATCH, PATCH)
        if not checked[0]:  # optical.npy patch must equal s2.tif at this window (NaN-aware)
            s2 = ref.read(window=win, boundless=False).astype(np.float32)
            same = (s2 == opt8) | (np.isnan(s2) & np.isnan(opt8))
            if s2.shape != opt8.shape or not same.all():
                raise SystemExit(f"{row['site']} idx {row['idx']}: optical.npy does not match s2.tif at the "
                                 "meta window; extra bands would be misaligned")
            checked[0] = True
        x = ds.read(window=win, boundless=False).astype(np.float32)
        if x.shape != (4, PATCH, PATCH):
            raise SystemExit(f"{row['site']} idx {row['idx']}: window outside s2extra.tif")
        return x


class TerraMindEncoder(Encoder):
    name = "terramind-l"
    layers = [5, 11, 17, 23]
    uses_sar = True
    weights_info = {"repo": TM_REPO, "revision": TM_REVISION, "file": TM_FILE, "license": "Apache-2.0",
                    "terratorch_backbone": "terramind_v1_large", "merge_method": "mean"}

    def build(self, args):
        import torch
        from terratorch.models.backbones.terramind.model.terramind_register import (
            terramind_v1_large, v1_pretraining_mean, v1_pretraining_std)

        self.mean_s2 = v1_pretraining_mean["untok_sen2l2a@224"]
        self.std_s2 = v1_pretraining_std["untok_sen2l2a@224"]
        self.mean_s1 = v1_pretraining_mean["untok_sen1rtc@224"]
        self.std_s1 = v1_pretraining_std["untok_sen1rtc@224"]
        self.extra = ExtraBands(args.sites_dir)
        ckpt = load_checkpoint(args.weights or ROOT / "weights/terramind-l" / TM_FILE, TM_REPO, TM_FILE, TM_REVISION)
        model = terramind_v1_large(pretrained=False, modalities=["S2L2A", "S1RTC"], img_size=PATCH,
                                   merge_method="mean")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        own = model.state_dict()
        keep, skipped = {}, []
        for k, v in sd.items():
            if k in own:
                if v.shape == own[k].shape:
                    keep[k] = v
                else:
                    skipped.append(k)
        # Only the fixed sin-cos position buffers may differ (14 x 14 in the checkpoint, rebuilt for 16 x 16).
        bad = [k for k in skipped if not k.endswith(".pos_emb")]
        missing = [k for k in own if k not in keep and not k.endswith(".pos_emb")]
        if bad or missing:
            raise SystemExit(f"TerraMind checkpoint mismatch: shape {bad[:5]} missing {missing[:5]}")
        res = model.load_state_dict(keep, strict=False)
        assert all(k.endswith(".pos_emb") for k in res.missing_keys) and not res.unexpected_keys
        model.eval()
        self.model = model
        self.n_params = sum(p.numel() for p in model.parameters())
        self.weights_info = dict(self.weights_info, encoder_params=self.n_params, tensors_loaded=len(keep))
        print(f"[{self.name}] loaded {len(keep)} encoder tensors ({self.n_params / 1e6:.0f} M params), "
              f"pos_emb rebuilt for {PATCH} px: {sorted(skipped)}", flush=True)
        return model

    def optical(self, rows, opt8, args):
        extra = np.stack([self.extra.read(r, o) for r, o in zip(rows, opt8)])
        src = {b: opt8[:, i] for i, b in enumerate(S2_BANDS)}
        src.update({b: extra[:, i] for i, b in enumerate(EXTRA_BANDS)})
        return np.stack([src[b] for b in S2L2A_ORDER], axis=1)  # reflectance 0..1, NaN invalid

    def encode(self, optical, sar):
        import torch

        s2, _ = standardise_fill(optical * 10000.0, self.mean_s2, self.std_s2)
        s1, _ = standardise_fill(sar, self.mean_s1, self.std_s1)
        out = self.model({"S2L2A": torch.from_numpy(s2), "S1RTC": torch.from_numpy(s1)})
        return torch.stack([tokens_to_grid(out[i]) for i in self.layers], dim=1)

    def describe(self):
        return ("S2L2A 12 bands B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 (8 from optical.npy + B01 B06 B07 "
                "B09 from s2extra.tif), DN = reflectance*10000, standardised with TerraMind v1 S2L2A pretraining "
                "mean/std; S1RTC VV VH dB standardised with TerraMind v1 S1RTC mean/std; pixels with any NaN band "
                "set to 0 (band mean) per modality; clouds on all 12 optical bands; 256 px, sin-cos pos-emb "
                "built for 16x16; merge_method=mean (S2L2A and S1RTC tokens averaged per position)")


if __name__ == "__main__":
    run(TerraMindEncoder())
