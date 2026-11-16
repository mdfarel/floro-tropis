"""Extract frozen FLORO features for one (variant, split, condition) into the shared feature-file contract.

    python -I src/extract_floro.py --variant s2s1 --split test --cond cloud50 [--batch 8] [--threads 16]
                                   [--limit N] [--out-dir data/features] [--patches-dir data/patches]

Reads data/patches/{optical,sar}.npy + meta.csv (memory-mapped), takes the patches of --split in idx order,
applies clouds for the test conditions (clouds.apply_clouds, seed 2027, start_index = position of the batch's
first patch within the split), and for --split train encodes 4 augmentations of the input image
(0 identity, 1 hflip = flip W, 2 vflip = flip H, 3 rot180 = flip both; SAR is transformed the same way).

Writes data/features/<model>_<cond>_<split>.npy (float16; val/test (N, 4, 1024, 16, 16), train
(4, N, 4, 1024, 16, 16)) and the sidecar .json. Work goes to <name>.npy.tmp with a <name>.progress.json
checkpoint after every batch, so an interrupted run resumes where it stopped; the final .npy appears only
by an atomic rename once every patch is encoded, and the sidecar is written last (it is the "complete" marker).

Input scale: the default --input-scale floro_z is the FLORO pretraining normalisation, chosen on a priori grounds (a
development trial on 25 Segara Anakan test patches is disclosed in the README, Limitations; see floro_model.py).
Other scales are ablations; unless --out-dir is given they go to
data/features_<scale>/ so they can never overwrite the main data/features/ files.

Cache safety: every output carries a fingerprint (sidecar "fingerprint", progress-file "key") of everything that
determines its values: model, split, cond, shape, input scale, weights file (name + size), the split's idx list,
its meta.csv rows, a sha1 of the exact optical (and SAR) patch bytes it encodes, and a sha1 of the normalisation
numbers that reach the encoder (normalisation_params; stats.json bytes are recorded too, for provenance). A
finished output is skipped only when the whole fingerprint matches; a resume happens only when the progress key
matches. If patches.py or stats.json changed the inputs, the output is recomputed. If an existing output was
made with another input scale or weights, the run stops unless --overwrite is given.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from clouds import apply_clouds  # noqa: E402
from floro_model import (DEFAULT_INPUT_SCALE, DEFAULT_WEIGHTS, INPUT_SCALE_RATIONALE,  # noqa: E402
                         INPUT_SCALES, MODEL_NAMES, OUTPUT_LAYERS, build_floro, describe_input,
                         normalisation_params, normalisation_sha1)

ROOT = HERE.parent
COVER = {"clean": 0.0, "cloud50": 0.5, "cloud100": 1.0}
CLOUD_SEED = 2027
AUGS = ["identity", "hflip", "vflip", "rot180"]
META_ID_COLS = ["idx", "site", "split", "row", "col", "x0", "y0"]  # patch identity (labels do not affect features)
C, G = 1024, 16


def augment(x, a):
    """x: (B, C, H, W). 0 identity, 1 hflip (W axis), 2 vflip (H axis), 3 rot180 (both)."""
    if x is None or a == 0:
        return x
    if a == 1:
        y = x[..., ::-1]
    elif a == 2:
        y = x[..., ::-1, :]
    elif a == 3:
        y = x[..., ::-1, ::-1]
    else:
        raise ValueError(a)
    return np.ascontiguousarray(y)


def split_rows(meta_path, split):
    with open(meta_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    rows.sort(key=lambda r: int(r["idx"]))
    return rows


def _sha1_json(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_bytes_sha1(arrays, idx):
    """sha1 over the exact float32 patch bytes that will be encoded (arrays in order, patches in idx order)."""
    h = hashlib.sha1()
    for arr in arrays:
        h.update(f"{arr.dtype.str}{arr.shape[1:]}".encode())
        for i in idx:
            h.update(np.ascontiguousarray(arr[i], dtype=np.float32).tobytes())
    return h.hexdigest()


def _rel(p):
    """Path relative to the repository root when it lies inside it (keeps local paths out of sidecars)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def default_out_dir(input_scale):
    if input_scale == DEFAULT_INPUT_SCALE:
        return ROOT / "data/features"
    return ROOT / f"data/features_{input_scale}"


def _atomic_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(MODEL_NAMES), required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--cond", choices=list(COVER), required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="torch device for the encoder: cpu (default), cuda, or auto")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patches of the split (testing)")
    ap.add_argument("--out-dir", default=None,
                    help=f"default data/features for --input-scale {DEFAULT_INPUT_SCALE}, else data/features_<scale>")
    ap.add_argument("--patches-dir", default=str(ROOT / "data/patches"))
    ap.add_argument("--stats", default=None, help="default: <patches-dir>/stats.json")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--input-scale", choices=list(INPUT_SCALES), default=DEFAULT_INPUT_SCALE)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    if a.split in ("train", "val") and a.cond != "clean":
        ap.error("train and val use only --cond clean")
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    pdir = Path(a.patches_dir)
    stats_path = a.stats or str(pdir / "stats.json")
    model = MODEL_NAMES[a.variant]
    use_sar = a.variant == "s2s1"
    rows = split_rows(pdir / "meta.csv", a.split)
    if a.limit is not None:
        rows = rows[: a.limit]
    idx, sites = [int(r["idx"]) for r in rows], [r["site"] for r in rows]
    n = len(idx)
    if n == 0:
        sys.exit(f"no patches with split={a.split} in {pdir / 'meta.csv'}")
    optical = np.load(pdir / "optical.npy", mmap_mode="r")
    sar = np.load(pdir / "sar.npy", mmap_mode="r") if use_sar else None
    assert optical.ndim == 4 and optical.shape[1:] == (8, 256, 256), optical.shape
    assert max(idx) < optical.shape[0], "meta idx beyond optical.npy"
    if use_sar:
        assert sar.shape[0] == optical.shape[0] and sar.shape[1:] == (2, 256, 256), sar.shape
    train = a.split == "train"
    n_aug = len(AUGS) if train else 1
    shape = (len(AUGS), n, len(OUTPUT_LAYERS), C, G, G) if train else (n, len(OUTPUT_LAYERS), C, G, G)

    out_dir = Path(a.out_dir) if a.out_dir else default_out_dir(a.input_scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{a.cond}_{a.split}"
    final, side = out_dir / f"{stem}.npy", out_dir / f"{stem}.json"
    tmp, prog_path = out_dir / f"{stem}.npy.tmp", out_dir / f"{stem}.progress.json"

    # fingerprint of everything that determines the feature values
    t_fp = time.time()
    data_arrays = [optical, sar] if use_sar else [optical]
    key = {"model": model, "split": a.split, "cond": a.cond, "shape": list(shape), "input_scale": a.input_scale,
           "weights": {"name": Path(a.weights).name, "bytes": os.path.getsize(a.weights)},
           "augmentations": AUGS if train else None,
           "clouds": {"cover": COVER[a.cond], "seed": CLOUD_SEED} if COVER[a.cond] > 0 else None,
           "idx_sha1": hashlib.sha1(np.asarray(idx, np.int64).tobytes()).hexdigest(),
           "meta_rows_sha1": _sha1_json([[r[k] for k in META_ID_COLS] for r in rows]),
           "patches_sha1": patch_bytes_sha1(data_arrays, idx),
           "norm_sha1": normalisation_sha1(a.variant, stats_path, a.input_scale)}
    print(f"[extract] {stem}: fingerprint of {n} patches in {time.time() - t_fp:.1f} s", flush=True)

    if final.exists() and side.exists() and not a.overwrite:
        with open(side) as f:
            old = json.load(f)
        old_key = old.get("fingerprint")
        try:
            hdr = np.load(final, mmap_mode="r")
            hdr_ok = hdr.shape == shape and hdr.dtype == np.float16
            del hdr
        except Exception:  # noqa: BLE001
            hdr_ok = False
        if old_key == key and hdr_ok:
            print(f"[extract] {final.name} already complete with matching fingerprint {list(shape)}; skipping "
                  f"(use --overwrite to force)")
            return 0
        old_key = old_key or {}
        if (old.get("input_scale"), (old_key.get("weights") or {}).get("name", old.get("weights"))) != \
                (a.input_scale, key["weights"]["name"]):
            sys.exit(f"[extract] {final} exists with input_scale={old.get('input_scale')} weights={old.get('weights')}; "
                     f"this run is input_scale={a.input_scale} weights={key['weights']['name']}. Use another "
                     f"--out-dir or pass --overwrite.")
        diff = sorted(k for k in key if old_key.get(k) != key[k]) if old_key else ["no fingerprint in sidecar"]
        if not hdr_ok:
            diff.append("npy shape/dtype")
        print(f"[extract] {final.name} is stale ({', '.join(diff)} changed); recomputing", flush=True)

    done, enc_seconds, enc_samples = 0, 0.0, 0
    mm = None
    if tmp.exists() and prog_path.exists() and not a.overwrite:
        with open(prog_path) as f:
            prog = json.load(f)
        if prog.get("key") == key:
            mm = np.lib.format.open_memmap(tmp, mode="r+")
            assert mm.shape == shape and mm.dtype == np.float16
            done, enc_seconds, enc_samples = prog["done"], prog["encode_seconds"], prog["encoded_samples"]
            print(f"[extract] resuming {tmp.name} at {done}/{n} patches")
        else:
            print(f"[extract] discarding {tmp.name}: its progress key does not match this run", flush=True)
    if mm is None:
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=shape)
        _atomic_json(prog_path, {"key": key, "done": 0, "encode_seconds": 0.0, "encoded_samples": 0})

    enc = None
    if done < n:
        enc = build_floro(a.variant, stats_path, weights_path=a.weights, input_scale=a.input_scale)
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        enc.to(dev)
        print(f"[extract] encoder on {dev}", flush=True)
    cover = COVER[a.cond]
    for pos in range(done, n, a.batch):
        ids = idx[pos: pos + a.batch]
        b = len(ids)
        opt = np.asarray(optical[ids], dtype=np.float32)
        if cover > 0:
            opt = apply_clouds(opt, cover, seed=CLOUD_SEED, start_index=pos)
        sb = np.asarray(sar[ids], dtype=np.float32) if use_sar else None
        for aug in range(n_aug):
            t0 = time.time()
            feats = enc.encode(augment(opt, aug), augment(sb, aug))
            arr = torch.stack(feats, dim=1).float().cpu().numpy().astype(np.float16)  # (b, 4, 1024, 16, 16)
            enc_seconds += time.time() - t0
            enc_samples += b
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"non-finite features at split position {pos} aug {aug}")
            if train:
                mm[aug, pos: pos + b] = arr
            else:
                mm[pos: pos + b] = arr
        mm.flush()
        done = pos + b
        _atomic_json(prog_path, {"key": key, "done": done, "encode_seconds": enc_seconds,
                                 "encoded_samples": enc_samples})
        if (pos // a.batch) % 10 == 0 or done == n:
            sps = enc_seconds / max(enc_samples, 1)
            eta = sps * (n - done) * n_aug
            print(f"[extract] {stem}: {done}/{n} patches, {sps:.2f} s/sample, eta {eta/60:.1f} min", flush=True)

    mm.flush()
    del mm
    if enc is not None:
        assert enc.norm_sha1 == key["norm_sha1"], "normalisation changed during the run"
    # drop any old sidecar first: a crash between the rename and the new sidecar must never leave new
    # features next to an old (possibly matching) fingerprint
    try:
        os.remove(side)
    except FileNotFoundError:
        pass
    os.replace(tmp, final)
    sps = enc_seconds / max(enc_samples, 1)
    sidecar = {
        "model": model,
        "cond": a.cond,
        "split": a.split,
        "shape": list(shape),
        "layers": list(OUTPUT_LAYERS),
        "seconds_per_sample": round(sps, 4),
        "input": describe_input(a.variant, a.input_scale),
        "dtype": "float16",
        "axes": ("aug, patch, layer, channel, grid_h, grid_w" if train else "patch, layer, channel, grid_h, grid_w"),
        "augmentations": AUGS if train else None,
        "clouds": {"cover": cover, "seed": CLOUD_SEED, "start_index": "position of the batch's first patch in the split"}
        if cover > 0 else None,
        "input_scale": a.input_scale,
        "input_scale_rationale": INPUT_SCALE_RATIONALE[a.input_scale],
        "weights": Path(a.weights).name,
        "stats": _rel(stats_path),
        "stats_sha1": _sha1_file(stats_path),
        "normalisation": normalisation_params(a.variant, stats_path, a.input_scale),
        "fingerprint": key,
        "threads": a.threads,
        "device": a.device,
        "batch": a.batch,
        "limit": a.limit,
        "n_patches": n,
        "sites": dict(Counter(sites)),
        "idx": idx,
    }
    _atomic_json(side, sidecar)
    try:
        os.remove(prog_path)
    except FileNotFoundError:
        pass
    print(f"[extract] wrote {final} {list(shape)} float16, {sps:.3f} s/sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
