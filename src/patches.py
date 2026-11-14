"""Tile the per-site rasters into 256 x 256 patches (CONTRACT.md, section "Patches").

Input: data/sites/<site>/{s2,s1,label}.tif, one shared 10 m UTM grid per site (checked: size, transform, CRS).
Output (--out, default data/patches):
  optical.npy  float32 (N, 8, 256, 256)  B2 B3 B4 B5 B8 B8A B11 B12 reflectance, NaN where s2.tif is NaN
  sar.npy      float32 (N, 2, 256, 256)  VV, VH in dB, NaN where s1.tif is NaN
  label.npy    uint8   (N, 256, 256)     1 mangrove, 0 other, 255 ignore (see "Labels")
  meta.csv     idx, site, split, row, col, x0, y0, mangrove_frac, s2_valid_frac
  stats.json   per-band mean/std over the train split, finite pixels only; physical min/max
  info.json    provenance: per-site grid (CRS, transform), val rows, kept/dropped counts, stats pixel counts

Rules
  Tiling     windows start at pixel (0, 0) and step 256 with no overlap; a window that would cross the right or
             bottom edge of the raster is dropped, so a site gives (height // 256) x (width // 256) windows.
  Validity   s2_valid_frac = share of window pixels where all 8 S2 bands are finite; s1 valid share = share where
             VV and VH are both finite. A window is kept when both shares are >= 0.8.
  Split      sites whose sites.py split is "test" -> test. Train sites -> train, except the bottom two rows of
             the window grid (grid rows n_rows-2 and n_rows-1, counted before any window is dropped) -> val.
  Order      idx runs over the selected sites in sites.py order (whatever order --sites lists them in), row-major
             (row, col) inside a site. Downstream code takes the patches of one split in idx order.
  meta.csv   row, col = window index in the site grid (pixel offset row*256, col*256). x0, y0 = map coordinates in
             metres of the window's upper-left corner, in that site's UTM CRS (CRS per site in info.json).
             mangrove_frac = label==1 pixels / labelled pixels (label != 255).
  Values     copied unchanged from the rasters (no clipping, no masking of finite values). Finite values outside
             the physical bounds are counted and printed; FLORO treats them as invalid itself.
  Labels     copied from label.tif (0/1). 255 (ignore) is meant for pixels outside the raster; incomplete edge
             windows are dropped, so no such pixel exists and 255 is never written. A 255 already present in
             label.tif would pass through as ignore; any other value stops the run.
  stats.json mean and population std (ddof 0) per band over the finite pixels of train patches only (never
             val/test); min/max are physical bounds (optical 0/1, sar -60/20), not empirical.

Usage
  python -I src/patches.py                                   all sites in sites.py -> data/patches
  python -I src/patches.py --sites mahakam kubu_raya --out data/patches_small
  --sites-dir DIR          read site rasters from DIR instead of data/sites
  --force-split train      TEST ONLY: treat every selected site as a train site (train + bottom-two-row val);
                           refused for data/patches
Relative --out / --sites-dir paths are taken from the repo root.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sites import FLORO_OPTICAL, SITES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITES_DIR = ROOT / "data/sites"
DEFAULT_OUT = ROOT / "data/patches"
PATCH = 256
MIN_S2_VALID = 0.8
MIN_S1_VALID = 0.8
VAL_ROWS = 2
IGNORE = 255
SAR_NAMES = ["VV", "VH"]
OPTICAL_BOUNDS = (0.0, 1.0)
SAR_BOUNDS = (-60.0, 20.0)
RASTERS = {"s2": (8, "float32"), "s1": (2, "float32"), "label": (1, "uint8")}
META_COLUMNS = ["idx", "site", "split", "row", "col", "x0", "y0", "mangrove_frac", "s2_valid_frac"]


class PatchError(RuntimeError):
    """Input problem that must stop the run (missing raster, grid mismatch, bad label values, ...)."""


# ---------------------------------------------------------------- checks


def check_site(site, sites_dir):
    """Check that the three rasters exist, have the expected bands/dtypes and share one grid. Returns the grid."""
    d = sites_dir / site
    missing = [f"{k}.tif" for k in RASTERS if not (d / f"{k}.tif").is_file()]
    if missing:
        raise PatchError(f"{site}: missing {', '.join(missing)} in {d} "
                         f"(download not finished? run: python -I src/download.py {site})")
    meta = {}
    for k, (count, dtype) in RASTERS.items():
        path = d / f"{k}.tif"
        try:
            with rasterio.open(path) as ds:
                meta[k] = {"count": ds.count, "dtypes": sorted(set(ds.dtypes)), "width": ds.width,
                           "height": ds.height, "crs": ds.crs, "transform": ds.transform}
        except RasterioIOError as e:
            raise PatchError(f"{site}: cannot open {path}: {e}") from e
        m = meta[k]
        if m["count"] != count or m["dtypes"] != [dtype]:
            raise PatchError(f"{site}: {k}.tif has {m['count']} band(s) of {m['dtypes']}, expected {count} x {dtype}")
    ref = meta["s2"]
    problems = []
    for k in ("s1", "label"):
        m = meta[k]
        if (m["width"], m["height"]) != (ref["width"], ref["height"]):
            problems.append(f"{k}.tif size {m['width']}x{m['height']} != s2.tif {ref['width']}x{ref['height']}")
        if m["crs"] != ref["crs"]:
            problems.append(f"{k}.tif CRS {m['crs']} != s2.tif {ref['crs']}")
        if m["transform"] != ref["transform"]:
            problems.append(f"{k}.tif transform {tuple(m['transform'])[:6]} != s2.tif {tuple(ref['transform'])[:6]}")
    if ref["crs"] is None:
        problems.append("s2.tif has no CRS")
    if problems:
        raise PatchError(f"{site}: s2/s1/label rasters do not share one grid:\n    " + "\n    ".join(problems))
    n_rows, n_cols = ref["height"] // PATCH, ref["width"] // PATCH
    if n_rows < 1 or n_cols < 1:
        raise PatchError(f"{site}: raster {ref['width']}x{ref['height']} is smaller than one {PATCH} px window")
    return {"width": ref["width"], "height": ref["height"], "crs": ref["crs"], "transform": ref["transform"],
            "n_rows": n_rows, "n_cols": n_cols}


def assign_split(role, row, n_rows):
    """role is the site's split in sites.py ("train" or "test"); row is the window's grid row."""
    if role == "test":
        return "test"
    if role == "train":
        return "val" if row >= n_rows - VAL_ROWS else "train"
    raise PatchError(f"unknown site split {role!r} (expected 'train' or 'test')")


# ---------------------------------------------------------------- pass 1: which windows to keep


def _window_counts(mask, n_rows, n_cols):
    """(H, W) bool -> (n_rows, n_cols) count of True per complete 256 x 256 window."""
    m = mask[: n_rows * PATCH, : n_cols * PATCH]
    return m.reshape(n_rows, PATCH, n_cols, PATCH).sum(axis=(1, 3), dtype=np.int64)


def scan_site(site, grid, sites_dir, role):
    """Per-window validity and mangrove share. Reads one band at a time to keep memory low."""
    d = sites_dir / site
    nr, nc = grid["n_rows"], grid["n_cols"]
    npx = PATCH * PATCH
    with rasterio.open(d / "s2.tif") as ds:
        ok = np.ones((ds.height, ds.width), bool)
        for b in range(1, ds.count + 1):
            ok &= np.isfinite(ds.read(b))
    s2_frac = _window_counts(ok, nr, nc) / npx
    with rasterio.open(d / "s1.tif") as ds:
        ok = np.ones((ds.height, ds.width), bool)
        for b in range(1, ds.count + 1):
            ok &= np.isfinite(ds.read(b))
    s1_frac = _window_counts(ok, nr, nc) / npx
    del ok
    with rasterio.open(d / "label.tif") as ds:
        lab = ds.read(1)
    bad = sorted(set(np.unique(lab).tolist()) - {0, 1, IGNORE})
    if bad:
        raise PatchError(f"{site}: label.tif holds values {bad}; expected 0 (other), 1 (mangrove), 255 (ignore)")
    n_mang = _window_counts(lab == 1, nr, nc)
    n_lab = _window_counts(lab != IGNORE, nr, nc)
    n_ign = int((n_lab.size * npx) - n_lab.sum())
    del lab

    t = grid["transform"]
    windows = []
    for r in range(nr):
        for c in range(nc):
            x0, y0 = t * (c * PATCH, r * PATCH)
            windows.append({
                "site": site, "row": r, "col": c, "split": assign_split(role, r, nr),
                "x0": float(x0), "y0": float(y0),
                "mangrove_frac": float(n_mang[r, c] / n_lab[r, c]) if n_lab[r, c] else float("nan"),
                "s2_valid_frac": float(s2_frac[r, c]), "s1_valid_frac": float(s1_frac[r, c]),
                "keep": bool(s2_frac[r, c] >= MIN_S2_VALID and s1_frac[r, c] >= MIN_S1_VALID),
            })
    return windows, n_ign


# ---------------------------------------------------------------- pass 2: write arrays


class BandStats:
    """Running mean / population variance per band over finite pixels (Chan et al. pairwise merge, float64)."""

    def __init__(self, n_bands):
        self.n = np.zeros(n_bands, np.int64)
        self.mean = np.zeros(n_bands, np.float64)
        self.m2 = np.zeros(n_bands, np.float64)

    def add(self, x):
        for b in range(x.shape[0]):
            v = x[b][np.isfinite(x[b])].astype(np.float64)
            if v.size == 0:
                continue
            nb, mb = v.size, v.mean()
            m2b = float(((v - mb) ** 2).sum())
            na = self.n[b]
            n = na + nb
            delta = mb - self.mean[b]
            self.mean[b] += delta * nb / n
            self.m2[b] += m2b + delta * delta * na * nb / n
            self.n[b] = n

    def result(self, names, bounds):
        if (self.n == 0).any():
            empty = [nm for nm, k in zip(names, self.n) if k == 0]
            raise PatchError(f"no finite train pixels for band(s) {empty}; cannot compute stats.json")
        std = np.sqrt(self.m2 / self.n)
        return {nm: {"mean": float(m), "std": float(s), "min": bounds[0], "max": bounds[1]}
                for nm, m, s in zip(names, self.mean, std)}


def _out_of_bounds(x, lo, hi):
    f = np.isfinite(x)
    return int(((x < lo) & f).sum() + ((x > hi) & f).sum())


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def build(sites, out, sites_dir, force_split=None):
    t_start = time.time()

    # checks first, for every site, so a missing or mismatched raster stops the run before anything is written
    grids, errors = {}, []
    for s in sites:
        try:
            grids[s] = check_site(s, sites_dir)
        except PatchError as e:
            errors.append(str(e))
    if errors:
        raise PatchError("input check failed:\n  " + "\n  ".join(errors))
    roles = {s: (force_split or SITES[s]["split"]) for s in sites}

    # pass 1
    windows, ignore_px = [], {}
    for s in sites:
        t0 = time.time()
        w, ignore_px[s] = scan_site(s, grids[s], sites_dir, roles[s])
        windows += w
        k = sum(x["keep"] for x in w)
        print(f"[scan] {s}: {grids[s]['n_rows']}x{grids[s]['n_cols']} windows, kept {k}, {time.time() - t0:.1f} s",
              flush=True)
    kept = [w for w in windows if w["keep"]]
    n = len(kept)
    if n == 0:
        raise PatchError("no window passed the validity rules; nothing to write")
    if not any(w["split"] == "train" for w in kept):
        raise PatchError("no train patches among the selected sites; stats.json needs the train split "
                         "(include at least one train site)")
    for s in sites:
        if roles[s] == "train" and not any(w["keep"] and w["site"] == s and w["split"] == "val" for w in windows):
            print(f"WARNING: train site {s} has no kept val patch in its bottom {VAL_ROWS} rows", flush=True)

    # pass 2: write to hidden partial files, rename once everything is complete
    out.mkdir(parents=True, exist_ok=True)
    names = {"optical": "optical.npy", "sar": "sar.npy", "label": "label.npy", "meta": "meta.csv",
             "stats": "stats.json", "info": "info.json"}
    tmp = {k: out / f".{v}.partial" for k, v in names.items()}
    try:
        arrays = {"optical": (np.float32, (n, 8, PATCH, PATCH)), "sar": (np.float32, (n, 2, PATCH, PATCH)),
                  "label": (np.uint8, (n, PATCH, PATCH))}
        for k, (dt, shape) in arrays.items():  # create the full-size files once; mapped again per site below
            np.lib.format.open_memmap(tmp[k], mode="w+", dtype=dt, shape=shape).flush()
        st_opt, st_sar = BandStats(8), BandStats(2)
        oob = {}
        i = 0
        for s in sites:
            ks = [w for w in kept if w["site"] == s]
            if not ks:
                continue
            t0 = time.time()
            d = sites_dir / s
            # map the outputs per site and unmap after it, so written pages never pile up across sites
            opt, sar, lab = (np.lib.format.open_memmap(tmp[k], mode="r+") for k in ("optical", "sar", "label"))
            with rasterio.open(d / "s2.tif") as ds:
                s2 = ds.read()
            with rasterio.open(d / "s1.tif") as ds:
                s1 = ds.read()
            with rasterio.open(d / "label.tif") as ds:
                lb = ds.read(1)
            for w in ks:
                r0, c0 = w["row"] * PATCH, w["col"] * PATCH
                o = s2[:, r0:r0 + PATCH, c0:c0 + PATCH]
                a = s1[:, r0:r0 + PATCH, c0:c0 + PATCH]
                opt[i] = o
                sar[i] = a
                lab[i] = lb[r0:r0 + PATCH, c0:c0 + PATCH]
                if w["split"] == "train":
                    st_opt.add(o)
                    st_sar.add(a)
                key = (s, w["split"])
                prev = oob.get(key, (0, 0))
                oob[key] = (prev[0] + _out_of_bounds(o, *OPTICAL_BOUNDS), prev[1] + _out_of_bounds(a, *SAR_BOUNDS))
                w["idx"] = i
                i += 1
            del s2, s1, lb, o, a  # o, a are views that would keep the whole site raster alive
            opt.flush()
            sar.flush()
            lab.flush()
            del opt, sar, lab
            print(f"[write] {s}: {len(ks)} patches, {time.time() - t0:.1f} s", flush=True)
        assert i == n, (i, n)

        stats = {"optical": st_opt.result(FLORO_OPTICAL, OPTICAL_BOUNDS), "sar": st_sar.result(SAR_NAMES, SAR_BOUNDS)}
        with open(tmp["stats"], "w") as f:
            json.dump(stats, f, indent=2)

        with open(tmp["meta"], "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(META_COLUMNS)
            for w in kept:
                wr.writerow([w["idx"], w["site"], w["split"], w["row"], w["col"], repr(w["x0"]), repr(w["y0"]),
                             f"{w['mangrove_frac']:.6f}", f"{w['s2_valid_frac']:.6f}"])

        info = {
            "created_by": "src/patches.py",
            "created_unix": round(time.time()),
            "sites_dir": _rel(sites_dir),
            "force_split": force_split,
            "patch": PATCH,
            "rules": {
                "tiling": "non-overlapping windows from pixel (0, 0); incomplete right/bottom edge windows dropped",
                "keep": f"s2_valid_frac >= {MIN_S2_VALID} (all 8 bands finite) and s1 valid >= {MIN_S1_VALID} "
                        "(VV and VH finite)",
                "split": f"test sites -> test; train sites -> train except the bottom {VAL_ROWS} window-grid rows -> val",
                "order": "selected sites in sites.py order, row-major (row, col) within a site",
                "x0_y0": "map coordinates (m) of the window's upper-left corner in the site's CRS; "
                         "pixel offset = (row*256, col*256)",
                "label_ignore": "255 marks pixels outside the raster; edge windows are dropped, so none are written",
                "stats": "mean and population std over finite pixels of train patches only; min/max physical bounds",
                "values": "copied unchanged from the site rasters (no clipping)",
            },
            "sites": {},
            "stats_pixels": {"optical": dict(zip(FLORO_OPTICAL, st_opt.n.tolist())),
                             "sar": dict(zip(SAR_NAMES, st_sar.n.tolist()))},
            "n_patches": {sp: sum(w["split"] == sp for w in kept) for sp in ("train", "val", "test")},
        }
        for s in sites:
            g = grids[s]
            ws = [w for w in windows if w["site"] == s]
            info["sites"][s] = {
                "role": roles[s], "crs": g["crs"].to_string(), "transform": list(tuple(g["transform"])[:6]),
                "width": g["width"], "height": g["height"], "grid_rows": g["n_rows"], "grid_cols": g["n_cols"],
                "val_rows": list(range(g["n_rows"] - VAL_ROWS, g["n_rows"])) if roles[s] == "train" else [],
                "windows": len(ws), "kept": sum(w["keep"] for w in ws),
                "ignore_pixels_in_label_tif": ignore_px[s],
                "out_of_bounds_values": {sp: {"optical": v[0], "sar": v[1]} for (ss, sp), v in oob.items() if ss == s},
            }
        with open(tmp["info"], "w") as f:
            json.dump(info, f, indent=2)

        for k, v in names.items():
            os.replace(tmp[k], out / v)
    finally:
        for p in tmp.values():
            if p.exists():
                p.unlink()

    print_summary(sites, windows, roles, oob, stats, st_opt, st_sar, out, time.time() - t_start)
    return {"n": n, "windows": windows, "stats": stats}


def print_summary(sites, windows, roles, oob, stats, st_opt, st_sar, out, seconds):
    hdr = f"{'site':<15} {'split':<6} {'windows':>7} {'kept':>5} {'dropped':>7} {'s2<0.8':>6} {'s1<0.8':>6} " \
          f"{'mangrove':>8} {'s2valid':>7} {'oob opt':>8} {'oob sar':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    totals = {}
    for s in sites:
        for sp in ("train", "val", "test"):
            ws = [w for w in windows if w["site"] == s and w["split"] == sp]
            if not ws:
                continue
            k = [w for w in ws if w["keep"]]
            n_s2 = sum(w["s2_valid_frac"] < MIN_S2_VALID for w in ws)
            n_s1 = sum(w["s1_valid_frac"] < MIN_S1_VALID for w in ws)
            mf = np.mean([w["mangrove_frac"] for w in k]) if k else float("nan")
            vf = np.mean([w["s2_valid_frac"] for w in k]) if k else float("nan")
            o = oob.get((s, sp), (0, 0))
            print(f"{s:<15} {sp:<6} {len(ws):>7} {len(k):>5} {len(ws) - len(k):>7} {n_s2:>6} {n_s1:>6} "
                  f"{mf:>8.4f} {vf:>7.4f} {o[0]:>8} {o[1]:>8}")
            t = totals.setdefault(sp, {"w": 0, "k": [], "s2": 0, "s1": 0})
            t["w"] += len(ws)
            t["k"] += k
            t["s2"] += n_s2
            t["s1"] += n_s1
    print("-" * len(hdr))
    for sp, t in totals.items():
        mf = np.mean([w["mangrove_frac"] for w in t["k"]]) if t["k"] else float("nan")
        vf = np.mean([w["s2_valid_frac"] for w in t["k"]]) if t["k"] else float("nan")
        print(f"{'ALL':<15} {sp:<6} {t['w']:>7} {len(t['k']):>5} {t['w'] - len(t['k']):>7} {t['s2']:>6} {t['s1']:>6} "
              f"{mf:>8.4f} {vf:>7.4f}")
    print("mangrove = mean mangrove share of kept patches; s2<0.8 / s1<0.8 count windows failing each rule "
          "(a window can fail both);\noob = finite values outside the physical bounds (optical 0..1, sar -60..20), "
          "copied unchanged.")
    print("\nstats.json (train split only, finite pixels):")
    for grp, st in (("optical", st_opt), ("sar", st_sar)):
        for (nm, v), cnt in zip(stats[grp].items(), st.n):
            print(f"  {grp:<7} {nm:<4} mean {v['mean']:>9.5f}  std {v['std']:>8.5f}  pixels {cnt}")
    print(f"\nwrote {_rel(out)}/{{optical,sar,label}}.npy, meta.csv, stats.json, info.json in {seconds:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tile site rasters into 256 x 256 patches (CONTRACT.md, Patches).")
    ap.add_argument("--sites", nargs="+", default=list(SITES), help="site names from sites.py (default: all)")
    ap.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)), help="output folder (default data/patches)")
    ap.add_argument("--sites-dir", default=str(DEFAULT_SITES_DIR.relative_to(ROOT)),
                    help="folder holding <site>/{s2,s1,label}.tif (default data/sites)")
    ap.add_argument("--force-split", choices=["train"], default=None,
                    help="TEST ONLY: treat every selected site as a train site; not allowed with the default --out")
    a = ap.parse_args(argv)

    out = (ROOT / a.out).resolve()
    sites_dir = (ROOT / a.sites_dir).resolve()
    unknown = [s for s in a.sites if s not in SITES]
    if unknown:
        ap.error(f"unknown site(s) {unknown}; known: {list(SITES)}")
    if len(set(a.sites)) != len(a.sites):
        ap.error(f"duplicate site names in --sites {a.sites}")
    if a.force_split and out == DEFAULT_OUT.resolve():
        ap.error("--force-split is a test-only flag and may not write to data/patches; pass another --out")
    if a.force_split:
        print(f"*** TEST MODE: every selected site treated as split '{a.force_split}' ***", flush=True)
    sites = [s for s in SITES if s in a.sites]  # canonical sites.py order, so idx order never depends on the CLI
    try:
        build(sites, out, sites_dir, a.force_split)
    except PatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
