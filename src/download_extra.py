"""Fetch the four Sentinel-2 L2A bands that src/download.py does not keep, for TerraMind's 12-band S2L2A input.

  data/sites/<site>/s2extra.tif  4 bands float32, order B01 B06 B07 B09, surface reflectance 0..1 (DN / 10000),
                                 NaN where never clear. Same 10 m grid as s2.tif (shape, transform, CRS copied
                                 from s2.tif), same composite recipe as download.py:
                                 - the SAME scene selection: identical STAC search (sentinel-2-l2a, site bbox,
                                   2019-07-01/2021-06-30, eo:cloud_cover < 40), sorted by cloud cover, first MAX_S2;
                                 - the SAME per-scene clear mask: SCL in CLEAR_SCL (SCL resampled nearest) and DN > 0;
                                 - bilinear resampling to 10 m (B01/B09 are 60 m, B06/B07 20 m), groupby solar_day,
                                   per-pixel median over time.
Check: by default B05 is recomputed the same way and compared with band 4 of s2.tif. Equality proves that the
scene selection, grid and compositing match download.py; the run stops (no s2extra.tif) when it does not.
Provenance (STAC item ids, check result) is stored as GeoTIFF tags of s2extra.tif.

Usage: python -I src/download_extra.py <site> [<site> ...] [--no-check]
Run it only after download.py has finished a site (needs data/sites/<site>/s2.tif). Uses the main .venv.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from odc.geo.geobox import GeoBox
from odc.stac import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import CATALOG, CLEAR_SCL, MAX_S2, write  # noqa: E402  (same constants as the main download)
from sites import S2_BANDS, SITES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sites"
EXTRA_BANDS = ["B01", "B06", "B07", "B09"]
CHECK_BAND = "B05"  # present in s2.tif, recomputed to prove identical scenes and grid
# Must stay identical to the Sentinel-2 search in download.py build().
S2_DATETIME = "2019-07-01/2021-06-30"
S2_QUERY = {"eo:cloud_cover": {"lt": 40}}


def search_s2(site):
    """Same search, same order, same cut as download.py (Python's sort is stable, so ties keep search order)."""
    cat = pystac_client.Client.open(CATALOG)  # unsigned; load() signs each read (patch_url) so SAS tokens never expire mid-site
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=SITES[site]["bbox"], datetime=S2_DATETIME,
                            query=S2_QUERY).items())
    return sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:MAX_S2]


def composite(ds, band, clear):
    v = ds[band].where(clear & (ds[band] > 0)).astype("float32")
    return (v.median("time", skipna=True) / 10000.0).compute().values.astype("float32")


def build(site, check=True):
    d = OUT / site
    s2_path = d / "s2.tif"
    if not (s2_path.is_file() and (d / "label.tif").is_file()):  # label.tif is written last by download.py
        raise SystemExit(f"[{site}] download.py has not finished this site (s2.tif/label.tif missing)")
    with rasterio.open(s2_path) as s:
        crs, transform, h, w = s.crs, s.transform, s.height, s.width
        ref = s.read(S2_BANDS.index(CHECK_BAND) + 1) if check else None
    gbox = GeoBox((h, w), transform, crs)

    items = search_s2(site)
    print(f"[{site}] S2 scenes: {len(items)} (same search as download.py)", flush=True)
    bands = EXTRA_BANDS + ([CHECK_BAND] if check else [])
    ds = load(items, patch_url=pc.sign, bands=bands + ["SCL"], like=gbox, groupby="solar_day", chunks={"x": 1024, "y": 1024},
              resampling={"SCL": "nearest", "*": "bilinear"})
    if ds.odc.geobox.shape != (h, w) or ds.odc.geobox.affine != transform:
        raise SystemExit(f"[{site}] loaded grid {ds.odc.geobox} differs from s2.tif")
    clear = ds["SCL"].isin(CLEAR_SCL).persist()

    tags = {"bands": " ".join(EXTRA_BANDS), "units": "surface reflectance (DN / 10000), NaN = never clear",
            "recipe": "download_extra.py: same search/sort/clear mask/median as download.py",
            "stac_items": " ".join(i.id for i in items), "solar_days": str(ds.sizes["time"])}
    baselines = [str(i.properties.get("s2:processing_baseline")) for i in items]
    tags["processing_baselines"] = " ".join(baselines)
    if any(b[:1].isdigit() and b >= "04.00" for b in baselines):  # download.py does not remove the +1000 DN offset either
        print(f"[{site}] WARNING: processing baseline >= 04.00 among {baselines}: DN carry a +1000 offset in "
              "both s2.tif and s2extra.tif", flush=True)
    if check:
        b05 = composite(ds, CHECK_BAND, clear)
        both = np.isfinite(b05) & np.isfinite(ref)
        nan_agree = float((np.isfinite(b05) == np.isfinite(ref)).mean())
        max_diff = float(np.abs(b05[both] - ref[both]).max()) if both.any() else float("nan")
        msg = f"check {CHECK_BAND} vs s2.tif: NaN-mask agreement {nan_agree:.6f}, max |diff| {max_diff:.2e}"
        print(f"[{site}] {msg}", flush=True)
        tags["check"] = msg
        if nan_agree < 0.9999 or not max_diff <= 1e-6:
            raise SystemExit(f"[{site}] {CHECK_BAND} does not reproduce s2.tif: scene selection or grid differ. "
                             "Not writing s2extra.tif.")

    out = []
    for b in EXTRA_BANDS:
        out.append(composite(ds, b, clear))
        print(f"[{site}]   {b} done, finite {np.isfinite(out[-1]).mean():.3f}", flush=True)
    arr = np.stack(out)

    tmp = d / "s2extra.tif.part"
    write(tmp, arr, transform, crs, nodata=np.nan)
    with rasterio.open(tmp, "r+") as dst:
        dst.update_tags(**tags)
        for i, b in enumerate(EXTRA_BANDS, 1):
            dst.set_band_description(i, b)
    os.replace(tmp, d / "s2extra.tif")
    fin = np.isfinite(arr)
    print(f"[{site}] wrote {d / 's2extra.tif'}: {w}x{h}, finite per band "
          f"{[round(float(f.mean()), 3) for f in fin]}, value range "
          f"{float(np.nanmin(arr)):.4f}..{float(np.nanmax(arr)):.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sites", nargs="+", choices=list(SITES))
    p.add_argument("--no-check", action="store_true", help="skip the B05 reproduction check (faster)")
    a = p.parse_args()
    for s in a.sites:
        build(s, check=not a.no_check)
