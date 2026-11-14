"""Build per-site rasters on one 10 m UTM grid:
  s2.tif    8 bands, surface reflectance 0-1, median of low-cloud scenes 2019-07..2021-06, NaN where never clear
  s1.tif    VV, VH in dB, median of Sentinel-1 RTC scenes in 2020
  label.tif 1 = mangrove where the 10 m pixel overlaps any GMW v3 2020 mangrove cell, 0 = other

Data: Microsoft Planetary Computer (sentinel-2-l2a, sentinel-1-rtc), Global Mangrove Watch v3 (CC BY 4.0).
Usage: python -I src/download.py <site> [<site> ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from odc.stac import load
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).parent))
from sites import S1_BANDS, S2_BANDS, SITES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GMW = ROOT / "data/raw/gmw/gmw2020.vrt"
OUT = ROOT / "data/sites"
CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLEAR_SCL = [4, 5, 6, 7, 11]  # vegetation, bare, water, unclassified, snow
MAX_S2 = 14
MAX_S1 = 12


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def write(path, arr, transform, crs, nodata=None):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata,
                       compress="deflate", tiled=True) as d:
        d.write(arr)


def build(site):
    cfg = SITES[site]
    bbox = cfg["bbox"]
    crs = utm_crs((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    out = OUT / site
    out.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open(CATALOG)  # unsigned; load() signs each read (patch_url) so SAS tokens never expire mid-site

    # --- Sentinel-2 ---
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2019-07-01/2021-06-30",
                            query={"eo:cloud_cover": {"lt": 40}}).items())
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:MAX_S2]
    print(f"[{site}] S2 scenes: {len(items)}", flush=True)
    ds = load(items, patch_url=pc.sign, bands=S2_BANDS + ["SCL"], bbox=bbox, crs=crs, resolution=10, groupby="solar_day",
              chunks={"x": 1024, "y": 1024}, resampling={"SCL": "nearest", "*": "bilinear"})
    clear = ds["SCL"].isin(CLEAR_SCL)
    bands = []
    for b in S2_BANDS:
        v = ds[b].where(clear & (ds[b] > 0)).astype("float32")
        # processing baseline >= 04.00 would add a +1000 DN offset; not corrected here (all scenes used are
        # pre-2022, baseline < 04.00, recorded by download_extra.py)
        bands.append((v.median("time", skipna=True) / 10000.0).compute().values)
        print(f"[{site}]   {b} done", flush=True)
    s2 = np.stack(bands).astype("float32")
    gt = ds.odc.geobox.affine
    h, w = s2.shape[1:]
    write(out / "s2.tif", s2, gt, crs, nodata=np.nan)
    clear_frac = np.isfinite(s2[0]).mean()
    print(f"[{site}] S2 clear fraction {clear_frac:.3f}", flush=True)

    # --- Sentinel-1 RTC ---
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime="2020-01-01/2020-12-31").items())
    items = items[:: max(1, len(items) // MAX_S1)][:MAX_S1]
    print(f"[{site}] S1 scenes: {len(items)}", flush=True)
    s1ds = load(items, patch_url=pc.sign, bands=S1_BANDS, like=ds.odc.geobox, chunks={"x": 1024, "y": 1024}, groupby="solar_day",
                resampling="bilinear")
    s1 = []
    for b in S1_BANDS:
        v = s1ds[b].where(s1ds[b] > 0).astype("float32").median("time", skipna=True).compute().values
        s1.append(10 * np.log10(np.clip(v, 1e-6, None)))
        s1[-1][~np.isfinite(v)] = np.nan
    s1 = np.stack(s1).astype("float32")
    write(out / "s1.tif", s1, gt, crs, nodata=np.nan)

    # --- GMW label on the same grid: average resampling with GMW's 0 as nodata (VRT NoDataValue), so any
    # overlap with a mangrove cell gives 1 ---
    frac = np.zeros((h, w), "float32")
    with rasterio.open(GMW) as g:
        reproject(source=rasterio.band(g, 1), destination=frac, dst_transform=gt, dst_crs=crs,
                  resampling=Resampling.average, src_nodata=None, dst_nodata=0)
    label = (frac >= 0.5).astype("uint8")
    write(out / "label.tif", label, gt, crs)
    print(f"[{site}] done: {w}x{h} px, mangrove {label.mean():.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sites", nargs="+", choices=list(SITES), metavar="site",
                    help="one or more of: " + ", ".join(SITES))
    for s in ap.parse_args().sites:
        build(s)
