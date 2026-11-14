"""Record which satellite scenes each site used, in data/patches/scenes.json.

    python -I src/scenes.py [--sites <site> ...] [--out data/patches/scenes.json]

Sentinel-2: the STAC item ids that download_extra.py stored at download time as the `stac_items` tag of
  data/sites/<site>/s2extra.tif. download_extra.py repeats the search of download.py, and its B05 check proved that
  these scenes reproduce s2.tif exactly. The search is run again here (download_extra.search_s2, the same search,
  sort and cut as download.py) and the result is compared with the tags (`s2_search_reproduces_tags`).
Sentinel-1: download.py did not record which scenes it used. The ids are reconstructed by re-running its search and
  selection (sentinel-1-rtc, site bbox, 2020-01-01/2020-12-31, every len // MAX_S1-th item of the search response,
  first MAX_S1). This is valid only while Planetary Computer returns the same items in the same order;
  `search_matches` stores the size of the response, so a later change in the catalogue shows up as a different count.

Only the STAC search API is queried (no signing, no imagery is read). Needs data/sites/<site>/s2extra.tif.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import pystac_client
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import CATALOG, MAX_S1, MAX_S2  # noqa: E402  (same constants as the download)
from download_extra import search_s2  # noqa: E402  (same Sentinel-2 search, sort and cut as download.py)
from sites import SITES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITES_DIR = ROOT / "data/sites"
# Must stay identical to the Sentinel-1 search in download.py build().
S1_DATETIME = "2020-01-01/2020-12-31"


def search_s1(site):
    """Same search and the same items[::step][:MAX_S1] selection as download.py. Returns (selected, number found)."""
    cat = pystac_client.Client.open(CATALOG)  # unsigned: search only
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=SITES[site]["bbox"], datetime=S1_DATETIME).items())
    return items[:: max(1, len(items) // MAX_S1)][:MAX_S1], len(items)


def s2_tag_ids(site):
    path = SITES_DIR / site / "s2extra.tif"
    if not path.is_file():
        raise SystemExit(f"[{site}] {path.relative_to(ROOT)} missing: run src/download_extra.py {site} first")
    with rasterio.open(path) as d:
        tags = d.tags()
    if "stac_items" not in tags:
        raise SystemExit(f"[{site}] {path.relative_to(ROOT)} has no stac_items tag")
    return tags["stac_items"].split()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", nargs="+", choices=list(SITES), default=list(SITES), help="default: all 8 sites")
    p.add_argument("--out", default=str(ROOT / "data/patches/scenes.json"), help="output JSON")
    a = p.parse_args()

    today = datetime.date.today().isoformat()
    out = {"sentinel2": {}, "sentinel1": {}, "s2_search_reproduces_tags": {}}
    for site in a.sites:
        tag_ids = s2_tag_ids(site)
        search_ids = [i.id for i in search_s2(site)]
        same = search_ids == tag_ids
        out["sentinel2"][site] = {"ids": tag_ids, "source": "s2extra.tif tags written at download time"}
        if not same:
            out["sentinel2"][site][f"search_{today}_ids"] = search_ids
        out["s2_search_reproduces_tags"][site] = same

        s1, n_found = search_s1(site)
        out["sentinel1"][site] = {
            "ids": [i.id for i in s1],
            "source": f"reconstructed {today} by re-running the download.py search; not recorded at download time",
            "search_matches": n_found}
        print(f"[{site}] S2 {len(tag_ids)} ids (tags), search {'reproduces' if same else 'DIFFERS from'} the tags "
              f"({len(search_ids)} ids); S1 {len(s1)} of {n_found} search results", flush=True)
        if len(tag_ids) != MAX_S2 or len(s1) != MAX_S1:
            print(f"[{site}] WARNING: expected {MAX_S2} S2 and {MAX_S1} S1 scenes", flush=True)

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, dst)
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
