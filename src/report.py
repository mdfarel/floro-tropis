"""Collect results/*.json into one comparison (markdown + JSON), with the audit's reporting remedies.

    python -I src/report.py [--results-dir results] [--out results/summary]

Per model and test condition (clean / cloud50 / cloud100), mean and sd over seeds of:
  IoU mangrove pooled over all 400 test patches (CONTRACT main metric), mIoU, precision, recall,
  Matthews correlation (MCC) and Cohen's kappa (chance-corrected; a constant map scores 0),
  site-macro IoU (4 test sites weighted equally), pooled IoU without segara_anakan (sensitivity: 25 of its test
  patches were used in an input-scale trial during development), and per-site IoU.
Constant-map references (all mangrove, all other) are listed next to every condition.
Where per-patch confusion counts exist for every seed, a patch bootstrap (stratified by site, 2000 draws)
gives a 95% interval of the seed-mean pooled IoU.
Every result must be scored on the identical test set: 400 patches, 26,214,400 pixels, 5,541,164 mangrove,
6,553,600 pixels per site. Files that fail this check, or that were run with --limit, are refused.
results/unet.json (seeds 0-2) and results/unet_seeds34/unet.json (seeds 3-4) are merged into one 5-seed row.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["clean", "cloud50", "cloud100"]
SITES = ["segara_anakan", "kubu_raya", "mimika", "kendari"]
N_PIX, N_MANG, N_SITE = 26_214_400, 5_541_164, 6_553_600
ORDER = ["pixel-gbm", "pixel-gbm-sar", "unet", "floro-s2s1", "floro-s2only", "terramind-l", "prithvi2-300m",
         "floro-s2s1.dataset_raw"]
LABELS = {
    "pixel-gbm": "Per-pixel gradient boosting (S2+S1)",
    "pixel-gbm-sar": "Per-pixel gradient boosting, radar only (reference)",
    "unet": "U-Net from scratch (S2+S1)",
    "floro-s2s1": "FLORO frozen + head (S2+S1)",
    "floro-s2only": "FLORO frozen + head (S2 only)",
    "terramind-l": "TerraMind-L frozen + head (S2 12 bands + S1)",
    "prithvi2-300m": "Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1)",
    "floro-s2s1.dataset_raw": "Ablation: FLORO S2+S1 with dataset_raw input scaling",
}
NO_SAR = {"prithvi2-300m", "floro-s2only"}  # cloud100 = no input at all for these


# ------------------------------------------------------------------ confusion helpers (cm = [tn, fp, fn, tp])
def cm_scores(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    iou_m = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    iou_o = tn / (tn + fp + fn) if tn + fp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {"iou_mangrove": iou_m, "miou": (iou_m + iou_o) / 2, "precision": prec, "recall": rec,
            "mcc": mcc, "kappa": kappa, "pred_mangrove_share": (tp + fp) / n}


def site_cm(s):
    """Rebuild a per-site confusion matrix from metrics.scores output (exact up to float rounding)."""
    n = s["pixels"]
    pos = round(s["mangrove_share_truth"] * n)
    nan = float("nan")
    rec = nan if s["recall_mangrove"] is None else s["recall_mangrove"]  # results JSON writes NaN as null
    prec = nan if s["precision_mangrove"] is None else s["precision_mangrove"]
    tp = round(rec * pos) if rec == rec else 0
    fn = pos - tp
    if prec == prec and prec > 0:
        fp = round(tp / prec - tp)
    elif prec == prec:  # precision 0: predictions exist but none correct; recover fp from accuracy
        fp = n - pos - round(s["accuracy"] * n - tp) if s.get("accuracy") is not None else 0
    else:  # precision undefined: no mangrove predicted at this site
        fp = 0
    tn = n - tp - fn - fp
    return tn, fp, fn, tp


def agg(vals):
    v = np.array([x for x in vals if x is not None and x == x], float)
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0, "values": []}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size),
            "values": [round(float(x), 4) for x in v]}


# ------------------------------------------------------------------ loading + identity checks
def check_identity(name, d):
    problems = []
    lim = (d.get("training") or {}).get("limit") or (d.get("config") or {}).get("limit")
    if lim:
        problems.append(f"run with limit={lim}")
    for c, r in d.get("conditions", {}).items():
        for i, ps in enumerate(r.get("per_seed", [])):
            cm = np.array(ps["confusion_pooled"])
            if int(cm.sum()) != N_PIX or int(cm[1].sum()) != N_MANG:
                problems.append(f"{c} seed#{i}: {int(cm.sum())} px / {int(cm[1].sum())} mangrove")
            for s in SITES:
                if ps["per_site"].get(s, {}).get("pixels") != N_SITE:
                    problems.append(f"{c} seed#{i}: site {s} pixels {ps['per_site'].get(s, {}).get('pixels')}")
    if problems:
        raise SystemExit(f"{name}: not scored on the reference test set: " + "; ".join(problems[:5]))


def load(results_dir):
    res = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        if ".limit" in p.name or p.stem.startswith("summary"):
            continue
        d = json.loads(p.read_text())
        check_identity(p.stem, d)
        res[p.stem] = d
    extra = Path(results_dir) / "unet_seeds34" / "unet.json"
    if "unet" in res and extra.exists():
        e = json.loads(extra.read_text())
        check_identity("unet_seeds34", e)
        u = res["unet"]
        u["seeds"] = list(u["seeds"]) + list(e["seeds"])
        for c in CONDS:
            u["conditions"][c]["per_seed"] = u["conditions"][c]["per_seed"] + e["conditions"][c]["per_seed"]
        u["_merged_note"] = "seeds 0-2 from results/unet.json, seeds 3-4 from results/unet_seeds34/unet.json"
    return res


# ------------------------------------------------------------------ bootstrap over test patches (stratified by site)
def bootstrap_iou(per_seed_patches, sites_of_patch, n_boot=2000, seed=0):
    """per_seed_patches: list over seeds of (400, 4) [tn, fp, fn, tp]; returns 95% CI of the seed-mean pooled IoU."""
    rng = np.random.default_rng(seed)
    arr = np.stack([np.asarray(p, np.int64) for p in per_seed_patches])  # (S, 400, 4)
    groups = [np.flatnonzero(sites_of_patch == s) for s in SITES]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        tot = arr[:, pick].sum(axis=1)  # (S, 4)
        iou = tot[:, 3] / (tot[:, 3] + tot[:, 1] + tot[:, 2])
        vals[b] = iou.mean()
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def test_sites(patches_dir):
    import csv
    with open(Path(patches_dir) / "meta.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rows.sort(key=lambda r: int(r["idx"]))
    return np.array([r["site"] for r in rows])


# ------------------------------------------------------------------ main
def summarize(name, d, sites_of_patch):
    out = {}
    for c in CONDS:
        r = d.get("conditions", {}).get(c)
        if r is None:
            continue
        seeds = r["per_seed"]
        pooled = [cm_scores(*[int(x) for x in np.array(ps["confusion_pooled"]).ravel()]) for ps in seeds]
        persite = [{s: cm_scores(*site_cm(ps["per_site"][s])) for s in SITES} for ps in seeds]
        no_seg = []
        for ps in seeds:
            tot = np.sum([site_cm(ps["per_site"][s]) for s in SITES if s != "segara_anakan"], axis=0)
            no_seg.append(cm_scores(*[int(x) for x in tot])["iou_mangrove"])
        row = {k: agg([p[k] for p in pooled]) for k in pooled[0]}
        row["site_macro_iou"] = agg([np.mean([ps[s]["iou_mangrove"] for s in SITES]) for ps in persite])
        row["iou_without_segara"] = agg(no_seg)
        row["per_site"] = {s: {k: agg([ps[s][k] for ps in persite]) for k in ("iou_mangrove", "precision", "recall")}
                           for s in SITES}
        if all("per_patch_confusion" in ps and len(ps["per_patch_confusion"]) == 400 for ps in seeds):
            row["iou_mangrove_ci95_patch_bootstrap"] = bootstrap_iou([ps["per_patch_confusion"] for ps in seeds],
                                                                     sites_of_patch)
        out[c] = row
    return out


def f3(x):
    return "-" if x is None or x != x else f"{x:.3f}"


def ms(a):
    return "-" if a["n"] == 0 else (f"{a['mean']:.3f}" if a["n"] == 1 else f"{a['mean']:.3f} ± {a['sd']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--patches-dir", default=str(ROOT / "data/patches"))
    ap.add_argument("--out", default=str(ROOT / "results/summary"))
    a = ap.parse_args()
    res = load(a.results_dir)
    sites_of_patch = test_sites(a.patches_dir)
    names = [n for n in ORDER if n in res] + [n for n in res if n not in ORDER]
    S = {n: summarize(n, res[n], sites_of_patch) for n in names}
    share = N_MANG / N_PIX
    L = []
    L += ["# FLORO-Tropis pilot: test results", "",
          "4 unseen test sites (Segara Anakan, Kubu Raya, Mimika, Kendari), 400 patches of 256 x 256 px at 10 m, "
          f"{N_PIX:,} pixels, mangrove share {share:.3f} (Global Mangrove Watch v3 2020). Mean ± sd over seeds. "
          "Brackets: 95% patch-bootstrap interval of the seed-mean pooled IoU (stratified by site, 2000 draws), "
          "only where per-patch counts exist.", ""]
    for c, title in (("clean", "Clean imagery"), ("cloud50", "50% cloud (simulated, identical masks for all models)"),
                     ("cloud100", "Optical missing (radar only)")):
        L += [f"## {title}", "", "| Model | Seeds | Mangrove IoU | mIoU | MCC | Precision | Recall | Site-mean IoU | "
              "IoU without Segara Anakan |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            r = S[n].get(c)
            if r is None or (n == "pixel-gbm-sar" and c != "cloud100"):
                continue  # the radar-only reference ignores optical, so it belongs only in the radar-only table
            if c == "cloud100" and n in NO_SAR:
                L.append(f"| {LABELS.get(n, n)} | {r['iou_mangrove']['n']} | n/a (no radar, empty input) | | | | | | |")
                continue
            ci = r.get("iou_mangrove_ci95_patch_bootstrap")
            iou = ms(r["iou_mangrove"]) + (f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else "")
            L.append(f"| {LABELS.get(n, n)} | {r['iou_mangrove']['n']} | {iou} | {ms(r['miou'])} | {ms(r['mcc'])} | "
                     f"{ms(r['precision'])} | {ms(r['recall'])} | {ms(r['site_macro_iou'])} | {ms(r['iou_without_segara'])} |")
        L += [f"| Constant map: all mangrove | – | {share:.3f} | {share / 2:.3f} | 0 | {share:.3f} | 1 | – | – |",
              f"| Constant map: all non-mangrove | – | 0 | {(1 - share) / 2:.3f} | 0 | – | 0 | – | – |", ""]
    L += ["## Mangrove IoU per site, clean imagery", "", "| Model | " + " | ".join(SITES) + " |", "|---|" + "---|" * len(SITES)]
    for n in names:
        r = S[n].get("clean")
        if r and n != "pixel-gbm-sar":
            L.append(f"| {LABELS.get(n, n)} | " + " | ".join(ms(r["per_site"][s]["iou_mangrove"]) for s in SITES) + " |")
    L += ["", "Note: at Kendari about a third of the pixels labelled mangrove in GMW 2020 appear as aquaculture ponds "
          "or open land in the 2019-2021 imagery (NDVI < 0.3), so scores there measure agreement with the GMW map "
          "rather than the ability to recognise mangrove alone.", ""]
    L += ["## Per seed (pooled mangrove IoU)", "", "| Model | clean | cloud50 | cloud100 |", "|---|---|---|---|"]
    for n in names:
        cells = [", ".join(f"{v:.3f}" for v in S[n][c]["iou_mangrove"]["values"]) if c in S[n] else "-" for c in CONDS]
        L.append(f"| {LABELS.get(n, n)} | " + " | ".join(cells) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    Path(a.out + ".json").write_text(json.dumps({"models": S, "labels": LABELS,
                                                 "reference": {"pixels": N_PIX, "mangrove": N_MANG}}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
