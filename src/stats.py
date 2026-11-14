"""Recompute the statistics quoted in README.md from the published files only.

    .venv/bin/python -I src/stats.py [--results-dir results] [--patches-dir data/patches]

Inputs (all in the repository; no imagery, features or model weights are needed):
  results/<model>.json             per-seed confusion counts per test condition and per site
  results/unet_seeds34/unet.json   U-Net seeds 3-4, merged with results/unet.json exactly as src/report.py does
  results/summary.json             patch-bootstrap intervals and seed values (written by src/report.py)
  data/patches/meta.csv            test-patch sites, mangrove fraction and clear-sky fraction

Seed scores are recomputed from the confusion counts, not taken from the 4-decimal values in summary.json.
Tests used:
  Welch    two-sided Welch t-test on per-seed scores (scipy.stats.ttest_ind, equal_var=False).
  Perm     exact two-sided permutation test on the difference of seed means: every split of the pooled seed
           values into groups of the observed sizes is enumerated; p = share of splits whose |difference| is
           at least the observed one.
  Holm     Holm step-down correction, family-wise alpha 0.05, over the 14 headline Welch comparisons.
"""
import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import CONDS, N_MANG, N_PIX, N_SITE, SITES, cm_scores, load, site_cm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PATCH_PX = 256 * 256
ALPHA = 0.05
SHORT = {
    "pixel-gbm": "Gradient boosting (S2+S1)",
    "pixel-gbm-sar": "Gradient boosting, radar only",
    "unet": "U-Net (S2+S1)",
    "floro-s2s1": "FLORO (S2+S1)",
    "floro-s2only": "FLORO (S2 only)",
    "terramind-l": "TerraMind-L",
    "prithvi2-300m": "Prithvi-EO-2.0",
    "floro-s2s1.dataset_raw": "FLORO dataset_raw ablation",
}
GBM, UNET = "pixel-gbm", "unet"
FM3 = ["terramind-l", "floro-s2s1", "prithvi2-300m"]  # the three pipelines grouped in the README
DEEP4 = ["unet", "floro-s2s1", "terramind-l", "prithvi2-300m"]  # the README's "four deep models"
DEEP5 = DEEP4 + ["floro-s2only"]
MAIN6 = ["pixel-gbm", "unet", "floro-s2s1", "floro-s2only", "terramind-l", "prithvi2-300m"]
ABL = "floro-s2s1.dataset_raw"
NO_SAR = {"prithvi2-300m", "floro-s2only"}  # cloud100 = no input at all
EASY, HARD = ("kubu_raya", "mimika"), ("segara_anakan", "kendari")
# U-Net seed 3, first run (finished before a machine restart, then re-run). Its result is not in results/;
# these are the 3-decimal values quoted in the README, used only for the "keep the first run" sensitivity line.
UNET_SEED3_FIRST_RUN = {"clean": 0.743, "cloud50": 0.394}


# ------------------------------------------------------------------ seed scores from confusion counts
def seed_scores(d, cond):
    """Per-seed arrays for one model and condition: pooled IoU and MCC, site-mean IoU, IoU without Segara Anakan,
    per-site IoU (seeds x 4 sites). Same arithmetic as src/report.py."""
    out = {"iou": [], "mcc": [], "site_mean": [], "no_seg": [], "per_site": []}
    for ps in d["conditions"][cond]["per_seed"]:
        pooled = cm_scores(*[int(x) for x in np.array(ps["confusion_pooled"]).ravel()])
        cms = {s: site_cm(ps["per_site"][s]) for s in SITES}
        per_site = [cm_scores(*cms[s])["iou_mangrove"] for s in SITES]
        no_seg = cm_scores(*[int(x) for x in np.sum([cms[s] for s in SITES if s != "segara_anakan"], axis=0)])
        out["iou"].append(pooled["iou_mangrove"])
        out["mcc"].append(pooled["mcc"])
        out["site_mean"].append(float(np.mean(per_site)))
        out["no_seg"].append(no_seg["iou_mangrove"])
        out["per_site"].append(per_site)
    return {k: np.array(v, float) for k, v in out.items()}


# ------------------------------------------------------------------ tests
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    return float(t), float(df), float(p)


def perm_exact(a, b):
    """Exact two-sided permutation test on mean(a) - mean(b); returns (p, n_extreme, n_splits)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, na = np.concatenate([a, b]), a.size
    obs = abs(a.mean() - b.mean())
    total, hits = pool.sum(), 0
    splits = list(itertools.combinations(range(pool.size), na))
    for idx in splits:
        sa = pool[list(idx)].sum()
        diff = sa / na - (total - sa) / (pool.size - na)
        hits += abs(diff) >= obs - 1e-12
    return hits / len(splits), hits, len(splits)


def wins(a, b):
    """Number of (seed of a, seed of b) pairs in which a scores higher, and the number of pairs."""
    return int(sum(x > y for x in a for y in b)), len(a) * len(b)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj, order


def mde(sd, n1, n2, power=0.8, alpha=ALPHA):
    """Smallest true difference a two-sided t-test detects with the given power, both groups with sd `sd`
    (Welch df with equal sds; noncentral t)."""
    se = sd * math.sqrt(1 / n1 + 1 / n2)
    df = (1 / n1 + 1 / n2) ** 2 / ((1 / n1) ** 2 / (n1 - 1) + (1 / n2) ** 2 / (n2 - 1))
    tc = stats.t.ppf(1 - alpha / 2, df)

    def gap(delta):
        nc = delta / se
        lower = stats.nct.cdf(-tc, df, nc)  # negligible for delta > 0; scipy can return NaN far in the tail
        return stats.nct.sf(tc, df, nc) + (0.0 if lower != lower else lower) - power

    return float(optimize.brentq(gap, 1e-9 * se, 10 * se))


# ------------------------------------------------------------------ printing
def table(headers, rows, title=None):
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    if title:
        print(title)
    for j, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)))
        if j == 0:
            print("  " + "  ".join("-" * x for x in w))
    print()


def section(t):
    print("=" * 100)
    print(t)
    print("=" * 100)


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"


def fp(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def welch_row(a_name, b_name, a, b, extra=()):
    t, df, p = welch(a, b)
    return [*extra, SHORT[a_name], SHORT[b_name], f"{a.mean():.3f}", f"{b.mean():.3f}", f"{a.mean() - b.mean():+.3f}",
            f"{t:.2f}", f"{df:.1f}", fp(p)]


WELCH_H = ["A", "B", "mean A", "mean B", "A - B", "t", "df", "p (Welch)"]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", default="results", help="relative paths resolve against the repository root")
    ap.add_argument("--patches-dir", default="data/patches")
    a = ap.parse_args()
    a.results_dir, a.patches_dir = (str(ROOT / x) if not Path(x).is_absolute() else x for x in (a.results_dir, a.patches_dir))

    res = load(a.results_dir)
    summ = json.loads((Path(a.results_dir) / "summary.json").read_text())["models"]
    S = {n: {c: seed_scores(res[n], c) for c in CONDS if c in res[n]["conditions"]} for n in res}
    with open(Path(a.patches_dir) / "meta.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    test = [r for r in meta if r["split"] == "test"]

    rd = Path(a.results_dir).resolve()
    rd = rd.relative_to(ROOT) if rd.is_relative_to(ROOT) else rd
    print(f"Inputs: {len(res)} result files from {rd}/ "
          f"(U-Net seeds 0-4 merged), summary.json, meta.csv ({len(test)} test patches).")
    print("All scores are on the identical test set: 400 patches, 26,214,400 pixels (checked by src/report.py load).")
    print()

    # 1 -------------------------------------------------------------------------------------------------
    section("1. Headline table, recomputed from confusion counts (pooled mangrove IoU; cloud100 as MCC)")
    rows, maxdiff = [], 0.0
    for n in [k for k in SHORT if k in S]:
        r = [SHORT[n], len(S[n]["clean"]["iou"])]
        for c in ("clean", "cloud50"):
            r.append("-" if n == "pixel-gbm-sar" else ms(S[n][c]["iou"]))
            maxdiff = max(maxdiff, abs(S[n][c]["iou"].mean() - summ[n][c]["iou_mangrove"]["mean"]))
        r.append("n/a (no input)" if n in NO_SAR else f"{S[n]['cloud100']['mcc'].mean():.2f} ± "
                 f"{S[n]['cloud100']['mcc'].std(ddof=1):.2f}")
        rows.append(r)
    table(["Model", "Seeds", "Clean IoU", "Cloud50 IoU", "Cloud100 MCC"], rows)
    print(f"  Largest difference from the seed means in results/summary.json: {maxdiff:.1e}")
    print(f"  Constant all-mangrove map: IoU = mangrove share = {N_MANG / N_PIX:.4f} ({N_MANG:,} of {N_PIX:,} px)")
    print()

    # 2 -------------------------------------------------------------------------------------------------
    section("2. Clean imagery: deep models among themselves (Welch on per-seed pooled IoU)")
    rows = []
    for x, y in itertools.combinations(DEEP5, 2):
        tag = "4 main" if x in DEEP4 and y in DEEP4 else "with S2-only"
        rows.append(welch_row(x, y, S[x]["clean"]["iou"], S[y]["clean"]["iou"], extra=[tag]))
    rows.sort(key=lambda r: (r[0] != "4 main", float(r[-1])))
    table(["Set"] + WELCH_H, rows)
    p4 = [float(r[-1]) for r in rows if r[0] == "4 main"]
    print(f"  Among the 4 main deep models (U-Net, FLORO S2+S1, TerraMind-L, Prithvi): smallest p = {min(p4):.4f}; "
          f"pooled IoU spans {min(S[n]['clean']['iou'].mean() for n in DEEP4):.3f} to "
          f"{max(S[n]['clean']['iou'].mean() for n in DEEP4):.3f}.")
    print()

    # 3 -------------------------------------------------------------------------------------------------
    section("3. Clean imagery: each deep model vs gradient boosting (Welch on per-seed values)")
    for metric, label in (("iou", "pooled IoU"), ("site_mean", "site-mean IoU (4 sites weighted equally)"),
                          ("no_seg", "pooled IoU without Segara Anakan")):
        rows = [welch_row(n, GBM, S[n]["clean"][metric], S[GBM]["clean"][metric]) for n in DEEP5 + [ABL]]
        table(WELCH_H, rows, title=f"  {label}")

    # 4 -------------------------------------------------------------------------------------------------
    section("4. 50% simulated cloud: foundation-model pipelines vs the U-Net (per-seed pooled IoU)")
    u = S[UNET]["cloud50"]["iou"]
    print(f"  U-Net cloud50 seeds: {', '.join(f'{v:.3f}' for v in u)}")
    rows = []
    for n in FM3 + ["floro-s2only", ABL, GBM]:
        v = S[n]["cloud50"]["iou"]
        t, df, pw = welch(v, u)
        pp, hits, tot = perm_exact(v, u)
        w, nw = wins(v, u)
        rows.append([SHORT[n], len(v), f"{v.mean():.3f}", f"{v.mean() - u.mean():+.3f}", f"{t:.2f}", f"{df:.1f}",
                     fp(pw), f"{pp:.4f} ({hits}/{tot})", f"{w}/{nw}"])
    fm = np.concatenate([S[n]["cloud50"]["iou"] for n in FM3])
    t, df, pw = welch(fm, u)
    pp, hits, tot = perm_exact(fm, u)
    w, nw = wins(fm, u)
    rows.append(["3 FM pipelines as a class", len(fm), f"{fm.mean():.3f}", f"{fm.mean() - u.mean():+.3f}", f"{t:.2f}",
                 f"{df:.1f}", fp(pw), f"{pp:.4f} ({hits}/{tot})", f"{w}/{nw}"])
    table(["Model (vs U-Net)", "Seeds", "Mean", "- U-Net", "t", "df", "p (Welch)", "p (exact perm, splits)",
           "Seed pairs won"], rows)
    print("  Class = TerraMind-L, FLORO (S2+S1) and Prithvi seeds pooled (9) vs U-Net seeds (5). This grouping was")
    print("  chosen after seeing the results; the per-model rows are the pre-grouping comparisons.")
    print()

    # 5 -------------------------------------------------------------------------------------------------
    section("5. Holm correction over the 14 headline comparisons (Welch, per-seed pooled IoU, alpha 0.05)")
    fam = [(n, UNET, c) for c in ("clean", "cloud50") for n in FM3] + \
          [(n, GBM, c) for c in ("clean", "cloud50") for n in DEEP4]
    raw = []
    for x, y, c in fam:
        raw.append(welch(S[x][c]["iou"], S[y][c]["iou"])[2])
    adj, order = holm(np.array(raw))
    rows = []
    for rank, i in enumerate(order):
        x, y, c = fam[i]
        d = S[x][c]["iou"].mean() - S[y][c]["iou"].mean()
        thr = ALPHA / (len(fam) - rank)
        rows.append([rank + 1, f"{SHORT[x]} vs {SHORT[y]}", c, f"{d:+.3f}", fp(raw[i]), f"{thr:.4f}", fp(adj[i]),
                     "yes" if adj[i] < ALPHA else "no"])
    table(["Rank", "Comparison", "Condition", "A - B", "p raw", "Holm threshold", "p Holm", "Significant"], rows)
    surv = [f"{SHORT[fam[i][0]]} {'>' if S[fam[i][0]][fam[i][2]]['iou'].mean() > S[fam[i][1]][fam[i][2]]['iou'].mean() else '<'}"
            f" {SHORT[fam[i][1]]} ({fam[i][2]})" for i in order if adj[i] < ALPHA]
    print(f"  Remaining significant after Holm: {len(surv)}: " + "; ".join(surv))
    print()

    # 6 -------------------------------------------------------------------------------------------------
    section("6. Share of clean IoU kept under 50% cloud (cloud50 / clean)")
    rows = []
    for n in sorted([k for k in SHORT if k in S and k != "pixel-gbm-sar"],
                    key=lambda k: -np.mean(S[k]["cloud50"]["iou"] / S[k]["clean"]["iou"])):
        r = S[n]["cloud50"]["iou"] / S[n]["clean"]["iou"]
        rows.append([SHORT[n], len(r), f"{100 * r.mean():.1f}%",
                     f"{100 * S[n]['cloud50']['iou'].mean() / S[n]['clean']['iou'].mean():.1f}%",
                     f"{100 * r.min():.0f}-{100 * r.max():.0f}%", f"{S[n]['cloud50']['iou'].std(ddof=1):.3f}"])
    table(["Model", "Seeds", "Mean of seed ratios", "Ratio of means", "Seed range", "Cloud50 sd"], rows)
    print("  'Mean of seed ratios' is what results/figures/cloud50_retention.png shows (src/plot_results.py).")
    print()

    # 7 -------------------------------------------------------------------------------------------------
    section("7. Sites: per-site and site-mean IoU (clean, cloud50) and the share of variance explained by site (clean)")
    for c in ("clean", "cloud50"):
        rows = []
        for n in [k for k in SHORT if k in S and k != "pixel-gbm-sar"]:
            ps = S[n][c]["per_site"].mean(axis=0)
            rows.append([SHORT[n]] + [f"{v:.3f}" for v in ps] + [ms(S[n][c]["site_mean"]), ms(S[n][c]["no_seg"])])
        table(["Model"] + SITES + ["Site-mean IoU", "IoU without Segara"], rows,
              title=f"  {c}: seed-mean IoU per site; site-mean IoU and IoU without Segara Anakan as mean ± sd over seeds")
    for label, models in (("6 main pipelines (GBM, U-Net, FLORO S2+S1, FLORO S2 only, TerraMind-L, Prithvi)", MAIN6),
                          ("6 main pipelines + dataset_raw ablation", MAIN6 + [ABL])):
        M = np.array([S[n]["clean"]["per_site"].mean(axis=0) for n in models])  # models x sites
        g = M.mean()
        ss_tot = ((M - g) ** 2).sum()
        ss_site = M.shape[0] * ((M.mean(axis=0) - g) ** 2).sum()
        ss_model = M.shape[1] * ((M.mean(axis=1) - g) ** 2).sum()
        print(f"  {label}:")
        print(f"    model x site table of seed-mean IoU, two-way sums of squares: site {100 * ss_site / ss_tot:.1f}%, "
              f"model {100 * ss_model / ss_tot:.1f}%, model-by-site residual {100 * (ss_tot - ss_site - ss_model) / ss_tot:.1f}%")
    easy = [S[n]["clean"]["per_site"].mean(axis=0)[[SITES.index(s) for s in EASY]] for n in MAIN6]
    hard = [S[n]["clean"]["per_site"].mean(axis=0)[[SITES.index(s) for s in HARD]] for n in MAIN6]
    abl = S[ABL]["clean"]["per_site"].mean(axis=0)
    print(f"  Clean, 6 main pipelines: Kubu Raya and Mimika {np.min(easy):.3f}-{np.max(easy):.3f}; "
          f"Segara Anakan and Kendari {np.min(hard):.3f}-{np.max(hard):.3f}.")
    print(f"  Ablation: " + ", ".join(f"{s} {abl[i]:.3f}" for i, s in enumerate(SITES)))
    sm5 = [S[n]["clean"]["site_mean"].mean() for n in [GBM] + DEEP4]
    print(f"  Site-mean IoU of the five main pipelines (GBM, U-Net, FLORO S2+S1, TerraMind-L, Prithvi): "
          f"{min(sm5):.3f}-{max(sm5):.3f}")
    print()

    # 8 -------------------------------------------------------------------------------------------------
    section("8. Optical missing (cloud100): per-seed MCC and trivial single-site maps")
    m_json = {}
    for n, d in res.items():  # mangrove pixels per site from every per-seed record; must agree everywhere
        for c in d["conditions"]:
            for ps in d["conditions"][c]["per_seed"]:
                for s in SITES:
                    m_json.setdefault(s, set()).add(round(ps["per_site"][s]["mangrove_share_truth"] * ps["per_site"][s]["pixels"]))
    m_meta = {s: sum(round(float(r["mangrove_frac"]) * PATCH_PX) for r in test if r["site"] == s) for s in SITES}
    rows = []
    for s in SITES:
        rows.append([s, f"{N_SITE:,}", ", ".join(f"{v:,}" for v in sorted(m_json[s])), f"{m_meta[s]:,}",
                     f"{m_meta[s] / N_SITE:.3f}", f"{m_meta[s] / N_MANG:.3f}"])
    table(["Site", "Pixels", "Mangrove px (results JSON)", "Mangrove px (meta.csv)", "Mangrove share",
           "Share of all mangrove"], rows)
    consistent = all(len(m_json[s]) == 1 and m_json[s] == {m_meta[s]} for s in SITES) and sum(m_meta.values()) == N_MANG
    print(f"  Results JSON and meta.csv agree, total {sum(m_meta.values()):,} mangrove px: {consistent}")
    print(f"  Mangrove pixels at Kubu Raya + Mimika: {100 * (m_meta['kubu_raya'] + m_meta['mimika']) / N_MANG:.1f}%")
    print()
    rows, triv = [], {}
    for s in SITES:  # map that marks every pixel of one site as mangrove and everything else as other
        tp, fpx = m_meta[s], N_SITE - m_meta[s]
        fn, tn = N_MANG - tp, N_PIX - N_SITE - (N_MANG - tp)
        sc = cm_scores(tn, fpx, fn, tp)
        triv[s] = sc["mcc"]
        rows.append([f"only {s} = mangrove", f"{sc['mcc']:.4f}", f"{sc['iou_mangrove']:.4f}"])
    table(["Trivial map", "MCC", "IoU"], rows)
    t_mim = triv["mimika"]
    rows = []
    for n in [k for k in SHORT if k in S and k not in NO_SAR]:
        v = S[n]["cloud100"]["mcc"]
        rows.append([SHORT[n], ", ".join(f"{x:.3f}" for x in v), ms(v), f"{v.max():.4f}",
                     f"{int((v >= t_mim).sum())}/{v.size}", "yes" if v.mean() < t_mim else "no"])
    table(["Model", "MCC per seed", "Mean ± sd", "Best seed", f"Seeds >= {t_mim:.4f}", "Mean below Mimika map"], rows)

    # 9 -------------------------------------------------------------------------------------------------
    section("9. Other numbers quoted in the README")
    a_, b_ = S["floro-s2s1"]["cloud50"]["per_site"], S["floro-s2only"]["cloud50"]["per_site"]
    lower = sum(b_[j, k] < a_[i, k] for k in range(len(SITES)) for i in range(a_.shape[0]) for j in range(b_.shape[0]))
    print(f"  Cloud50, FLORO S2 only below FLORO S2+S1 in {lower} of {len(SITES) * a_.shape[0] * b_.shape[0]} "
          f"site-by-seed pairs (every S2+S1 seed vs every S2-only seed, per site).")
    rows, widths = [], {}
    for n in summ:  # as reported in results/summary.md (radar-only reference only at cloud100, no-input rows skipped)
        for c in CONDS:
            ci = (summ[n].get(c) or {}).get("iou_mangrove_ci95_patch_bootstrap")
            if not ci or (n == "pixel-gbm-sar" and c != "cloud100") or (n in NO_SAR and c == "cloud100"):
                continue
            widths.setdefault(c, []).append((ci[1] - ci[0]) / 2)
            rows.append([SHORT.get(n, n), c, f"[{ci[0]:.3f}, {ci[1]:.3f}]", f"{(ci[1] - ci[0]) / 2:.4f}"])
    table(["Model", "Condition", "95% patch-bootstrap interval", "Half-width"], rows,
          title="  Patch-bootstrap intervals of the seed-mean pooled IoU (results/summary.json; only runs with "
                "per-patch counts for every seed)")
    for c, w in widths.items():
        print(f"  Half-width range, {c}: {min(w):.4f} to {max(w):.4f}")
    vf = np.array([float(r["s2_valid_frac"]) for r in test])
    print(f"  Test patches: mean share of pixels with a clear-sky Sentinel-2 composite value (s2_valid_frac) "
          f"{100 * vf.mean():.2f}%")
    split = {k: sum(r["split"] == k for r in meta) for k in ("train", "val", "test")}
    print(f"  Patches per split (meta.csv): {split}")
    sds = [S[n]["clean"]["iou"].std(ddof=1) for n in DEEP4]
    sd = math.sqrt(np.mean(np.square(sds)))
    print(f"  Smallest clean-IoU difference detectable with 80% power (two-sided t-test, alpha 0.05), using the "
          f"root-mean-square seed sd of the 4 main deep models ({sd:.4f}): 3 vs 3 seeds {mde(sd, 3, 3):.3f}; "
          f"3 vs 5 seeds {mde(sd, 3, 5):.3f}")
    u_ids = list(res[UNET]["seeds"])
    for c in ("clean", "cloud50"):
        alt = S[UNET][c]["iou"].copy()
        alt[u_ids.index(3)] = UNET_SEED3_FIRST_RUN[c]
        print(f"  U-Net {c} with the first seed-3 run ({UNET_SEED3_FIRST_RUN[c]:.3f}, value quoted in the README) in "
              f"place of the re-run ({S[UNET][c]['iou'][u_ids.index(3)]:.3f}): {ms(alt)} (published: {ms(S[UNET][c]['iou'])})")
    u_extra = json.loads((Path(a.results_dir) / "unet_seeds34" / "unet.json").read_text())  # load() merges scores only
    be = {p["seed"]: p["best_epoch"] for p in res[UNET]["per_seed"] + u_extra["per_seed"]}
    print(f"  U-Net best epoch per seed (published runs): {be}")
    print(f"  Parameters: UPerNet head {res['floro-s2s1']['n_params_head']:,}; U-Net {res[UNET]['params']:,}")
    enc = {n: [v["sidecar"]["seconds_per_sample"] for k, v in res[n]["inputs"].items() if k.startswith("test_")]
           for n in res if "inputs" in res[n]}
    allv = [x for v in enc.values() for x in v]
    print(f"  Encoder feature extraction, test splits (sidecar seconds_per_sample): {min(allv):.2f}-{max(allv):.2f} s "
          f"per patch (" + "; ".join(f"{SHORT[n]} {min(v):.2f}-{max(v):.2f}" for n, v in enc.items()) + ")")
    ut = [p["test_seconds"] / (len(CONDS) * len(test)) for p in res[UNET]["per_seed"] + u_extra["per_seed"]]
    print(f"  U-Net test time per patch and condition (test_seconds / (3 conditions x 400 patches)), per seed: "
          + ", ".join(f"{x:.3f}" for x in ut) + " s")
    print()

    # 10 ------------------------------------------------------------------------------------------------
    section("10. README statements this script cannot check (they need files that are not published)")
    for line in (
            "Kendari: share of GMW 2020 mangrove pixels with NDVI < 0.3 (needs the patch arrays data/patches/optical.npy"
            " and label.npy, which are rebuilt by the scripts and not published).",
            "Label widening of the mapped area by the any-overlap rule (needs the GMW v3 rasters and site labels).",
            "Leave-one-site-out check of the FLORO input scale (8 of 8 cells): its code and output are not included.",
            "Head re-runs agreeing within 0.006 pooled IoU, and the first U-Net seed-3 run (0.743 clean, 0.394 cloud50,"
            " best epoch 2): these come from run logs, which are not published.",
    ):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
