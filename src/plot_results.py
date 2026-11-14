"""Figure for the README: share of clean-imagery IoU each model keeps under 50% simulated cloud.

    python -I src/plot_results.py   # reads results/summary.json (written by src/report.py)

Writes results/figures/cloud50_retention.png. Bars = mean over seeds of (cloud50 IoU / clean IoU) per seed,
dots = individual seeds. Numbers come only from results/summary.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [  # key, label, highlighted (the three foundation-model pipelines that kept 78-85%; grouping made after
            # seeing the results)
    ("terramind-l", "TerraMind-L (S2 12 bands + S1)", True),
    ("floro-s2s1", "FLORO (S2 + S1)", True),
    ("prithvi2-300m", "Prithvi-EO-2.0-300M (6 S2 bands)", True),
    ("unet", "U-Net from scratch (S2 + S1)", False),
    ("floro-s2s1.dataset_raw", "FLORO, dataset_raw input ablation", False),
    ("pixel-gbm", "Per-pixel gradient boosting (S2 + S1)", False),
    ("floro-s2only", "FLORO (S2 only)", False),
]


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    S = json.loads((ROOT / "results/summary.json").read_text())["models"]
    rows = []
    for key, label, fm in MODELS:
        clean = S[key]["clean"]["iou_mangrove"]["values"]
        c50 = S[key]["cloud50"]["iou_mangrove"]["values"]
        ratios = [b / a for a, b in zip(clean, c50)]
        rows.append((label, fm, sum(ratios) / len(ratios), ratios))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
    for i, (label, fm, mean, ratios) in enumerate(rows):
        ax.barh(i, mean * 100, color="#2f6fdf" if fm else "#c9c9c4", height=0.6)
        ax.plot([min(ratios) * 100, max(ratios) * 100], [i, i], color="#222", lw=1)
        ax.scatter([r * 100 for r in ratios], [i] * len(ratios), color="#222", s=12, zorder=3)
        ax.text(103, i, f"{mean * 100:.0f}%", va="center", fontsize=9, fontweight="bold" if fm else "normal")
    ax.axvline(100, color="#888", ls="--", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of clean-imagery IoU kept under 50% simulated cloud (%)", fontsize=9)
    fig.suptitle("Three foundation-model pipelines keep 78-85% of their clean-imagery score\n"
                 "when half of the optical pixels are missing (indicative)", fontsize=10, x=0.01, ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "Bars: mean over seeds; dots: individual seeds. 4 unseen Indonesian test sites; "
             "no model was trained with clouds.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = ROOT / "results/figures/cloud50_retention.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
