"""Binary segmentation metrics for mangrove (1) vs other (0). Label 255 is ignored.

Usage:
    acc = ConfusionAccumulator()
    acc.update(pred, label, site="kubu_raya")   # pred, label: int arrays of the same shape
    report = acc.report()                       # pooled and per-site numbers
    summary = summarize_seeds([report_seed0, report_seed1, ...])
"""
from collections import defaultdict

import numpy as np

IGNORE = 255


def _cm(pred, label):
    pred = np.asarray(pred).astype(np.int64).ravel()
    label = np.asarray(label).astype(np.int64).ravel()
    keep = label != IGNORE
    pred, label = pred[keep], label[keep]
    if pred.size and (pred.min() < 0 or pred.max() > 1):
        raise ValueError("pred must be 0/1")
    return np.bincount(label * 2 + pred, minlength=4).reshape(2, 2)  # rows = truth, cols = pred


def scores(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    iou_m = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    iou_o = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    total = cm.sum()
    return {
        "iou_mangrove": iou_m,
        "iou_other": iou_o,
        "miou": float(np.nanmean([iou_m, iou_o])),
        "precision_mangrove": prec,
        "recall_mangrove": rec,
        "f1_mangrove": f1,
        "accuracy": (tp + tn) / total if total > 0 else float("nan"),
        "pixels": int(total),
        "mangrove_share_truth": (tp + fn) / total if total > 0 else float("nan"),
    }


class ConfusionAccumulator:
    def __init__(self):
        self.pooled = np.zeros((2, 2), np.int64)
        self.by_site = defaultdict(lambda: np.zeros((2, 2), np.int64))
        self.per_update = []  # one [tn, fp, fn, tp] per update() call (= per patch in every caller), for bootstrap CIs

    def update(self, pred, label, site=None):
        cm = _cm(pred, label)
        self.per_update.append([int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])])
        self.pooled += cm
        if site is not None:
            self.by_site[site] += cm

    def report(self):
        return {
            "pooled": scores(self.pooled),
            "per_site": {s: scores(cm) for s, cm in sorted(self.by_site.items())},
            "confusion_pooled": self.pooled.tolist(),
            "per_patch_confusion": self.per_update,
        }


def iou_report(preds, labels, sites):
    """preds, labels: (N, H, W) arrays; sites: length-N list of site names."""
    acc = ConfusionAccumulator()
    for p, l, s in zip(preds, labels, sites):
        acc.update(p, l, s)
    return acc.report()


def _agg(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan")}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def summarize_seeds(reports):
    """Mean and sd across seeds for pooled and per-site scores; keeps the per-seed reports."""
    keys = [k for k in reports[0]["pooled"] if k != "pixels"]
    out = {"pooled": {k: _agg([r["pooled"][k] for r in reports]) for k in keys}, "per_site": {}}
    for site in reports[0]["per_site"]:
        out["per_site"][site] = {k: _agg([r["per_site"][site][k] for r in reports]) for k in keys}
    out["per_seed"] = reports
    return out
