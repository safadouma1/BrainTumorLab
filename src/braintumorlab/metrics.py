"""Segmentation metrics per BraTS region: Dice, IoU, HD95 (mm), sensitivity, volume difference."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .volumes import REGIONS, region_masks


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def _surface(m):
    return m & ~binary_erosion(m)


def hd95(a, b, spacing=(1.0, 1.0, 1.0)):
    """95th percentile symmetric Hausdorff distance in mm. 0 if both empty, NaN if only one is."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = _surface(a), _surface(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def region_metrics(pred_labels, gt_labels, spacing=(1.0, 1.0, 1.0)) -> dict:
    p, g = region_masks(pred_labels), region_masks(gt_labels)
    out = {}
    for r in REGIONS:
        tp = (p[r] & g[r]).sum()
        out[r] = {"dice": float(dice(p[r], g[r])), "iou": float(iou(p[r], g[r])),
                  "hd95": hd95(p[r], g[r], spacing),
                  "sensitivity": float(tp / g[r].sum()) if g[r].sum() else float("nan"),
                  "volume_pred_ml": float(p[r].sum() * np.prod(spacing) / 1000),
                  "volume_gt_ml": float(g[r].sum() * np.prod(spacing) / 1000)}
    return out


def summarize(per_case: list) -> dict:
    """Mean / std / median over cases for every region and metric (NaN ignored)."""
    out = {}
    for r in REGIONS:
        out[r] = {}
        for k in ("dice", "iou", "hd95", "sensitivity"):
            v = np.array([c[r][k] for c in per_case], float)
            v = v[~np.isnan(v)]
            out[r][k] = ({"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
                          "n": int(v.size)} if v.size else None)
    return out
