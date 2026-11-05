"""Radiomics features (implemented with numpy / scipy / scikit-image, no PyRadiomics needed).

Shape      : volume, surface area, sphericity, compactness, max 3D diameter, elongation, flatness
First order: mean, std, median, P10, P90, IQR, skewness, kurtosis, entropy, energy (per modality)
Texture    : GLCM contrast, dissimilarity, homogeneity, energy, correlation (per modality,
             32 grey levels, averaged over the axial slices of the ROI and 4 directions)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .volumes import MOD_LABEL, MODALITIES, label_masks, region_masks


def shape_features(mask, spacing=(1.0, 1.0, 1.0)) -> dict:
    from scipy.spatial import ConvexHull
    from skimage.measure import marching_cubes, mesh_surface_area

    m = np.asarray(mask, bool)
    n = int(m.sum())
    vol = n * float(np.prod(spacing))
    if n < 10:
        return {"voxels": n, "volume_ml": vol / 1000}
    verts, faces, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5, spacing=spacing)
    area = float(mesh_surface_area(verts, faces))
    pts = np.argwhere(m) * np.asarray(spacing)
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    try:
        hull = verts[ConvexHull(verts).vertices]
        if len(hull) > 1500:
            hull = hull[np.random.RandomState(0).choice(len(hull), 1500, replace=False)]
        diam = float(np.sqrt(((hull[:, None] - hull[None]) ** 2).sum(-1)).max())
    except Exception:
        diam = float("nan")
    return {"voxels": n, "volume_ml": vol / 1000, "surface_area_cm2": area / 100,
            "sphericity": float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area),
            "surface_volume_ratio": area / vol,
            "max_3d_diameter_mm": diam,
            "major_axis_mm": float(4 * np.sqrt(ev[0])), "minor_axis_mm": float(4 * np.sqrt(ev[1])),
            "least_axis_mm": float(4 * np.sqrt(ev[2])),
            "elongation": float(np.sqrt(ev[1] / ev[0])), "flatness": float(np.sqrt(ev[2] / ev[0]))}


def first_order(vol, mask) -> dict:
    v = np.asarray(vol)[np.asarray(mask, bool)]
    if v.size < 2:
        return {}
    hist, _ = np.histogram(v, bins=32)
    p = hist[hist > 0] / v.size
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
            "skewness": float(stats.skew(v)), "kurtosis": float(stats.kurtosis(v)),
            "entropy": float(-(p * np.log2(p)).sum()), "energy": float((v ** 2).sum() / v.size)}


def glcm_features(vol, mask, levels=32) -> dict:
    from skimage.feature import graycomatrix, graycoprops

    m = np.asarray(mask, bool)
    if m.sum() < 20:
        return {}
    v = np.asarray(vol)
    lo, hi = np.percentile(v[m], [1, 99])
    q = (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * (levels - 1)).astype(np.uint8) + 1
    q[~m] = 0
    props = {k: [] for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")}
    zs = np.where(m.any((0, 1)))[0]
    for z in zs[:: max(1, len(zs) // 20)]:
        sl = q[:, :, z]
        if (sl > 0).sum() < 10:
            continue
        g = graycomatrix(sl, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels + 1, symmetric=True)
        g = g[1:, 1:].astype(np.float64)   # drop the background level
        s = g.sum(axis=(0, 1), keepdims=True)
        if (s == 0).any():
            continue
        g = g / s
        for k in props:
            props[k].append(float(np.nanmean(graycoprops(g, k))))
    return {f"glcm_{k}": float(np.mean(v)) for k, v in props.items() if v}


def compute_radiomics(p, labels=None) -> dict:
    """p: patient dict (volumes.load_patient). labels: segmentation to use (default p['seg'])."""
    lab = p["seg"] if labels is None else labels
    sp = tuple(p.get("spacing", (1.0, 1.0, 1.0)))
    regions = {**region_masks(lab), **{k: v for k, v in label_masks(lab).items() if k != "ET"}}
    out = {"shape": {r: shape_features(m, sp) for r, m in regions.items()},
           "first_order": {}, "texture": {}}
    out["shape"]["brain"] = {"volume_ml": float(p["brain"].sum() * np.prod(sp) / 1000)}
    out["shape"]["WT"]["tumor_brain_ratio_pct"] = 100 * float(regions["WT"].sum() / max(p["brain"].sum(), 1))
    healthy = p["brain"] & ~regions["WT"]
    for m in MODALITIES:
        out["first_order"][MOD_LABEL[m]] = {"WT": first_order(p[m], regions["WT"]),
                                           "healthy": first_order(p[m], healthy)}
        out["texture"][MOD_LABEL[m]] = glcm_features(p[m], regions["WT"])
    return out
