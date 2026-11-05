"""Data analysis (report section 3bis): modality statistics, threshold baseline + ablation,
tumor volumes, tumor localisation, patient features and clustering.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from .preprocessing import tumor_volumes
from .volumes import MOD_LABEL, MODALITIES, list_patients, load_nii, load_patient, nii_path, labels_from_seg


# ------------------------------------------------------------------
# Per-modality statistics
# ------------------------------------------------------------------
def region_stats(vol, mask):
    v = vol[mask]
    return {"mean": float(v.mean()), "std": float(v.std())} if v.size else {"mean": np.nan, "std": np.nan}


def compute_snr(vol, mask):
    """mean / std inside `mask`. Not meaningful on z-scored data (mean ~ 0): prefer the CNR."""
    s = region_stats(vol, mask)
    return s["mean"] / s["std"] if s["std"] and s["std"] > 0 else np.nan


def compute_cnr(vol, tumor, healthy):
    """Contrast-to-noise ratio = (mean tumor - mean healthy) / std healthy."""
    t, h = region_stats(vol, tumor), region_stats(vol, healthy)
    return (t["mean"] - h["mean"]) / h["std"] if h["std"] and h["std"] > 0 else np.nan


def compute_contrast(vol, mask, p_low=1, p_high=99):
    """Robust contrast = P99 - P1 (max - min depends on a few outlier voxels)."""
    v = vol[mask]
    return float(np.percentile(v, p_high) - np.percentile(v, p_low)) if v.size else np.nan


def masks(p):
    tumor = p["seg"] > 0 if p["seg"] is not None else np.zeros_like(p["brain"])
    return p["brain"], tumor, p["brain"] & ~tumor


def modality_table(p) -> pd.DataFrame:
    brain, tumor, healthy = masks(p)
    rows = []
    for m in MODALITIES:
        v = p[m]
        h, t = region_stats(v, healthy), region_stats(v, tumor)
        rows.append({"modality": MOD_LABEL[m], "CNR_tumor": compute_cnr(v, tumor, healthy),
                     "contrast_P99_P1": compute_contrast(v, brain), "variance_brain": float(v[brain].var()),
                     "mean_healthy": h["mean"], "std_healthy": h["std"],
                     "mean_tumor": t["mean"], "std_tumor": t["std"],
                     "diff_mean_tumor_healthy": t["mean"] - h["mean"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Threshold baseline + ablation (15 modality combinations)
# ------------------------------------------------------------------
def _signed_score(p, mods):
    """Average of brain z-scores, each modality oriented so that tumor is 'higher'
    (tumor is hypo-intense in T1, so a single high threshold is wrong for T1)."""
    brain, tumor, healthy = masks(p)
    scores = []
    for m in mods:
        v = p[m][brain]
        z = (v - v.mean()) / (v.std() + 1e-8)
        sign = 1.0 if p[m][tumor].mean() >= p[m][healthy].mean() else -1.0
        scores.append(sign * z)
    return np.mean(scores, 0)


def threshold_baseline(p, mods, percentile=90):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

    brain, tumor, _ = masks(p)
    y = tumor[brain].astype(np.uint8)
    s = _signed_score(p, mods)
    pred = (s > np.percentile(s, percentile)).astype(np.uint8)
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = np.nan
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "auc": auc}


def ablation_study(p, percentile=90) -> pd.DataFrame:
    rows = []
    for r in range(1, 5):
        for combo in itertools.combinations(MODALITIES, r):
            name = " + ".join(MOD_LABEL[m] for m in combo) + (" only" if r == 1 else "")
            rows.append({"scenario": name, **threshold_baseline(p, combo, percentile)})
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# Volumes + localisation for the whole dataset
# ------------------------------------------------------------------
def load_seg(folder):
    path = nii_path(folder, "seg")
    if path:
        data, aff, sp = load_nii(path)
        return labels_from_seg(data), aff, sp
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(nii_path(folder, "seg_onehot")))
    return labels_from_seg(np.asarray(img.dataobj)), img.affine, img.header.get_zooms()[:3]


def volumes_table(data_root) -> pd.DataFrame:
    """Tumor volumes (mL) for every patient, computed from the label map (1 voxel = 1 mm^3)."""
    rows = []
    for pid, split, folder in list_patients(data_root):
        seg, _, sp = load_seg(folder)
        rows.append({"patient_id": pid, "split_preproc": split, **tumor_volumes(seg, float(np.prod(sp)))})
    return pd.DataFrame(rows)


def tumor_localization(seg, brain, affine, voxel_mm=1.0):
    """Hemisphere / anterior-posterior / superior-inferior / depth, using the NIfTI orientation."""
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt

    wt = np.asarray(seg) > 0
    if not wt.any():
        return None
    codes = nib.aff2axcodes(affine)
    axis = {}
    for ax, c in enumerate(codes):
        axis[{"L": "LR", "R": "LR", "A": "AP", "P": "AP", "S": "SI", "I": "SI"}[c]] = (ax, c)
    c_t, c_b = np.argwhere(wt).mean(0), np.argwhere(brain).mean(0)
    bmin, bmax = np.argwhere(brain).min(0), np.argwhere(brain).max(0)

    def toward(key, letter):
        ax, c = axis[key]
        r = (c_t[ax] - bmin[ax]) / max(bmax[ax] - bmin[ax], 1)
        return r if c == letter else 1 - r

    ax_lr, c_lr = axis["LR"]
    d = (c_t[ax_lr] - c_b[ax_lr]) * voxel_mm
    if abs(d) < 5:
        hemi = "midline / bilateral"
    else:
        side = c_lr if d > 0 else ("R" if c_lr == "L" else "L")
        hemi = "left hemisphere" if side == "L" else "right hemisphere"
    r_a, r_s = toward("AP", "A"), toward("SI", "S")
    dist = distance_transform_edt(brain) * voxel_mm
    depth = float(dist[wt].min())
    depth_cat = "superficial" if depth < 5 else "intermediate" if depth < 15 else "deep"
    lo, hi = np.argwhere(wt).min(0), np.argwhere(wt).max(0)
    return {"centroid_vox": [round(float(x), 1) for x in c_t],
            "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "hemisphere": hemi,
            "anterior_posterior": "anterior" if r_a > 2 / 3 else "posterior" if r_a < 1 / 3 else "central",
            "superior_inferior": "superior" if r_s > 2 / 3 else "inferior" if r_s < 1 / 3 else "middle",
            "depth_to_surface_mm": round(depth, 1), "depth_category": depth_cat}


def localization_table(data_root) -> pd.DataFrame:
    rows = []
    for pid, split, folder in list_patients(data_root):
        p = load_patient(folder)
        loc = tumor_localization(p["seg"], p["brain"], p["affine"])
        if loc:
            rows.append({"patient_id": pid, "split_preproc": split, **loc})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Patient features + clustering
# ------------------------------------------------------------------
FEATURE_CASES = ("all_modalities", "all_modalities_plus_tumor", "flair_only", "t1ce_only", "tumor_focus")


def patient_features(p, case="all_modalities") -> dict:
    brain, tumor, healthy = masks(p)
    mods = {"flair_only": ["flair"], "t1ce_only": ["t1ce"]}.get(case, MODALITIES)
    f = {}
    for m in mods:
        if case != "tumor_focus":
            f[f"CNR_{m}"] = compute_cnr(p[m], tumor, healthy)
            f[f"contrast_{m}"] = compute_contrast(p[m], brain)
        f[f"diff_mean_{m}"] = region_stats(p[m], tumor)["mean"] - region_stats(p[m], healthy)["mean"]
    if case == "all_modalities_plus_tumor":
        f["tumor_brain_ratio"] = tumor.sum() / max(brain.sum(), 1)
    return f


def features_table(data_root, split="train", max_patients=None, case="all_modalities"):
    rows = []
    for pid, _, folder in list_patients(data_root, split, max_patients):
        try:
            rows.append({"patient_id": pid, **patient_features(load_patient(folder), case)})
        except Exception as e:
            print(f"{pid} skipped: {e}")
    return pd.DataFrame(rows)


def cluster_patients(df, n_clusters=3, use_umap=True, seed=0):
    """K-Means on standardised features (not on the UMAP projection, which distorts
    distances). UMAP (or PCA if umap-learn is missing) is only used for the 2D plot."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in df.columns if c != "patient_id"]
    X = df[cols].to_numpy(np.float64)
    X = np.where(np.isnan(X), np.nanmean(X, 0), X)
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters, random_state=seed, n_init=10).fit_predict(Xs)
    emb = None
    if use_umap:
        try:
            import umap

            emb = umap.UMAP(n_neighbors=min(15, len(Xs) - 1), random_state=seed).fit_transform(Xs)
        except ImportError:
            print("umap-learn not installed -> PCA projection")
    if emb is None:
        emb = PCA(2, random_state=seed).fit_transform(Xs)
    out = df.copy()
    out["cluster"] = labels
    return out, emb


# ------------------------------------------------------------------
# Generation metrics (on REAL generated images)
# ------------------------------------------------------------------
def generation_metrics(real, gen, brain=None):
    """SSIM / PSNR after scaling both images to [0,1] with the real image's 1-99 percentiles."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    real, gen = np.asarray(real, np.float32), np.asarray(gen, np.float32)
    brain = np.ones(real.shape, bool) if brain is None else np.asarray(brain, bool)
    lo, hi = np.percentile(real[brain], [1, 99])

    def n(x):
        return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1) * brain

    r, g = n(real), n(gen)
    if np.allclose(r, g):
        return {"SSIM": 1.0, "PSNR": float("inf")}
    return {"SSIM": float(ssim(r, g, data_range=1.0)), "PSNR": float(psnr(r, g, data_range=1.0))}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def plot_modalities_with_seg(p, z=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    _, tumor, _ = masks(p)
    z = z if z is not None else int(np.argmax(tumor.sum((0, 1))))
    seg = p["seg"][:, :, z]
    cmap = ListedColormap(["none", "red", "limegreen", "none", "gold"])
    fig, ax = plt.subplots(4, 3, figsize=(10, 12))
    for r, m in enumerate(MODALITIES):
        sl = np.rot90(p[m][:, :, z])
        sg = np.ma.masked_equal(np.rot90(seg), 0)
        lo, hi = np.percentile(p[m][p["brain"]], [1, 99])
        ax[r, 0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
        ax[r, 0].set_title(f"{MOD_LABEL[m]} (slice {z})")
        ax[r, 1].imshow(sg, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax[r, 1].set_title("red=NCR/NET (1), green=ED (2), yellow=ET (4)", fontsize=8)
        ax[r, 2].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
        ax[r, 2].imshow(sg, cmap=cmap, vmin=0, vmax=4, alpha=0.5, interpolation="nearest")
        ax[r, 2].set_title(f"{MOD_LABEL[m]} + segmentation")
        for a in ax[r]:
            a.axis("off")
    fig.suptitle(p["patient_id"], fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_histograms(p, bins=100):
    import matplotlib.pyplot as plt

    _, tumor, healthy = masks(p)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, m in zip(axes, MODALITIES):
        h, t = p[m][healthy], p[m][tumor]
        lo, hi = np.percentile(np.concatenate([h, t]), [0.5, 99.5])
        ax.hist(h, bins=bins, range=(lo, hi), alpha=0.6, density=True, label="healthy")
        ax.hist(t, bins=bins, range=(lo, hi), alpha=0.6, density=True, label="tumor")
        ax.set_title(f"{MOD_LABEL[m]}  CNR={compute_cnr(p[m], tumor, healthy):.2f}")
        ax.legend()
    plt.tight_layout()
    plt.show()
