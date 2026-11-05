"""BraTS 2020 preprocessing pipeline (V5).

Steps per patient:
  brain mask -> bounding-box crop (15 voxel margin) -> isotropic 1 mm resampling
  -> pad/crop to (160, 192, 160) -> QC normalisation (z-score + min-max, clip, denoise)
  -> "soft" preprocessing for the AI models (median filter, 1-99 % clip, z-score INSIDE the brain,
     background = 0) -> one-hot [BG, NCR, ED, ET] -> split 70/15/15 -> NIfTI + CSV metadata.

The z-score is computed inside the brain only, so the background stays at 0 and `volume > 0`
is a valid brain mask.

Usage (Colab):
    from braintumorlab.preprocessing import ensure_brats2020_extracted, run_pipeline
    root = ensure_brats2020_extracted("/content/drive/MyDrive/miniProjetAI/archive.zip")
    run_pipeline(root, "/content/BraTS_2020_Data_preprocessing_nii_V5")
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_opening, gaussian_filter,
                           generate_binary_structure, median_filter, zoom)

SEG_CLASSES = [0, 1, 2, 4]
SEG_CLASS_NAMES = ["BG", "NCR/NET", "ED", "ET"]   # same order as SEG_CLASSES
TARGET_SHAPE = (160, 192, 160)
MODALITIES = ["flair", "t1", "t1ce", "t2"]


# ------------------------------------------------------------------
# 0) Archive extraction
# ------------------------------------------------------------------
def ensure_brats2020_extracted(zip_path, extract_to="/content"):
    """Extract the BraTS 2020 training folder from the Kaggle archive (only once)."""
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        roots = sorted({n.split("/")[0] for n in names if "/" in n})
        cands = [d for d in roots if "2020" in d and "BraTS" in d]
        target = cands[0] if cands else roots[0]
        dest = os.path.join(extract_to, target)
        if not os.path.isdir(dest):
            print(f"Extracting {target} ...")
            for m in names:
                if m.startswith(target + "/"):
                    zf.extract(m, extract_to)
    sub = os.path.join(dest, "MICCAI_BraTS2020_TrainingData")
    return sub if os.path.isdir(sub) else dest


# ------------------------------------------------------------------
# 1) Geometry
# ------------------------------------------------------------------
def compute_brain_mask(flair):
    """Light skull stripping (BraTS is already skull-stripped): threshold > 0 + morphology."""
    mask = flair > 0
    s = generate_binary_structure(3, 1)
    return binary_opening(binary_closing(mask, s, iterations=1), s, iterations=1)


def compute_bbox(mask, margin=15):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin, np.array(mask.shape) - 1)
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))


def resample_isotropic(img, zooms, spacing=1.0, order=1):
    f = tuple(float(z) / spacing for z in zooms[:3])
    return img if all(abs(x - 1) < 1e-6 for x in f) else zoom(img, f, order=order)


def pad_or_crop(img, shape=TARGET_SHAPE):
    pads = [((t - s) // 2, t - s - (t - s) // 2) if s < t else (0, 0) for s, t in zip(img.shape, shape)]
    if any(p != (0, 0) for p in pads):
        img = np.pad(img, pads)
    start = [max((s - t) // 2, 0) for s, t in zip(img.shape, shape)]
    return img[tuple(slice(a, a + t) for a, t in zip(start, shape))]


# ------------------------------------------------------------------
# 2) Intensity (QC)
# ------------------------------------------------------------------
def normalize_intensity(img, method="zscore_minmax", clip=(0.5, 99.5)):
    img = img.astype(np.float32)
    mask = img != 0
    if not mask.any():
        return img
    v = img[mask]
    if clip:
        v = np.clip(v, *np.percentile(v, clip))
    out = np.zeros_like(img)
    if method == "zscore":
        out[mask] = (v - v.mean()) / (v.std() + 1e-8)
    elif method == "minmax":
        out[mask] = (v - v.min()) / (np.ptp(v) + 1e-8)
    elif method == "zscore_minmax":
        z = (v - v.mean()) / (v.std() + 1e-8)
        out[mask] = (z - z.min()) / (np.ptp(z) + 1e-8)
    else:
        raise ValueError(f"Unknown normalisation: {method}")
    return out


def remove_outliers(img, low=0.5, high=99.5):
    return np.clip(img, *np.percentile(img, [low, high]))


def denoise(img, median_size=3, sigma=0.5):
    return gaussian_filter(median_filter(img, size=median_size), sigma=sigma)


def check_normalization(img, name, method="zscore_minmax", verbose=True):
    """zscore: test on brain voxels; min-max: test on the whole volume."""
    if method == "zscore":
        v = img[img != 0]
        ok = v.size > 0 and abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.1
    else:
        v = img.ravel()
        ok = v.min() >= -1e-3 and v.max() <= 1 + 1e-3
    if verbose:
        print(f"[{method}] {name}: mean={v.mean():.3f} std={v.std():.3f} -> {'OK' if ok else 'PROBLEM'}")
    return bool(ok)


def snr(values):
    s = float(np.std(values))
    return float(np.mean(values)) / s if s > 0 else 0.0


# ------------------------------------------------------------------
# 3) Quality control
# ------------------------------------------------------------------
def check_segmentation(seg, shape, pid):
    if seg.shape != shape:
        print(f"{pid}: segmentation not aligned {seg.shape} vs {shape}")
        return False
    vals = set(np.unique(np.rint(seg)).astype(int).tolist())
    if not vals <= set(SEG_CLASSES):
        print(f"{pid}: invalid labels {vals}")
        return False
    return True


# ------------------------------------------------------------------
# 4) Soft preprocessing for AI + one-hot + volumes
# ------------------------------------------------------------------
def preprocess_soft(img, brain, median_size=2, low=1.0, high=99.0):
    """Median filter + percentile clip + z-score computed INSIDE the brain; background = 0."""
    img = img.astype(np.float32)
    if not brain.any():
        return np.zeros_like(img)
    if median_size and median_size > 1:
        img = median_filter(img, size=median_size)
    v = np.clip(img[brain], *np.percentile(img[brain], [low, high]))
    out = np.zeros_like(img)
    out[brain] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def one_hot(seg):
    """Channels: 0=BG, 1=NCR/NET (label 1), 2=ED (label 2), 3=ET (label 4)."""
    oh = np.zeros(seg.shape + (4,), np.uint8)
    for i, c in enumerate(SEG_CLASSES):
        oh[..., i] = seg == c
    return oh


def tumor_volumes(seg, voxel_mm3=1.0):
    """Volumes (mm3 and mL) and ratios from labels 0/1/2/4."""
    seg = np.rint(seg).astype(np.int16)
    n = {"NCR": int((seg == 1).sum()), "ED": int((seg == 2).sum()), "ET": int((seg == 4).sum())}
    n["WT"] = n["NCR"] + n["ED"] + n["ET"]
    n["TC"] = n["NCR"] + n["ET"]
    out = {}
    for k, c in n.items():
        out[f"vol_{k}"] = c * voxel_mm3
        out[f"vol_{k}_ml"] = c * voxel_mm3 / 1000.0
    for k in ("ET", "ED", "NCR"):
        out[f"ratio_{k}_WT"] = out[f"vol_{k}"] / out["vol_WT"] if out["vol_WT"] else np.nan
    return out


# ------------------------------------------------------------------
# 5) Split + grade labels
# ------------------------------------------------------------------
def partition(names, ratios=(0.7, 0.15, 0.15), seed=42):
    names = list(names)
    np.random.RandomState(seed).shuffle(names)
    n = len(names)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    if n_tr + n_va > n - 1:
        n_tr = int(np.floor(ratios[0] * n))
        n_va = max(1, n - n_tr - 1)
    return {"train": names[:n_tr], "val": names[n_tr:n_tr + n_va], "test": names[n_tr + n_va:]}


def load_grade_map(root):
    """{patient_id: (grade_name, grade_label)} from name_mapping.csv (HGG=1, LGG=0)."""
    path = os.path.join(root, "name_mapping.csv")
    if not os.path.exists(path):
        print("name_mapping.csv not found -> grade UNK")
        return {}
    df = pd.read_csv(path)
    id_col = next((c for c in df.columns if "2020" in c and "ID" in c), None) or df.columns[-1]
    g_col = next(c for c in df.columns if "grade" in c.lower())
    out = {}
    for _, r in df.iterrows():
        g = str(r[g_col]).upper()
        out[str(r[id_col]).strip()] = ("HGG", 1) if "HGG" in g else ("LGG", 0) if "LGG" in g else ("UNK", -1)
    return out


# ------------------------------------------------------------------
# 6) Pipeline
# ------------------------------------------------------------------
def save_nii(vol, path, ref=None):
    import nibabel as nib

    v = np.asarray(vol, np.float32)
    affine = ref.affine if ref is not None else np.eye(4)
    nib.save(nib.Nifti1Image(v, affine), path)


def _find(folder, suffix):
    hits = glob.glob(os.path.join(folder, f"*_{suffix}.nii*"))
    return hits[0] if hits else None


def preprocess_patient(folder, shape=TARGET_SHAPE, spacing=1.0, normalization="zscore_minmax", verbose=False):
    import nibabel as nib

    pid = os.path.basename(folder.rstrip("/"))
    paths = {m: _find(folder, m) for m in MODALITIES + ["seg"]}
    if any(v is None for v in paths.values()):
        print(f"{pid}: missing file(s) {[k for k, v in paths.items() if v is None]} -> skipped")
        return None
    nii = {k: nib.load(p) for k, p in paths.items()}
    data = {k: np.asarray(v.dataobj, np.float32) for k, v in nii.items()}
    if not check_segmentation(data["seg"], data["flair"].shape, pid):
        return None
    bbox = compute_bbox(compute_brain_mask(data["flair"]), 15)
    fixed = {}
    for k in MODALITIES + ["seg"]:
        v = resample_isotropic(data[k][bbox], nii[k].header.get_zooms(), spacing, 0 if k == "seg" else 1)
        fixed[k] = pad_or_crop(v, shape)
    seg = np.rint(fixed["seg"]).astype(np.int16)
    brain = np.zeros(shape, bool)
    for m in MODALITIES:
        brain |= fixed[m] > 0
    brain |= seg > 0
    stats, soft = {}, {}
    for m in MODALITIES:
        norm = normalize_intensity(fixed[m], normalization)
        check_normalization(norm, f"{pid}-{m}", normalization, verbose)
        clean = denoise(remove_outliers(norm))
        stats[f"snr_{m}_after"] = snr(clean[brain])
        soft[m] = preprocess_soft(fixed[m], brain)
    return soft, seg, brain, stats, nii["flair"]


def run_pipeline(root, output_root, max_patients=None, seed=42, overwrite=False, make_zip=False):
    if overwrite and os.path.isdir(output_root):
        shutil.rmtree(output_root)
    grades = load_grade_map(root)
    folders = sorted(f for f in glob.glob(os.path.join(root, "BraTS20_Training_*")) if os.path.isdir(f))
    if not folders:
        raise RuntimeError(f"No BraTS20_Training_* folder in {root}")
    if max_patients:
        folders = sorted(np.random.RandomState(seed).choice(folders, min(max_patients, len(folders)), False))
    splits = partition([os.path.basename(f) for f in folders], seed=seed)
    subset = {n: s for s, ns in splits.items() for n in ns}
    print({k: len(v) for k, v in splits.items()})
    rows = []
    for i, folder in enumerate(folders, 1):
        pid = os.path.basename(folder)
        out_dir = os.path.join(output_root, subset[pid], pid)
        gname, glabel = grades.get(pid, ("UNK", -1))
        print(f"[{i}/{len(folders)}] {pid} ({gname}, {subset[pid]})")
        res = preprocess_patient(folder)
        if res is None:
            continue
        soft, seg, brain, stats, ref = res
        os.makedirs(out_dir, exist_ok=True)
        for m in MODALITIES:
            save_nii(soft[m], os.path.join(out_dir, f"{m}.nii.gz"), ref)
        save_nii(seg, os.path.join(out_dir, "seg.nii.gz"), ref)
        save_nii(one_hot(seg), os.path.join(out_dir, "seg_onehot.nii.gz"), ref)
        save_nii(brain.astype(np.float32), os.path.join(out_dir, "brain_mask.nii.gz"), ref)
        rows.append({"patient_id": pid, "year": 2020, "grade_name": gname, "grade_label": glabel,
                     "subset": subset[pid], "split_preproc": subset[pid],
                     "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     **tumor_volumes(seg, 1.0), **stats})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_root, "patients_metadata_2020.csv"), index=False)
    for f in ("name_mapping.csv", "survival_info.csv"):
        if os.path.exists(os.path.join(root, f)):
            shutil.copy(os.path.join(root, f), os.path.join(output_root, f))
    if make_zip:
        shutil.make_archive(output_root.rstrip("/"), "zip", output_root)
    print(f"Done: {len(df)}/{len(folders)} patients -> {output_root}")
    return df


def main():
    ap = argparse.ArgumentParser(description="BraTS 2020 preprocessing (V5)")
    ap.add_argument("--root", help="MICCAI_BraTS2020_TrainingData folder")
    ap.add_argument("--zip", help="archive.zip (used when --root is not given)")
    ap.add_argument("--out", default="/content/BraTS_2020_Data_preprocessing_nii_V5")
    ap.add_argument("--max-patients", type=int)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run_pipeline(a.root or ensure_brats2020_extracted(a.zip), a.out, a.max_patients, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
