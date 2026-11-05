"""Loading MRI volumes and segmentations (preprocessed dataset, uploads, DICOM).

Conventions used everywhere in the project
-------------------------------------------
* Arrays are reoriented to RAS (``nib.as_closest_canonical``) and indexed ``[x, y, z]``.
* BraTS labels: 0 = background, 1 = NCR/NET (necrotic core), 2 = ED (edema), 4 = ET (enhancing).
* One-hot files written by the preprocessing have channels ``[BG, NCR, ED, ET]``.
* Evaluation regions: WT = {1,2,4}, TC = {1,4}, ET = {4}.
* Background handling: older preprocessed files were z-scored on the whole volume, so the
  background is a constant non-zero value (about -0.4). ``remove_constant_background`` sets it to 0.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

MODALITIES = ["flair", "t1", "t1ce", "t2"]
MOD_LABEL = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1ce", "t2": "T2"}
LABELS = {1: "NCR/NET", 2: "ED", 4: "ET"}
REGIONS = ("WT", "TC", "ET")


# ------------------------------------------------------------------
# Dataset layout
# ------------------------------------------------------------------
def find_data_root(root) -> str:
    """Return the folder that contains train/, val/ and test/ (root itself or a direct child)."""
    root = str(root)

    def ok(p):
        return all(os.path.isdir(os.path.join(p, s)) for s in ("train", "val", "test"))

    if ok(root):
        return root
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub) and ok(sub):
            return sub
    raise FileNotFoundError(f"No train/val/test folders under {root}")


def list_patients(data_root, split=None, max_patients=None):
    """List of (patient_id, split, folder)."""
    out = []
    for s in ([split] if split else ["train", "val", "test"]):
        for d in sorted(glob.glob(os.path.join(str(data_root), s, "*"))):
            if os.path.isdir(d):
                out.append((os.path.basename(d), s, d))
    return out[:max_patients] if max_patients else out


def nii_path(folder, name):
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(str(folder), name + ext)
        if os.path.exists(p):
            return p
    return None


def load_nii(path, canonical=True):
    """(array float32, affine, voxel spacing in mm)."""
    import nibabel as nib

    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


# ------------------------------------------------------------------
# Intensities
# ------------------------------------------------------------------
def remove_constant_background(vol, tol=1e-4):
    """If the 8 corners share the same non-zero value, treat it as background and set it to 0.
    Returns (volume, foreground_mask)."""
    v = np.asarray(vol, dtype=np.float32)
    c = np.array([v[i, j, k] for i in (0, -1) for j in (0, -1) for k in (0, -1)])
    bg = float(c[0])
    if np.ptp(c) < tol:
        fg = np.abs(v - bg) > tol
        if abs(bg) > tol:
            v = np.where(fg, v, 0.0).astype(np.float32)
        return v, fg
    return v, np.abs(v) > tol


def zscore_in_mask(vol, mask):
    out = np.zeros_like(vol, dtype=np.float32)
    vals = vol[mask]
    if vals.size:
        out[mask] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return out


def to_uint8(vol, mask=None, p_low=0.5, p_high=99.5):
    """Window to [p_low, p_high] percentiles inside the brain and scale to 0-255 (for display)."""
    v = np.asarray(vol, dtype=np.float32)
    vals = v[mask] if mask is not None and mask.any() else v[v != 0]
    if vals.size == 0:
        return np.zeros(v.shape, np.uint8)
    lo, hi = np.percentile(vals, [p_low, p_high])
    out = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    if mask is not None:
        out = out * mask
    return out.astype(np.uint8)


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------
def labels_from_seg(seg):
    """Accept labels 0/1/2/4 (3D) or one-hot [BG, NCR, ED, ET] (4D, channel first or last)."""
    s = np.asarray(seg)
    if s.ndim == 3:
        return np.rint(s).astype(np.uint8)
    if s.ndim == 4:
        oh = s if s.shape[0] == 4 else np.moveaxis(s, -1, 0)
        lab = np.zeros(oh.shape[1:], np.uint8)
        lab[oh[1] > 0.5] = 1
        lab[oh[2] > 0.5] = 2
        lab[oh[3] > 0.5] = 4
        return lab
    raise ValueError(f"Unexpected segmentation shape {s.shape}")


def region_masks(labels):
    lab = np.asarray(labels)
    return {"WT": lab > 0, "TC": (lab == 1) | (lab == 4), "ET": lab == 4}


def label_masks(labels):
    lab = np.asarray(labels)
    return {"NCR": lab == 1, "ED": lab == 2, "ET": lab == 4}


def regions_to_labels(tc, wt, et):
    """Region channels (TC, WT, ET) -> BraTS labels."""
    lab = np.zeros(np.asarray(wt).shape, np.uint8)
    lab[np.asarray(wt) > 0] = 2
    lab[np.asarray(tc) > 0] = 1
    lab[np.asarray(et) > 0] = 4
    return lab


# ------------------------------------------------------------------
# Patient
# ------------------------------------------------------------------
def load_patient(folder, with_seg=True):
    """Load one preprocessed patient folder.

    Returns dict with keys flair/t1/t1ce/t2 (z-scored in the brain, background 0),
    seg (labels or None), brain (bool), affine, spacing, patient_id.
    """
    folder = str(folder)
    out = {"patient_id": os.path.basename(folder.rstrip("/")), "folder": folder, "missing": []}
    brain = None
    for m in MODALITIES:
        p = nii_path(folder, m)
        if p is None:
            out["missing"].append(m)
            continue
        v, aff, sp = load_nii(p)
        v, fg = remove_constant_background(v)
        out[m] = v
        brain = fg if brain is None else (brain | fg)
        out["affine"], out["spacing"] = aff, sp
    if brain is None:
        raise FileNotFoundError(f"No MRI modality found in {folder}")
    for m in out["missing"]:
        out[m] = np.zeros(brain.shape, np.float32)
    bm = nii_path(folder, "brain_mask")
    if bm:
        brain = load_nii(bm)[0] > 0.5
    out["seg"] = None
    if with_seg:
        sp_ = nii_path(folder, "seg")
        if sp_:
            out["seg"] = labels_from_seg(load_nii(sp_)[0])
        elif nii_path(folder, "seg_onehot"):
            import nibabel as nib

            img = nib.as_closest_canonical(nib.load(nii_path(folder, "seg_onehot")))
            out["seg"] = labels_from_seg(np.asarray(img.dataobj))
    if out["seg"] is not None:
        brain = brain | (out["seg"] > 0)
    out["brain"] = brain
    for m in MODALITIES:
        out[m] = zscore_in_mask(out[m], brain)
    return out


def stack_modalities(p, modalities=MODALITIES):
    return np.stack([p[m] for m in modalities], 0).astype(np.float32)


# ------------------------------------------------------------------
# Uploads (NIfTI or DICOM series)
# ------------------------------------------------------------------
def dicom_folder_to_nifti(folder, out_path):
    """Read the largest DICOM series found in `folder` and write it as NIfTI."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    best = None
    for root, _, _ in os.walk(folder):
        for sid in reader.GetGDCMSeriesIDs(root) or []:
            files = reader.GetGDCMSeriesFileNames(root, sid)
            if best is None or len(files) > len(best):
                best = files
    if not best:
        raise ValueError("No DICOM series found in the uploaded archive.")
    reader.SetFileNames(best)
    sitk.WriteImage(reader.Execute(), str(out_path))
    return out_path


def resample_to_reference(moving_path, reference_path, out_path, is_label=False):
    """Resample a NIfTI onto the grid of a reference NIfTI (same physical space assumed)."""
    import SimpleITK as sitk

    ref = sitk.ReadImage(str(reference_path))
    mov = sitk.ReadImage(str(moving_path))
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out = sitk.Resample(mov, ref, sitk.Transform(), interp, 0.0, mov.GetPixelID())
    sitk.WriteImage(out, str(out_path))
    return out_path


def prepare_uploaded_case(files: dict, out_dir):
    """files: {modality: path to .nii/.nii.gz or .zip of DICOM}. Writes <mod>.nii.gz in out_dir.

    All modalities are resampled onto the FLAIR grid. Intensities are normalised at load time.
    """
    import shutil
    import tempfile
    import zipfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name, path in files.items():
        path = str(path)
        dst = out_dir / f"_raw_{name}.nii.gz"
        if path.lower().endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            dicom_folder_to_nifti(tmp, dst)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            import nibabel as nib

            nib.save(nib.load(path), str(dst))
        raw[name] = dst
    if "flair" not in raw:
        raise ValueError("FLAIR is required (it is the reference grid).")
    for name, p in raw.items():
        target = out_dir / f"{name}.nii.gz"
        if name == "flair":
            shutil.copy(p, target)
        else:
            resample_to_reference(p, raw["flair"], target, is_label=(name == "seg"))
    for p in raw.values():
        os.remove(p)
    standardize_case(out_dir)
    return out_dir


def standardize_case(folder, target_shape=(160, 192, 160)):
    """Bring an uploaded case to the training geometry (same as the BraTS preprocessing):
    brain bounding-box crop (15 voxel margin), 1 mm isotropic resampling, pad/crop to 160x192x160.

    The models were trained on BraTS images, which are skull-stripped and co-registered. A raw
    clinical MRI (with skull) should be skull-stripped first (e.g. HD-BET) for reliable results.
    """
    import nibabel as nib

    from .preprocessing import compute_bbox, compute_brain_mask, pad_or_crop, resample_isotropic

    folder = Path(folder)
    names = [n for n in MODALITIES + ["seg"] if nii_path(folder, n)]
    imgs = {n: nib.as_closest_canonical(nib.load(nii_path(folder, n))) for n in names}
    flair = np.asarray(imgs["flair"].dataobj, np.float32)
    flair, _ = remove_constant_background(flair)
    bbox = compute_bbox(compute_brain_mask(flair), 15)
    zooms = imgs["flair"].header.get_zooms()[:3]
    for n, img in imgs.items():
        v = np.asarray(img.dataobj, np.float32)[bbox]
        v = pad_or_crop(resample_isotropic(v, zooms, 1.0, 0 if n == "seg" else 1), target_shape)
        if n == "seg":
            v = np.rint(v)
        nib.save(nib.Nifti1Image(v.astype(np.float32), np.diag([1.0, 1.0, 1.0, 1.0])), str(folder / f"{n}.nii.gz"))
