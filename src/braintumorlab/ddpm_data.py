"""Med-DDPM data: npz index + 2D slice dataset.

npz format (one file per patient):
    cond   (C, Z, Y, X) float32 - image channel(s) followed by 3 segmentation channels [NCR, ED, ET]
    target (1, Z, Y, X) float32 - T2
    patient_id (optional)

The dataset MedDDPM_dataset_T2 has C = 4: one MRI modality + 3 segmentation channels.
The target modality (T2) must never be inside `cond`, otherwise the model just copies it.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SPLIT_MAP = {"train": "train", "tr": "train", "training": "train", "val": "val", "valid": "val",
             "validation": "val", "dev": "val", "test": "test", "testing": "test"}


# ------------------------------------------------------------------
# Index
# ------------------------------------------------------------------
def _has_npz_col(path):
    try:
        return "npz_path" in pd.read_csv(path, nrows=3).columns
    except Exception:
        return False


def find_index_csv(root, max_depth=4):
    root = os.path.abspath(str(root))
    d0 = root.count(os.sep)
    named, other = [], []
    for cur, dirs, files in os.walk(root):
        if cur.count(os.sep) - d0 >= max_depth:
            dirs[:] = []
            continue
        for f in files:
            if f.lower().endswith(".csv") and _has_npz_col(os.path.join(cur, f)):
                (named if "index" in f.lower() else other).append(os.path.join(cur, f))
    found = sorted(named, key=lambda p: ("ready" not in p, p)) or sorted(other)
    if not found:
        raise FileNotFoundError(f"No CSV with an 'npz_path' column under {root}")
    return found[0]


def _remap(path, ddpm_root):
    """Index files store absolute Colab paths; remap them when the folder moved."""
    if os.path.exists(path):
        return path
    key = os.path.basename(str(ddpm_root).rstrip("/"))
    if key in path:
        cand = os.path.join(str(ddpm_root), path.split(key, 1)[1].lstrip("/"))
        if os.path.exists(cand):
            return cand
    return path


def load_index(ddpm_root, index_csv=None) -> pd.DataFrame:
    df = pd.read_csv(index_csv or find_index_csv(ddpm_root))
    col = next((c for c in ("split_preproc", "split") if c in df.columns), None)
    df["split_preproc"] = (df[col].astype(str).str.strip().str.lower().map(lambda x: SPLIT_MAP.get(x, x))
                           if col else "train")
    df["npz_path"] = [_remap(p, ddpm_root) for p in df["npz_path"]]
    df = df[df["npz_path"].map(os.path.exists)].reset_index(drop=True)
    if "patient_id" not in df.columns:
        df["patient_id"] = [os.path.splitext(os.path.basename(p))[0] for p in df["npz_path"]]
    df["patient_id"] = df["patient_id"].astype(str)
    if (df["split_preproc"] == "val").sum() == 0 and len(df) > 1:
        rng = np.random.RandomState(0)
        ids = df["patient_id"].unique()
        rng.shuffle(ids)
        df.loc[df["patient_id"].isin(ids[: max(1, len(ids) // 10)]), "split_preproc"] = "val"
    return df


# ------------------------------------------------------------------
# 2D helpers
# ------------------------------------------------------------------
def dilate_mask(m, k=7):
    return F.max_pool2d(m.unsqueeze(0), k, 1, k // 2).squeeze(0)


def erode_mask(m, k=7):
    return 1.0 - dilate_mask(1.0 - m, k)


def edge_from_mask(m, k=7):
    return (dilate_mask(m, k) - erode_mask(m, k)).clamp(0, 1)


def split_cond(c):
    """The last 3 channels of cond are the segmentation [NCR, ED, ET]."""
    return (c[:-3], c[-3:]) if c.shape[0] >= 4 else (c, None)


def remove_constant_background_2d(x, tol=1e-4):
    out = x.clone()
    for i in range(x.shape[0]):
        ch = x[i]
        corners = torch.stack([ch[0, 0], ch[0, -1], ch[-1, 0], ch[-1, -1]])
        if (corners - corners[0]).abs().max() < tol and corners[0].abs() > tol:
            out[i] = torch.where((ch - corners[0]).abs() > tol, ch, torch.zeros_like(ch))
    return out


def _resize(x, size, mode):
    return F.interpolate(x.unsqueeze(0), size=size, mode=mode,
                         **({"align_corners": False} if mode == "bilinear" else {})).squeeze(0)


def resize_cond(c, size):
    img, seg = split_cond(c)
    img = _resize(img, size, "bilinear")
    return torch.cat([img, (_resize(seg, size, "nearest") > 0.5).float()], 0) if seg is not None else img


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class SliceDataset(Dataset):
    """One 2D slice per patient per call. slice_mode: random | mid | tumor (largest tumor area)."""

    def __init__(self, index_df, split="train", max_patients=None, slice_mode="random", slice_axis="X",
                 augment=False, clip_value=3.0, img_threshold=0.05, crop_to_brain=True, crop_margin=12,
                 out_size=192, edge_k=7, soft_k=11, fix_background=True):
        df = index_df[index_df["split_preproc"] == split].reset_index(drop=True)
        if max_patients:
            df = df.head(max_patients)
        if len(df) == 0:
            raise RuntimeError(f"No sample for split={split}")
        self.df = df
        self.slice_mode, self.axis = slice_mode, slice_axis.upper()
        self.augment, self.clip = augment, float(clip_value)
        self.img_threshold, self.crop, self.margin = img_threshold, crop_to_brain, crop_margin
        self.out_size, self.edge_k, self.soft_k = out_size, edge_k, soft_k
        self.fix_background = fix_background
        self.pid_to_idx = {p: i for i, p in reversed(list(enumerate(df["patient_id"])))}

    def __len__(self):
        return len(self.df)

    def patient_ids(self):
        return list(dict.fromkeys(self.df["patient_id"]))

    def volume_shape(self, pid):
        with np.load(self.df.iloc[self.pid_to_idx[str(pid)]]["npz_path"]) as d:
            return d["cond"].shape

    def _pick(self, size, seg3):
        if self.slice_mode == "tumor" and seg3 is not None:
            axes = {"X": (0, 1, 2), "Y": (0, 1, 3), "Z": (0, 2, 3)}[self.axis]
            area = (seg3 > 0.5).sum(axis=axes)
            if area.max() > 0:
                return int(area.argmax())
        return size // 2 if self.slice_mode in ("mid", "tumor") else int(np.random.randint(size))

    def load(self, idx, slice_index=None):
        row = self.df.iloc[idx]
        with np.load(row["npz_path"], allow_pickle=True) as d:
            cond3 = d["cond"].astype(np.float32)
            tgt3 = d["target"].astype(np.float32)
        _, Z, Y, X = cond3.shape
        seg3 = cond3[-3:] if cond3.shape[0] >= 4 else None
        s = slice_index if slice_index is not None else self._pick({"X": X, "Y": Y, "Z": Z}[self.axis], seg3)
        sl = {"X": (slice(None), slice(None), slice(None), s), "Y": (slice(None), slice(None), s),
              "Z": (slice(None), s)}[self.axis]
        cond = torch.from_numpy(np.ascontiguousarray(cond3[sl]))
        tgt = torch.from_numpy(np.ascontiguousarray(tgt3[sl]))
        img, seg = split_cond(cond)
        if self.fix_background:
            tgt = remove_constant_background_2d(tgt)
            img = remove_constant_background_2d(img)
            cond = torch.cat([img, seg], 0) if seg is not None else img
        brain = (tgt.abs() > 1e-6).squeeze(0)
        a = img.abs()
        brain |= a.sum(0) > max(self.img_threshold, float(torch.quantile(a, 0.7)) * 0.1)
        tumor = None
        if seg is not None:
            tumor = (seg > 0.5).any(0)
            brain |= tumor
            tumor = tumor.unsqueeze(0).float()
        bm = brain.unsqueeze(0).float()
        if self.crop:
            ys, xs = torch.where(bm[0] > 0.5)
            if ys.numel():
                H, W = bm.shape[1:]
                y0, y1 = max(int(ys.min()) - self.margin, 0), min(int(ys.max()) + self.margin + 1, H)
                x0, x1 = max(int(xs.min()) - self.margin, 0), min(int(xs.max()) + self.margin + 1, W)
                cond, tgt, bm = cond[:, y0:y1, x0:x1], tgt[:, y0:y1, x0:x1], bm[:, y0:y1, x0:x1]
                tumor = tumor[:, y0:y1, x0:x1] if tumor is not None else None
        if self.out_size:
            hw = (self.out_size, self.out_size)
            cond, tgt, bm = resize_cond(cond, hw), _resize(tgt, hw, "bilinear"), _resize(bm, hw, "nearest")
            tumor = _resize(tumor, hw, "nearest") if tumor is not None else None
        img, seg = split_cond(cond)
        img = torch.clamp(img, -self.clip, self.clip) / self.clip * bm
        cond = torch.cat([img, (seg > 0.5).float()], 0) if seg is not None else img
        tgt = torch.clamp(tgt, -self.clip, self.clip) / self.clip * bm
        edge, soft = edge_from_mask(bm, self.edge_k), dilate_mask(bm, self.soft_k)
        if self.augment:
            for dim in (2, 1):
                if torch.rand(()) < 0.5:
                    cond, tgt, bm, edge, soft = (torch.flip(t, [dim]) for t in (cond, tgt, bm, edge, soft))
                    tumor = torch.flip(tumor, [dim]) if tumor is not None else None
        out = {"cond": cond, "target": tgt, "brain_mask": bm, "edge_mask": edge, "soft_mask": soft,
               "patient_id": str(row["patient_id"]), "slice_index": int(s)}
        if tumor is not None:
            out["tumor_mask"] = tumor
        return out

    def __getitem__(self, idx):
        return self.load(idx)

    def get_by_pid(self, pid, slice_index=None):
        if str(pid) not in self.pid_to_idx:
            return None
        s = self.load(self.pid_to_idx[str(pid)], slice_index)
        return {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in s.items()}


def make_loaders(train_ds, val_ds, batch_size=8, num_workers=2):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return (DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size, shuffle=False, **kw))


# ------------------------------------------------------------------
# Building a new npz dataset from the preprocessed NIfTI
# ------------------------------------------------------------------
def build_npz_dataset(data_root, out_root, cond_modalities=("flair", "t1", "t1ce")):
    """cond = [cond_modalities..., NCR, ED, ET] (Z, Y, X order), target = T2. Writes an index CSV."""
    from .volumes import list_patients, load_patient

    os.makedirs(os.path.join(out_root, "npz"), exist_ok=True)
    rows = []
    for pid, split, folder in list_patients(data_root):
        p = load_patient(folder)
        zyx = lambda v: np.transpose(v, (2, 1, 0))  # noqa: E731
        seg = p["seg"]
        cond = np.stack([zyx(p[m]) for m in cond_modalities] + [zyx(seg == k) for k in (1, 2, 4)]).astype(np.float32)
        path = os.path.join(out_root, "npz", f"{pid}.npz")
        np.savez_compressed(path, cond=cond, target=zyx(p["t2"])[None].astype(np.float32), patient_id=pid)
        rows.append({"patient_id": pid, "npz_path": path, "split_preproc": split})
    idx = os.path.join(out_root, "index_MedDDPM_T2.csv")
    pd.DataFrame(rows).to_csv(idx, index=False)
    return idx
