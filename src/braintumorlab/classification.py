"""3D multi-task classification: tumor grade (LGG / HGG) + radiological pseudo-profile.

Pseudo-profiles (LGG-DA, LGG-OLIGO, HGG-ANA, HGG-GBM) are K-Means clusters of tumor volumes,
NOT histological subtypes. K-Means is fitted on the training split only and the names are
assigned deterministically (cluster with the lower ET/WT ratio -> DA / ANA).

A legacy FLAIR-only checkpoint (`.pkl`) can still be loaded; only its grade output is used.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .volumes import nii_path, remove_constant_background, zscore_in_mask

GRADE_TO_IDX = {"LGG": 0, "HGG": 1}
IDX_TO_GRADE = {v: k for k, v in GRADE_TO_IDX.items()}
SUBTYPES = ["LGG-DA", "LGG-OLIGO", "HGG-ANA", "HGG-GBM"]
SUBTYPE_TO_IDX = {n: i for i, n in enumerate(SUBTYPES)}
MOD_FILES = {"FLAIR": "flair", "T1": "t1", "T1ce": "t1ce", "T2": "t2"}
VOLUME_FEATURES = ["vol_WT", "vol_ET", "vol_ED", "vol_NCR", "ratio_ET_WT", "ratio_ED_WT", "ratio_NCR_WT"]


# ------------------------------------------------------------------
# Patient table + pseudo-profiles
# ------------------------------------------------------------------
def build_patient_table(meta_csv, volumes_df=None) -> pd.DataFrame:
    """Merge metadata (grade, split) with CORRECT volumes, without _x/_y duplicate columns."""
    meta = pd.read_csv(meta_csv)
    meta = meta[[c for c in meta.columns if not c.startswith(("vol_", "ratio_"))]]
    if volumes_df is not None:
        meta = meta.merge(volumes_df[[c for c in volumes_df.columns if c not in meta.columns or c == "patient_id"]],
                          on="patient_id", how="left")
    if "subset" not in meta.columns:
        meta["subset"] = meta["split_preproc"]
    meta["grade_label"] = pd.to_numeric(meta["grade_label"], errors="coerce").fillna(-1).astype(int)
    meta["grade_str"] = meta["grade_label"].map(IDX_TO_GRADE)
    return meta


def add_pseudo_profiles(df, features=None, seed=42, name_by="ratio_ET_WT"):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    feats = [c for c in (features or VOLUME_FEATURES) if c in df.columns]
    if len(feats) < 2:
        raise ValueError("At least two volume features are required (compute volumes first).")
    df["pseudo_profile"] = None
    for grade, (low, high) in {"LGG": ("LGG-DA", "LGG-OLIGO"), "HGG": ("HGG-ANA", "HGG-GBM")}.items():
        g = df[df["grade_str"] == grade].dropna(subset=feats)
        tr = g[g["subset"] == "train"]
        if len(tr) < 2:
            continue
        scaler = StandardScaler().fit(tr[feats])
        km = KMeans(2, random_state=seed, n_init=10).fit(scaler.transform(tr[feats]))
        lab = km.predict(scaler.transform(g[feats]))
        key = name_by if name_by in g.columns else feats[0]
        means = [g.loc[lab == k, key].mean() if (lab == k).any() else np.inf for k in (0, 1)]
        rank = np.argsort(means)
        names = {int(rank[0]): low, int(rank[1]): high}
        df.loc[g.index, "pseudo_profile"] = [names[int(k)] for k in lab]
    df["grade_idx"] = df["grade_str"].map(GRADE_TO_IDX)
    df["subtype_idx"] = df["pseudo_profile"].map(SUBTYPE_TO_IDX)
    return df


def split_tables(df):
    d = df.dropna(subset=["grade_idx", "subtype_idx"]).copy()
    d[["grade_idx", "subtype_idx"]] = d[["grade_idx", "subtype_idx"]].astype(int)
    return tuple(d[d["subset"] == s].reset_index(drop=True) for s in ("train", "val", "test"))


def build_file_index(data_root, modalities=("FLAIR",)):
    idx = {}
    for split in ("train", "val", "test"):
        d = os.path.join(str(data_root), split)
        if not os.path.isdir(d):
            continue
        for pid in sorted(os.listdir(d)):
            entry = {m: nii_path(os.path.join(d, pid), MOD_FILES.get(m, m)) for m in modalities}
            if all(entry.values()):
                idx[pid] = entry
    return idx


# ------------------------------------------------------------------
# Input tensor
# ------------------------------------------------------------------
def load_patient_tensor(entry, modalities, downsample=4, legacy=False):
    """(C, D, H, W) tensor.

    legacy=False (new models): background set to 0, z-score inside the brain, average pooling.
    legacy=True: reproduces the preprocessing used to train the old .pkl model
    (whole-volume z-score, strided down-sampling [::s]).
    """
    import nibabel as nib

    vols = []
    for m in modalities:
        v = np.asarray(nib.load(entry[m]).dataobj, np.float32)
        if legacy:
            v = v[::downsample, ::downsample, ::downsample] if downsample > 1 else v
            v = (v - v.mean()) / (v.std() + 1e-8) if v.std() > 0 else v * 0
        else:
            v, fg = remove_constant_background(v)
            v = zscore_in_mask(v, fg)
        vols.append(v)
    x = torch.from_numpy(np.stack(vols, 0).astype(np.float32))
    if not legacy and downsample > 1:
        x = F.avg_pool3d(x.unsqueeze(0), downsample, downsample).squeeze(0)
    return x


class BraTS3DClassificationDataset(Dataset):
    def __init__(self, df, file_index, modalities=("FLAIR",), downsample=4):
        self.modalities, self.file_index, self.downsample = list(modalities), file_index, downsample
        self.samples = [{"id": str(r["patient_id"]), "grade": int(r["grade_idx"]), "subtype": int(r["subtype_idx"])}
                        for _, r in df.iterrows() if str(r["patient_id"]) in file_index]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {"image": load_patient_tensor(self.file_index[s["id"]], self.modalities, self.downsample),
                "grade": torch.tensor(s["grade"]), "subtype": torch.tensor(s["subtype"]), "patient_id": s["id"]}


# ------------------------------------------------------------------
# Model (same layer names as the original notebook -> old weights load)
# ------------------------------------------------------------------
class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        ch = [in_channels, 8, 16, 32, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv3d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm3d(ch[i + 1]), nn.ReLU(inplace=True),
                       nn.MaxPool3d(2) if i < 3 else nn.AdaptiveAvgPool3d(1)]
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class MultiTaskGliomaClassifier(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, n_grade=2, n_subtype=4, dropout=0.3):
        super().__init__()
        self.encoder = Encoder3D(in_channels, latent_dim)
        self.drop = nn.Dropout(dropout)
        self.head_grade = nn.Linear(latent_dim, n_grade)
        self.head_subtype = nn.Linear(latent_dim, n_subtype)

    def forward(self, x):
        z = self.encoder(x)
        h = self.drop(z)
        return self.head_grade(h), self.head_subtype(h), z


class FocalLoss(nn.Module):
    """Focal loss. pt comes from the UNWEIGHTED cross-entropy; the class weight is applied after."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


# ------------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------------
def make_loaders(train_ds, val_ds, test_ds, batch_size=2, balance="sampler", num_workers=2):
    """balance='sampler' (oversample LGG) OR 'weights' (class weights in the loss), never both."""
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if balance == "sampler":
        g = np.array([s["grade"] for s in train_ds.samples])
        w = 1.0 / np.maximum(np.bincount(g, minlength=2)[g], 1)
        train = DataLoader(train_ds, sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                                   len(w), replacement=True), **kw)
    else:
        train = DataLoader(train_ds, shuffle=True, **kw)
    return train, DataLoader(val_ds, **kw), DataLoader(test_ds, **kw)


def make_criteria(train_df, balance="sampler", device="cpu", gamma=2.0):
    alpha = None
    if balance == "weights":
        c = train_df["grade_idx"].value_counts().reindex([0, 1]).fillna(0).to_numpy(float)
        alpha = torch.tensor([min(2.0, (c[1] + 1) / (c[0] + 1)), 1.0], dtype=torch.float32)
    sc = train_df["subtype_idx"].value_counts().reindex(range(4)).fillna(0).to_numpy(float)
    w = 1.0 / np.maximum(sc, 1)
    return (FocalLoss(alpha, gamma).to(device),
            nn.CrossEntropyLoss(weight=torch.tensor(w / w.sum() * 4, dtype=torch.float32).to(device)))


def run_epoch(model, loader, crit_g, crit_s, device, optimizer=None, alpha_task=2.0, beta_task=1.0):
    from sklearn.metrics import balanced_accuracy_score

    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    yg, pg, prob, ys, ps = [], [], [], [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            x, g, s = b["image"].to(device), b["grade"].to(device), b["subtype"].to(device)
            lg, ls, _ = model(x)
            loss = alpha_task * crit_g(lg, g) + beta_task * crit_s(ls, s)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot += float(loss.detach()) * x.size(0)
            n += x.size(0)
            yg += g.tolist(); pg += lg.argmax(1).tolist(); prob += lg.softmax(1)[:, 1].tolist()
            ys += s.tolist(); ps += ls.argmax(1).tolist()
    return {"loss": tot / max(n, 1), "grade_bal_acc": balanced_accuracy_score(yg, pg) if yg else np.nan,
            "subtype_bal_acc": balanced_accuracy_score(ys, ps) if ys else np.nan,
            "y_grade": yg, "p_grade": pg, "prob_hgg": prob, "y_sub": ys, "p_sub": ps}


def train_classifier(model, train_loader, val_loader, crit_g, crit_s, device, epochs=60, lr=1e-4,
                     out_path="best_classifier.pt", patience=10):
    """Model selection on the grade balanced accuracy (val loss is dominated by HGG). Early stopping."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    best, wait, hist = -1.0, 0, []
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, crit_g, crit_s, device, opt)
        va = run_epoch(model, val_loader, crit_g, crit_s, device)
        sched.step(va["grade_bal_acc"])
        hist.append({"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"],
                     "val_grade_bal_acc": va["grade_bal_acc"], "val_subtype_bal_acc": va["subtype_bal_acc"]})
        mark = ""
        if va["grade_bal_acc"] > best:
            best, wait, mark = va["grade_bal_acc"], 0, " *"
            save_model(model, out_path)
        else:
            wait += 1
        print(f"epoch {ep:03d} | train {tr['loss']:.3f} | val {va['loss']:.3f} | "
              f"grade bal-acc {va['grade_bal_acc']:.3f} | profile bal-acc {va['subtype_bal_acc']:.3f}{mark}")
        if patience and wait >= patience:
            print("Early stopping.")
            break
    return pd.DataFrame(hist)


def save_model(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_classifier(path, device="cpu"):
    """Load a .pt (torch.save) or an old .pkl (pickled state_dict). Input channels are inferred."""
    if str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            sd = pickle.load(f)
    else:
        sd = torch.load(path, map_location=device)
    model = MultiTaskGliomaClassifier(in_channels=sd["encoder.features.0.weight"].shape[1])
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def classification_report_dict(res) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y, p = res["y_grade"], res["p_grade"]
    out = {"task": "classification", "n": len(y), "grade_accuracy": accuracy_score(y, p),
           "grade_balanced_accuracy": balanced_accuracy_score(y, p),
           "grade_f1_macro": f1_score(y, p, average="macro", zero_division=0),
           "grade_confusion": confusion_matrix(y, p, labels=[0, 1]).tolist(),
           "profile_balanced_accuracy": balanced_accuracy_score(res["y_sub"], res["p_sub"])}
    if len(set(y)) == 2:
        out["grade_auc"] = roc_auc_score(y, res["prob_hgg"])
    return out


@torch.no_grad()
def predict_grade(model, entry, modalities, downsample=4, legacy=False, device="cpu"):
    model.eval()
    lg, ls, _ = model(load_patient_tensor(entry, modalities, downsample, legacy).unsqueeze(0).to(device))
    pg, ps = lg.softmax(1)[0].cpu().numpy(), ls.softmax(1)[0].cpu().numpy()
    grade = IDX_TO_GRADE[int(pg.argmax())]
    # keep the profile consistent with the predicted grade (HGG -> HGG-*, LGG -> LGG-*)
    same = [i for i, s in enumerate(SUBTYPES) if s.startswith(grade)]
    profile = SUBTYPES[max(same, key=lambda i: ps[i])] if same else SUBTYPES[int(ps.argmax())]
    return {"grade": grade, "prob_LGG": float(pg[0]), "prob_HGG": float(pg[1]),
            "pseudo_profile": profile,
            "profile_probs": {SUBTYPES[i]: float(p) for i, p in enumerate(ps)}}
