"""3D tumor segmentation with MONAI (3D U-Net, Attention U-Net, SegResNet, SwinUNETR).

Input : 4 channels (FLAIR, T1, T1ce, T2), z-scored inside the brain.
Output: 3 sigmoid channels following the BraTS regions [TC, WT, ET], converted back to labels 0/1/2/4.

Colab (T4) usage:
    from braintumorlab.segmentation import SegConfig, prepare_cache, train_segmentation, evaluate_model
    cfg = SegConfig(data_root=..., out_dir=..., model_name="unet3d", epochs=60)
    prepare_cache(cfg)            # once: writes float16 .npy volumes to local disk (fast loading)
    train_segmentation(cfg)
    evaluate_model(cfg.out_dir + "/unet3d/best.pt", cfg, split="test", eval_dir=...)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .config import seed_everything
from .metrics import region_metrics, summarize
from .volumes import MODALITIES, list_patients, load_patient, regions_to_labels

MODEL_NAMES = {"unet3d": "3D U-Net", "attention_unet": "Attention U-Net", "segresnet": "SegResNet",
               "swin_unetr": "SwinUNETR"}


@dataclass
class SegConfig:
    data_root: str = "/content/BraTS_2020_Data_preprocessing_nii_FIXED"
    cache_dir: str = "/content/seg_cache"
    out_dir: str = "/content/drive/MyDrive/miniProjetAI/models/segmentation"
    model_name: str = "unet3d"
    roi: tuple = (128, 128, 128)
    batch_size: int = 1
    samples_per_volume: int = 2
    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-5
    val_every: int = 2
    num_workers: int = 2
    max_train: int | None = None
    max_val: int | None = None
    use_amp: bool = True
    seed: int = 0


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
def build_model(name="unet3d", in_channels=4, out_channels=3, roi=(128, 128, 128)):
    from monai.networks import nets

    if name == "unet3d":
        return nets.UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                         channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
                         norm="instance", dropout=0.1)
    if name == "attention_unet":
        return nets.AttentionUnet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                                  channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), dropout=0.1)
    if name == "segresnet":
        return nets.SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                              init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2)
    if name == "swin_unetr":
        return nets.SwinUNETR(in_channels=in_channels, out_channels=out_channels, feature_size=24,
                              spatial_dims=3)
    raise ValueError(f"Unknown model {name}; choose from {list(MODEL_NAMES)}")


def save_checkpoint(path, model, cfg, epoch, best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_name": cfg.model_name, "state_dict": model.state_dict(), "config": asdict(cfg),
                "epoch": epoch, "best_val_dice": best}, path)


def load_seg_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = build_model(ck["model_name"], roi=tuple(cfg.get("roi", (128, 128, 128))))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def labels_to_regions(lab):
    lab = np.asarray(lab)
    return np.stack([(lab == 1) | (lab == 4), lab > 0, lab == 4]).astype(np.uint8)   # TC, WT, ET


def _cache_path(cfg, pid):
    return os.path.join(cfg.cache_dir, f"{pid}.npz")


def prepare_cache(cfg: SegConfig, overwrite=False):
    """Write image (float16, 4 channels) + label (uint8) per patient to local disk."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    pats = list_patients(cfg.data_root)
    for i, (pid, split, folder) in enumerate(pats, 1):
        out = _cache_path(cfg, pid)
        if os.path.exists(out) and not overwrite:
            continue
        p = load_patient(folder)
        img = np.stack([p[m] for m in MODALITIES]).astype(np.float16)
        np.savez(out, image=img, label=p["seg"].astype(np.uint8), spacing=np.array(p["spacing"]))
        if i % 25 == 0:
            print(f"cached {i}/{len(pats)}")
    print("cache ready:", cfg.cache_dir)


def _items(cfg, split, limit=None):
    items = [{"id": pid, "path": _cache_path(cfg, pid)} for pid, _, _ in list_patients(cfg.data_root, split)]
    items = [it for it in items if os.path.exists(it["path"])]
    return items[:limit] if limit else items


class _Load:
    def __call__(self, d):
        with np.load(d["path"]) as z:
            return {"image": z["image"].astype(np.float32), "label": labels_to_regions(z["label"]),
                    "labels": z["label"], "spacing": z["spacing"], "id": d["id"]}


def make_loaders(cfg: SegConfig):
    from monai.data import DataLoader, Dataset
    from monai.transforms import (Compose, RandCropByPosNegLabeld, RandFlipd, RandScaleIntensityd,
                                  RandShiftIntensityd, SpatialPadd)

    train_tf = Compose([
        _Load(),
        SpatialPadd(["image", "label"], cfg.roi),
        RandCropByPosNegLabeld(["image", "label"], "label", cfg.roi, pos=2, neg=1,
                               num_samples=cfg.samples_per_volume),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd("image", factors=0.1, prob=0.5),
        RandShiftIntensityd("image", offsets=0.1, prob=0.5),
    ])
    tr = Dataset(_items(cfg, "train", cfg.max_train), train_tf)
    va = Dataset(_items(cfg, "val", cfg.max_val), _Load())

    def collate_val(batch):
        return batch[0]

    return (DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
            torch.utils.data.DataLoader(va, batch_size=1, num_workers=cfg.num_workers, collate_fn=collate_val))


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
@torch.no_grad()
def predict_volume(model, image, roi=(128, 128, 128), device="cpu", overlap=0.5, use_amp=True):
    """image (4, X, Y, Z) -> (labels uint8, probabilities (3, X, Y, Z) float16)."""
    from monai.inferers import sliding_window_inference

    device = torch.device(device)
    x = torch.as_tensor(np.asarray(image, np.float32))[None].to(device)
    with torch.autocast(device.type, enabled=use_amp and device.type == "cuda"):
        logits = sliding_window_inference(x, roi, 2, model, overlap=overlap, mode="gaussian")
    prob = torch.sigmoid(logits.float())[0].cpu().numpy()
    brain = np.any(np.asarray(image) != 0, axis=0)   # no tumor outside the brain
    tc, wt, et = (prob > 0.5) & brain
    tc, et = tc & wt, et & wt   # enforce ET in TC in WT
    return regions_to_labels(tc | et, wt, et), prob.astype(np.float16)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_segmentation(cfg: SegConfig, resume=None):
    from monai.losses import DiceCELoss

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = make_loaders(cfg)
    print(f"train patches/epoch: {len(tr.dataset) * cfg.samples_per_volume} | val patients: {len(va.dataset)}")
    model = build_model(cfg.model_name, roi=cfg.roi).to(device)
    loss_fn = DiceCELoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = os.path.join(cfg.out_dir, cfg.model_name)
    best, start, hist = -1.0, 1, []
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start, best = ck["epoch"] + 1, ck.get("best_val_dice", -1)
    for ep in range(start, cfg.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in tr:
            x, y = b["image"].to(device), b["label"].to(device).float()
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + float(loss.detach()), n + 1
        sched.step()
        row = {"epoch": ep, "train_loss": tot / max(n, 1)}
        if ep % cfg.val_every == 0 or ep == cfg.epochs:
            model.eval()
            dices = []
            for case in va:
                lab, _ = predict_volume(model, case["image"], cfg.roi, device, use_amp=amp)
                m = region_metrics(lab, case["labels"], tuple(case["spacing"]))
                dices.append([m[r]["dice"] for r in ("WT", "TC", "ET")])
            d = np.mean(dices, 0) if dices else np.zeros(3)
            row.update({"val_dice_WT": float(d[0]), "val_dice_TC": float(d[1]), "val_dice_ET": float(d[2]),
                        "val_dice_mean": float(d.mean())})
            if d.mean() > best:
                best = float(d.mean())
                save_checkpoint(os.path.join(out, "best.pt"), model, cfg, ep, best)
        save_checkpoint(os.path.join(out, "last.pt"), model, cfg, ep, best)
        hist.append(row)
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        print(f"epoch {ep:03d} | loss {row['train_loss']:.4f}"
              + (f" | val Dice WT {row['val_dice_WT']:.3f} TC {row['val_dice_TC']:.3f} ET {row['val_dice_ET']:.3f}"
                 if "val_dice_mean" in row else "") + f" | {time.time() - t0:.0f}s")
    return os.path.join(out, "best.pt"), hist


def evaluate_model(ckpt, cfg: SegConfig, split="test", eval_dir=None, max_cases=None, save_predictions_to=None):
    """Dice / IoU / HD95 per region on a split; writes <eval_dir>/seg_<model>.json (read by the app)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_seg_model(ckpt, device)
    roi = tuple(ck.get("config", {}).get("roi", cfg.roi))
    per_case = []
    for it in _items(cfg, split, max_cases):
        case = _Load()(it)
        lab, _ = predict_volume(model, case["image"], roi, device)
        m = region_metrics(lab, case["labels"], tuple(case["spacing"]))
        per_case.append({"patient_id": it["id"], **m})
        if save_predictions_to:
            os.makedirs(save_predictions_to, exist_ok=True)
            np.savez_compressed(os.path.join(save_predictions_to, f"{it['id']}.npz"), labels=lab)
        print(f"{it['id']}: Dice WT {m['WT']['dice']:.3f} TC {m['TC']['dice']:.3f} ET {m['ET']['dice']:.3f}")
    result = {"task": "segmentation", "model": ck["model_name"],
              "model_label": MODEL_NAMES.get(ck["model_name"], ck["model_name"]), "split": split,
              "n_cases": len(per_case), "epoch": ck.get("epoch"), "summary": summarize(per_case),
              "per_case": per_case}
    if eval_dir:
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, f"seg_{ck['model_name']}.json"), "w") as f:
            json.dump(result, f, indent=1)
    return result
