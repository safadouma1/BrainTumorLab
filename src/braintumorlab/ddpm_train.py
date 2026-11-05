"""Med-DDPM 2D training (same settings as the original cell F: eps prediction, T=300, base 48,
EMA 0.999, constraint weights W, 43 epochs, validation every 2 epochs)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from .config import seed_everything
from .ddpm_data import SliceDataset, load_index, make_loaders
from .diffusion import EMA, DiffusionScheduler2D, UNet2DCond
from .losses import constraint_pack


@dataclass
class DDPMConfig:
    ddpm_root: str = "/content/drive/MyDrive/miniProjetAI/MedDDPM_dataset_T2"
    index_csv: str = ""
    models_dir: str = ""
    seed: int = 0
    epochs: int = 43
    batch_size: int = 8
    lr: float = 2e-4
    num_timesteps: int = 300
    base: int = 48
    time_dim: int = 128
    dropout: float = 0.1
    ema_decay: float = 0.999
    val_every: int = 2
    ckpt_every: int = 5
    slice_axis: str = "X"
    out_size: int = 192
    fix_background: bool = True
    num_workers: int = 2
    use_amp: bool = True
    weights: dict = field(default_factory=lambda: {
        "c01": 0.25, "c02": 0.10, "c03": 0.05, "c04": 0.08, "c05": 0.04, "c07": 0.015, "c08": 0.010,
        "c10": 0.02, "c11": 0.03, "c12": 0.03, "c15": 0.15, "c16": 0.08, "c17": 0.03, "c18": 0.02,
        "c19": 0.03, "tumor_l1": 0.12, "c21": 0.08, "c22": 0.05, "c23": 0.02})


def _step(model, sched, batch, cfg, device):
    x0, cond = batch["target"], batch["cond"]
    t = torch.randint(0, sched.num_timesteps, (x0.shape[0],), device=device)
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    eps = model(x_t, cond, t)
    x0_hat = sched.predict_x0(x_t, t, eps).clamp(-1, 1)
    pack, _ = constraint_pack(x0_hat.float(), x0.float(), batch, cfg.weights, pred=eps)
    return F.mse_loss(eps, noise) + pack


def train_ddpm(cfg: DDPMConfig, resume=None):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_index(cfg.ddpm_root, cfg.index_csv or None)
    common = dict(slice_axis=cfg.slice_axis, out_size=cfg.out_size, fix_background=cfg.fix_background)
    tr_ds = SliceDataset(df, "train", slice_mode="random", augment=True, **common)
    va_ds = SliceDataset(df, "val", slice_mode="tumor", **common)
    tr, va = make_loaders(tr_ds, va_ds, cfg.batch_size, cfg.num_workers)
    cond_ch = tr_ds[0]["cond"].shape[0]
    models_dir = cfg.models_dir or os.path.join(cfg.ddpm_root, "MedDDPM2D_T2_models_v2")
    os.makedirs(models_dir, exist_ok=True)
    model = UNet2DCond(1, cond_ch, cfg.base, cfg.time_dim, cfg.dropout).to(device)
    sched = DiffusionScheduler2D(cfg.num_timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ema = EMA(model, cfg.ema_decay)
    amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, hist = 1, float("inf"), []
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["raw_model"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["optimizer"])
        start, best = ck["epoch"] + 1, ck.get("best_val", best)

    def save(path, epoch, val):
        torch.save({"model": ema.state_dict_for(model), "raw_model": model.state_dict(), "ema": ema.shadow,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "config": {**asdict(cfg), "cond_ch": cond_ch, "pred_type": "eps"},
                    "epoch": epoch, "val_loss": val, "best_val": best}, path)

    for ep in range(start, cfg.epochs + 1):
        t0, tot, n = time.time(), 0.0, 0
        model.train()
        for b in tr:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(device.type, enabled=amp):
                loss = _step(model, sched, b, cfg, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tot, n = tot + float(loss.detach()), n + 1
        val = float("nan")
        if ep % cfg.val_every == 0 or ep == cfg.epochs:
            ema.apply_shadow(model)
            model.eval()
            state = torch.get_rng_state()
            torch.manual_seed(123)          # same t / noise at every validation -> comparable curves
            with torch.no_grad():
                vs = [float(_step(model, sched, {k: (v.to(device) if torch.is_tensor(v) else v)
                                                 for k, v in b.items()}, cfg, device)) for b in va]
            torch.set_rng_state(state)
            ema.restore(model)
            val = sum(vs) / max(len(vs), 1)
            if val < best:
                best = val
                save(os.path.join(models_dir, "best_model_EMA.pt"), ep, val)
        save(os.path.join(models_dir, "last_model_EMA.pt"), ep, val)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": val})
        print(f"epoch {ep:03d} | train {tot / max(n, 1):.4f} | val {val:.4f} | best {best:.4f} | {time.time() - t0:.0f}s")
    with open(os.path.join(models_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return os.path.join(models_dir, "best_model_EMA.pt"), hist
