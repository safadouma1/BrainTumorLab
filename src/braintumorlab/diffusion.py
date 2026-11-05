"""Med-DDPM 2D: cosine scheduler, conditional UNet, EMA, DDIM sampling (full synthesis,
img2img and ROI inpainting) and checkpoint loading."""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Scheduler + time embedding
# ------------------------------------------------------------------
def cosine_alpha_bar(t, s=0.008):
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


class DiffusionScheduler2D(nn.Module):
    def __init__(self, num_timesteps=300, clip_beta_max=0.999):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        ab = cosine_alpha_bar(torch.linspace(0, 1, self.num_timesteps + 1))
        ab = ab / ab[0]
        betas = torch.clamp(1 - ab[1:] / ab[:-1], 1e-8, clip_beta_max)
        acp = torch.cumprod(1 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_alphas_cumprod", acp.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - acp).sqrt())

    def q_sample(self, x0, t, noise):
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0
                + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)

    def predict_x0(self, x_t, t, eps):
        return ((x_t - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * eps)
                / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1))


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
    return torch.cat([emb, torch.zeros_like(emb[:, :1])], -1) if dim % 2 else emb


# ------------------------------------------------------------------
# Conditional UNet (bilinear upsampling, no ConvTranspose -> fewer artefacts)
# ------------------------------------------------------------------
class ResBlockGN(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)) + self.time_proj(t_emb)[:, :, None, None])
        h = self.act(self.norm2(self.conv2(self.drop(h))))
        return h + self.skip(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res = ResBlockGN(out_ch + skip_ch, out_ch, time_dim, dropout=dropout)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([self.conv_in(x), skip], 1), t_emb)


class UNet2DCond(nn.Module):
    def __init__(self, in_ch=1, cond_ch=4, base=48, time_dim=128, dropout=0.0):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.enc1 = ResBlockGN(in_ch + cond_ch, base, time_dim, dropout=dropout)
        self.enc2 = ResBlockGN(base, base * 2, time_dim, dropout=dropout)
        self.enc3 = ResBlockGN(base * 2, base * 4, time_dim, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.mid1 = ResBlockGN(base * 4, base * 4, time_dim, dropout=dropout)
        self.mid2 = ResBlockGN(base * 4, base * 4, time_dim, dropout=dropout)
        self.up3 = UpBlock(base * 4, base * 2, base * 2, time_dim, dropout=dropout)
        self.up2 = UpBlock(base * 2, base, base, time_dim, dropout=dropout)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x_noisy, cond, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h1 = self.enc1(torch.cat([x_noisy, cond], 1), t_emb)
        h2 = self.enc2(self.pool(h1), t_emb)
        h3 = self.enc3(self.pool(h2), t_emb)
        m = self.mid2(self.mid1(h3, t_emb), t_emb)
        return self.out(self.up2(self.up3(m, h2, t_emb), h1, t_emb))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

    def state_dict_for(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd.update({k: v.clone() for k, v in self.shadow.items()})
        return sd


# ------------------------------------------------------------------
# ROI masks
# ------------------------------------------------------------------
def gaussian_blur(x, k=31, sigma=7.0):
    if k <= 1 or sigma <= 0:
        return x
    k = int(k) | 1
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    return F.conv2d(F.pad(x, [k // 2] * 4, mode="replicate"), (g[:, None] * g[None, :]).view(1, 1, k, k))


def tumor_mask_from_cond(cond, thr=0.5):
    return (cond[:, -3:] > thr).any(1, keepdim=True).float()


def make_soft_roi(tumor, brain=None, ring_k=17, soft_k=31, sigma=7.0):
    m = (tumor > 0.5).float()
    if ring_k > 1:
        k = int(ring_k) | 1
        m = F.max_pool2d(m, k, 1, k // 2)
    soft = torch.maximum(gaussian_blur(m, soft_k, sigma), (tumor > 0.5).float()).clamp(0, 1)
    return soft * (brain > 0.5).float() if brain is not None else soft


def apply_roi_identity_mix(x_gen, x0, roi, alpha=0.6):
    """Inside the ROI: alpha * real + (1 - alpha) * generated."""
    if alpha <= 0:
        return x_gen
    r = roi.clamp(0, 1)
    return r * (alpha * x0 + (1 - alpha) * x_gen) + (1 - r) * x_gen


# ------------------------------------------------------------------
# DDIM sampling
# ------------------------------------------------------------------
@torch.no_grad()
def ddim_sample(model, scheduler, cond, x_ref=None, strength=1.0, steps=50, eta=0.0, brain_mask=None,
                roi_only=False, tumor_mask=None, ring_k=17, roi_soft_k=31, roi_soft_sigma=7.0,
                dyn_thresh=True, clamp_x0=1.2, roi_id_mix=0.0, pred_type="eps", seed=None):
    """DDIM sampler.

    * strength = 1.0 and x_ref = None : full synthesis from pure noise (missing-modality generation).
    * strength < 1.0 : img2img, x_ref is noised to t = strength * T then denoised.
    * roi_only = True : only the (soft) tumor ROI is regenerated; the reference is re-injected
      outside the ROI at every step (RePaint-style inpainting).
    Returns (x_generated [B,1,H,W], roi [B,1,H,W]).
    """
    dev = cond.device
    B, _, H, W = cond.shape
    T = scheduler.num_timesteps
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if seed is not None else None

    def rn(shape):
        return torch.randn(shape, generator=gen, device=dev)

    bm = torch.ones(B, 1, H, W, device=dev) if brain_mask is None else (brain_mask > 0.5).float()
    if x_ref is None:
        strength, roi_only = 1.0, False
    roi = (make_soft_roi(tumor_mask if tumor_mask is not None else tumor_mask_from_cond(cond), bm,
                         ring_k, roi_soft_k, roi_soft_sigma) if roi_only else bm.clone())
    t0 = max(1, min(T, int(round(float(strength) * T))))
    ts = torch.linspace(t0 - 1, 0, max(1, min(int(steps), t0)), device=dev).round().long().unique_consecutive().tolist()
    acp = scheduler.alphas_cumprod
    tt = lambda t: torch.full((B,), int(t), device=dev, dtype=torch.long)  # noqa: E731
    x = rn((B, 1, H, W)) if x_ref is None or t0 >= T else scheduler.q_sample(x_ref, tt(ts[0]), rn((B, 1, H, W)))
    was_training = model.training
    model.eval()
    for i, t in enumerate(ts):
        tp = ts[i + 1] if i + 1 < len(ts) else -1
        out = model(x, cond, tt(t))
        x0 = out if pred_type == "x0" else scheduler.predict_x0(x, tt(t), out)
        if dyn_thresh:
            s = torch.quantile(x0.abs().reshape(B, -1).float(), 0.995, 1).clamp_min(clamp_x0).view(B, 1, 1, 1)
            x0 = torch.clamp(x0, -s, s) * (clamp_x0 / s)
        else:
            x0 = x0.clamp(-clamp_x0, clamp_x0)
        x0 = x0 * bm
        a_t = acp[t]
        a_p = acp[tp] if tp >= 0 else torch.ones_like(a_t)
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-8)
        sig = eta * ((1 - a_p) / (1 - a_t) * (1 - a_t / a_p)).clamp_min(0).sqrt()
        x = a_p.sqrt() * x0 + (1 - a_p - sig ** 2).clamp_min(0).sqrt() * eps
        if eta > 0 and tp >= 0:
            x = x + sig * rn(x.shape)
        if roi_only:
            known = scheduler.q_sample(x_ref, tt(tp), rn(x.shape)) if tp >= 0 else x_ref
            x = roi * x + (1 - roi) * known
    if was_training:
        model.train()
    x = x * bm
    if roi_only and roi_id_mix > 0:
        x = apply_roi_identity_mix(x, x_ref, roi, roi_id_mix)
    return x, roi


# ------------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------------
def load_ddpm_checkpoint(path, device="cpu"):
    """Rebuild UNet2DCond + scheduler from the weights themselves (no need to know BASE,
    cond_ch or T in advance). Also works with legacy checkpoints."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    w = sd["enc1.conv1.weight"]
    in_ch = sd["out.weight"].shape[0]
    info = {"path": str(path), "base": int(w.shape[0]), "cond_ch": int(w.shape[1] - in_ch),
            "time_dim": int(sd["time_mlp.0.weight"].shape[1]), "epoch": ck.get("epoch"),
            "val_loss": ck.get("val_loss"), "legacy": "config" not in ck}
    cfg = ck.get("config", {}) or {}
    sched = ck.get("scheduler")
    info["num_timesteps"] = int(cfg.get("num_timesteps") or (len(sched["betas"]) if isinstance(sched, dict)
                                                             and "betas" in sched else 300))
    info["pred_type"] = cfg.get("pred_type", "eps")
    model = UNet2DCond(in_ch, info["cond_ch"], info["base"], info["time_dim"])
    model.load_state_dict(sd)
    return model.to(device).eval(), DiffusionScheduler2D(info["num_timesteps"]).to(device), info


def find_checkpoint(*candidates):
    names = ("best_model_EMA.pt", "best_model_ema.pt", "last_model_EMA.pt")
    for c in candidates:
        if c and os.path.isfile(str(c)):
            return str(c)
    for c in candidates:
        if c and os.path.isdir(str(c)):
            for cur, _, files in os.walk(str(c)):
                for n in names:
                    if n in files:
                        return os.path.join(cur, n)
    return None
