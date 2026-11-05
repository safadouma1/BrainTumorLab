"""MRI generation with Med-DDPM (T2) and simulated scenarios.

Modes
-----
synthesis    T2 generated from pure noise, conditioned on the other channel(s) + segmentation
             -> the honest "missing modality" test (compare with the real T2: PSNR / SSIM)
img2img      real T2 slightly noised then denoised (strength controls how much is regenerated)
postop       G2: tumor resected (eroded), cavity, ROI regenerated
progression  G3: tumor grown (dilated), edema ring, ROI regenerated
response     G4: enhancing tumor and edema reduced (treatment response), ROI regenerated

Scenario images are simulations for teaching; there is no real follow-up MRI in BraTS to
validate them.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .ddpm_data import SliceDataset, load_index
from .diffusion import ddim_sample, find_checkpoint, gaussian_blur, load_ddpm_checkpoint, tumor_mask_from_cond

MODES = ("synthesis", "img2img", "postop", "progression", "response")


def _odd(k):
    return int(k) | 1


def dilate(m, k):
    return F.max_pool2d(m.float(), _odd(k), 1, _odd(k) // 2) if k > 1 else (m > 0.5).float()


def erode(m, k):
    return (1 - dilate(1 - (m > 0.5).float(), k) > 0.5).float() if k > 1 else (m > 0.5).float()


# ------------------------------------------------------------------
# Scenario edits (segmentation + initial T2)
# ------------------------------------------------------------------
def scenario_postop(cond, x, bm, residual=0.25, cavity=0.7):
    before = tumor_mask_from_cond(cond) * bm
    after, k = erode(before, 11) * bm, 11
    while residual > 0 and after.sum() > residual * before.sum() and k < 61:
        k += 4
        after = erode(before, k) * bm
    if residual <= 0:
        after = torch.zeros_like(before)
    c = cond.clone()
    c[:, -3:] = (cond[:, -3:] > 0.5).float() * after
    removed = ((before > 0.5) & (after < 0.5)).float() * bm
    if removed.sum() >= 1:
        ring = ((dilate(removed, 19) - dilate(removed, 9)) > 0.5).float() * bm
        vals = x[ring > 0.5] if ring.sum() >= 10 else x[(bm * (1 - removed)) > 0.5]
        soft = gaussian_blur(removed, 7, 1.5).clamp(0, 1) * bm
        x = x * (1 - soft) + (vals.mean() - cavity * vals.std()) * soft
    return c, x * bm, before, after


def scenario_progression(cond, x, bm, grow=11, gain_old=0.2, gain_new=0.45):
    seg = (cond[:, -3:] > 0.5).float()
    ncr, ed, et = seg[:, 0:1], seg[:, 1:2], seg[:, 2:3]
    before = tumor_mask_from_cond(cond) * bm
    after = torch.maximum(dilate(before, grow) * bm, before)
    et_n = torch.maximum(et, dilate(et, 2 * max(1, grow // 4) + 1) * after) if et.sum() > 0 else et
    ncr_n = ncr * (1 - et_n)
    ed_n = torch.maximum(ed, after - before).clamp(0, 1) * (1 - et_n) * (1 - ncr_n)
    c = cond.clone()
    c[:, -3:] = torch.cat([ncr_n, ed_n, et_n], 1) * bm
    hi = torch.quantile(x[bm > 0.5].float(), 0.99)
    old = gaussian_blur(before, 7, 1.5) * bm
    new = gaussian_blur(((after > 0.5) & (before < 0.5)).float(), 7, 1.5) * bm
    x = x + (hi - x).clamp_min(0) * (gain_old * old + gain_new * new)
    return c, x * bm, before, tumor_mask_from_cond(c) * bm


def scenario_response(cond, x, bm, shrink_et=9, shrink_ed=7, gain_et=-0.35, gain_ed=-0.2):
    seg = (cond[:, -3:] > 0.5).float()
    ncr, ed, et = seg[:, 0:1], seg[:, 1:2], seg[:, 2:3]
    before = tumor_mask_from_cond(cond) * bm
    et_c = erode(et, shrink_et) * bm
    ed_c = erode(ed, shrink_ed) * bm * (1 - et_c)
    c = cond.clone()
    c[:, -3:] = torch.cat([ncr * bm * (1 - et_c), ed_c, et_c], 1)
    med = torch.median(x[bm > 0.5])
    z_et = gaussian_blur(et, 5, 1.0) * bm
    z_ed = gaussian_blur(ed * (1 - et), 5, 1.0) * bm
    x = x + (x - med) * (gain_et * z_et + gain_ed * z_ed)
    return c, x.clamp(-1.5, 1.5) * bm, before, tumor_mask_from_cond(c) * bm


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------
def _np(t):
    return t.detach().float().cpu().numpy()


def to_unit(img, ref, mask):
    vals = ref[mask > 0.5] if (mask > 0.5).sum() > 10 else ref.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * (mask > 0.5)).astype(np.float32)


class MedDDPMGenerator:
    def __init__(self, ddpm_root, checkpoint=None, index_csv=None, device=None, slice_axis="X", out_size=192,
                 fix_background="auto"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.index = load_index(ddpm_root, index_csv)
        ck = find_checkpoint(checkpoint, os.path.join(str(ddpm_root), "MedDDPM2D_T2_models"), str(ddpm_root))
        if ck is None:
            raise FileNotFoundError("No Med-DDPM checkpoint (best_model_EMA.pt) found.")
        self.model, self.scheduler, self.info = load_ddpm_checkpoint(ck, self.device)
        if fix_background == "auto":   # a legacy checkpoint was trained without background correction
            fix_background = not self.info["legacy"]
        self.dataset = SliceDataset(self.index, "test" if (self.index["split_preproc"] == "test").any() else "val",
                                    slice_mode="tumor", slice_axis=slice_axis, out_size=out_size,
                                    fix_background=fix_background)
        self.all = {s: SliceDataset(self.index, s, slice_mode="tumor", slice_axis=slice_axis, out_size=out_size,
                                    fix_background=fix_background)
                    for s in ("train", "val", "test") if (self.index["split_preproc"] == s).any()}
        self.warning = None
        if self.info["cond_ch"] >= 4 + 3:
            self.warning = ("The condition has 4 image channels: if T2 is one of them the model can copy it. "
                            "Check how the npz files were built.")

    def patients(self):
        return [{"patient_id": p, "split": s} for s, ds in self.all.items() for p in ds.patient_ids()]

    def _ds(self, pid):
        for ds in self.all.values():
            if str(pid) in ds.pid_to_idx:
                return ds
        raise KeyError(f"{pid} is not in the Med-DDPM index")

    def num_slices(self, pid):
        ds = self._ds(pid)
        shp = ds.volume_shape(pid)
        return {"X": shp[3], "Y": shp[2], "Z": shp[1]}[ds.axis]

    @torch.no_grad()
    def run(self, pid, mode="synthesis", slice_index=None, strength=0.3, steps=50, seed=0, n_samples=1,
            residual=0.25, grow=11):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        b = self._ds(pid).get_by_pid(pid, slice_index)
        dev = self.device
        cond, real, bm = b["cond"].to(dev), b["target"].to(dev), b["brain_mask"].to(dev)
        before = after = None
        ref, c = real, cond
        if mode == "postop":
            c, ref, before, after = scenario_postop(cond, real, bm, residual)
        elif mode == "progression":
            c, ref, before, after = scenario_progression(cond, real, bm, grow)
        elif mode == "response":
            c, ref, before, after = scenario_response(cond, real, bm)
        kw = dict(steps=steps, brain_mask=bm, pred_type=self.info["pred_type"])
        samples = []
        for i in range(max(1, n_samples)):
            if mode == "synthesis":
                x, _ = ddim_sample(self.model, self.scheduler, c, None, 1.0, seed=seed + i, **kw)
            elif mode == "img2img":
                x, _ = ddim_sample(self.model, self.scheduler, c, ref, strength, seed=seed + i, **kw)
            else:
                x, _ = ddim_sample(self.model, self.scheduler, c, ref, max(strength, 0.05), roi_only=True,
                                   tumor_mask=torch.maximum(before, after), roi_id_mix=0.3, seed=seed + i, **kw)
            samples.append(_np(x[0, 0]))
        gen = np.mean(samples, 0)
        r, m = _np(real[0, 0]), _np(bm[0, 0])
        out = {"patient_id": str(pid), "slice_index": b["slice_index"][0], "mode": mode,
               "real": to_unit(r, r, m), "generated": to_unit(gen, r, m),
               "condition": to_unit(_np(cond[0, 0]), _np(cond[0, 0]), m),
               "seg": _np(cond[0, -3:]), "seg_after": _np(c[0, -3:]), "metrics": {}}
        out["error"] = np.abs(out["generated"] - out["real"])
        if n_samples > 1:
            out["uncertainty"] = np.std([to_unit(s, r, m) for s in samples], 0)
        from .analysis import generation_metrics

        if mode in ("synthesis", "img2img"):
            out["metrics"].update(generation_metrics(r, gen, m > 0.5))
            out["metrics"]["MAE"] = float(np.abs(out["generated"] - out["real"])[m > 0.5].mean())
        else:
            a0, a1 = float(before.sum()), float(after.sum())
            out["metrics"].update({"tumor_area_before_px": a0, "tumor_area_after_px": a1,
                                   "area_change_pct": 100 * (a1 - a0) / a0 if a0 else 0.0})
        return out

    def evaluate(self, split="test", steps=50, max_patients=None, seed=0):
        """Full-synthesis PSNR / SSIM on the tumor slice of every patient of a split."""
        ds = self.all[split]
        rows = []
        for pid in ds.patient_ids()[:max_patients]:
            r = self.run(pid, "synthesis", steps=steps, seed=seed)
            rows.append({"patient_id": pid, **r["metrics"]})
        return rows
