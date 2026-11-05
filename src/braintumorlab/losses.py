"""Constraint pack (23 losses + direct tumor L1) applied to x0_hat during Med-DDPM training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _4d(x):
    return x.unsqueeze(1) if x.ndim == 3 else x


def _m(mask, x):
    mask = _4d(mask).float()
    return mask.expand(-1, x.shape[1], -1, -1) if mask.shape[1] != x.shape[1] else mask


def masked_l1(p, t, mask, eps=1e-8):
    p, t = _4d(p), _4d(t)
    m = _m(mask, p)
    return ((p - t).abs() * m).sum() / (m.sum() + eps)


def masked_mse(p, t, mask, eps=1e-8):
    p, t = _4d(p), _4d(t)
    m = _m(mask, p)
    return (((p - t) ** 2) * m).sum() / (m.sum() + eps)


def charbonnier(p, t, mask, eps=1e-3):
    p, t = _4d(p), _4d(t)
    m = _m(mask, p)
    return (torch.sqrt((p - t) ** 2 + eps ** 2) * m).sum() / (m.sum() + 1e-8)


def sobel_grad(x):
    x = _4d(x)
    sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, sx, padding=1), F.conv2d(x, sx.transpose(2, 3), padding=1)


def sobel_mag(x):
    gx, gy = sobel_grad(x)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def laplace(x):
    x = _4d(x)
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1)


def dilate(m, k=7):
    return F.max_pool2d(_4d(m), k, 1, k // 2)


def erode(m, k=7):
    return 1.0 - dilate(1.0 - _4d(m), k)


def ring(m, k=9):
    m = _4d(m).float()
    return (dilate(m, k) - erode(m, k)).clamp(0, 1)


def _multiscale(fn, x, y, m, scales):
    tot = 0.0
    for s in scales:
        tot = tot + (fn(x, y, m) if s == 1 else
                     fn(F.avg_pool2d(x, s, s), F.avg_pool2d(y, s, s), F.max_pool2d(m, s, s)))
    return tot / len(scales)


def _zero(x):
    return x.new_tensor(0.0)


def c07_tv(x, bm):
    x = _4d(x)
    m = _m(bm, x)
    dy = (x[:, :, 1:] - x[:, :, :-1]).abs() * m[:, :, 1:] * m[:, :, :-1]
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs() * m[:, :, :, 1:] * m[:, :, :, :-1]
    return dy.mean() + dx.mean()


def c10_calibration(x, y, bm, eps=1e-8):
    x, y = _4d(x), _4d(y)
    m = _m(bm, x)
    w = m.sum() + eps
    mx, my = (x * m).sum() / w, (y * m).sum() / w
    sx = torch.sqrt(((x - mx) ** 2 * m).sum() / w + eps)
    sy = torch.sqrt(((y - my) ** 2 * m).sum() / w + eps)
    return (mx - my).abs() + (sx - sy).abs()


def c18_local_contrast(x, y, bm, k=9, eps=1e-6):
    x, y = _4d(x), _4d(y)
    w = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype) / k ** 2

    def std(z):
        mz = F.conv2d(z, w, padding=k // 2)
        return torch.sqrt(F.conv2d((z - mz) ** 2, w, padding=k // 2) + eps)

    return masked_l1(std(x), std(y), _m(bm, x))


def c19_symmetry(x, bm, tm=None, k=21):
    x = _4d(x)
    healthy = _m(bm, x)
    if tm is not None:
        healthy = (healthy * (1 - dilate(_m(tm, x), k))).clamp(0, 1)
    return masked_l1(x, torch.flip(x, [3]), healthy)


def c23_tumor_center(x, y, tm, eps=1e-6):
    if tm is None:
        return _zero(x)
    x, y = _4d(x), _4d(y)
    m = _m(tm, x)
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype) / H,
                            torch.arange(W, device=x.device, dtype=x.dtype) / W, indexing="ij")

    def com(w):
        s = w.sum((2, 3), keepdim=True) + eps
        return (w * yy).sum((2, 3), keepdim=True) / s, (w * xx).sum((2, 3), keepdim=True) / s

    (y1, x1), (y2, x2) = com((x.abs() * m).sum(1, keepdim=True)), com((y.abs() * m).sum(1, keepdim=True))
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2).mean()


def constraint_pack(x0_hat, x0, batch, weights: dict, pred=None, tumor_weight=5.0):
    """Weighted sum of the constraints listed in `weights` (keys c01..c23 and tumor_l1)."""
    x, y = _4d(x0_hat), _4d(x0)
    bm = batch["brain_mask"]
    em = batch.get("edge_mask", bm)
    tm = batch.get("tumor_mask")

    def tumor(fn):
        return fn() if tm is not None else _zero(x)

    def fft(kind):
        m = _m(bm, x)
        fx, fy = torch.fft.rfft2((x * m).float(), norm="ortho"), torch.fft.rfft2((y * m).float(), norm="ortho")
        if kind == "amp":
            return (fx.abs() - fy.abs()).abs().mean()
        d = torch.angle(fx) - torch.angle(fy)
        return torch.atan2(torch.sin(d), torch.cos(d)).abs().mean()

    def grad_dir():
        gx1, gy1 = sobel_grad(x)
        gx2, gy2 = sobel_grad(y)
        cos = (gx1 * gx2 + gy1 * gy2) / (torch.sqrt(gx1 ** 2 + gy1 ** 2 + 1e-8) * torch.sqrt(gx2 ** 2 + gy2 ** 2 + 1e-8))
        return masked_l1(1 - cos, torch.zeros_like(cos), bm)

    table = {
        "c01": lambda: masked_l1(x, y, bm),                       # brain L1
        "c02": lambda: masked_mse(x, y, bm),                      # brain MSE
        "c03": lambda: charbonnier(x, y, bm),                     # robust L1
        "c04": lambda: masked_l1(sobel_mag(x), sobel_mag(y), em),  # brain edges
        "c05": lambda: masked_l1(laplace(x), laplace(y), bm),     # fine details
        "c06": grad_dir,                                          # gradient direction
        "c07": lambda: c07_tv(x, bm),                             # total variation
        "c08": lambda: (x.abs() * (1 - _m(bm, x))).mean(),        # black background
        "c09": lambda: (F.relu(x - 1) + F.relu(-1 - x)).mean(),   # value range
        "c10": lambda: c10_calibration(x, y, bm),                 # mean / std calibration
        "c11": lambda: _multiscale(masked_l1, x, y, _m(bm, x), (1, 2, 4)),
        "c12": lambda: _multiscale(lambda a, b, m: masked_l1(sobel_mag(a), sobel_mag(b), m), x, y, _m(bm, x), (1, 2, 4)),
        "c13": lambda: fft("amp"),
        "c14": lambda: fft("phase"),
        "c15": lambda: tumor(lambda: ((x - y).abs() * (_m(bm, x) + (tumor_weight - 1) * _m(tm, x))).sum()
                             / ((_m(bm, x) + (tumor_weight - 1) * _m(tm, x)).sum() + 1e-8)),
        "c16": lambda: tumor(lambda: masked_l1(x, y, ring(tm, 9))),   # tumor border
        "c17": lambda: masked_l1(x, y, ring(bm, 11)),                 # brain border
        "c18": lambda: c18_local_contrast(x, y, bm),
        "c19": lambda: c19_symmetry(x, bm, tm),                       # left/right symmetry of healthy tissue
        "c20": lambda: c07_tv(pred, bm) if pred is not None else _zero(x),
        "c21": lambda: tumor(lambda: _multiscale(masked_l1, x, y, _m(tm, x), (1, 2, 4))),
        "c22": lambda: tumor(lambda: _multiscale(lambda a, b, m: masked_l1(sobel_mag(a), sobel_mag(b), m),
                                                 x, y, _m(tm, x), (1, 2))),
        "c23": lambda: c23_tumor_center(x, y, tm),
        "tumor_l1": lambda: tumor(lambda: masked_l1(x, y, tm)),
    }
    total, terms = _zero(x), {}
    for k, w in weights.items():
        if w and k in table:
            v = table[k]()
            terms[k] = float(v.detach())
            total = total + w * v
    return total, terms
