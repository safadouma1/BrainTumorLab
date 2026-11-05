"""Surface meshes (marching cubes) for the 3D viewer."""
from __future__ import annotations

import base64

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .volumes import label_masks

COLORS = {"brain": "#9aa7c7", "ED": "#22d3a6", "NCR": "#f43f5e", "ET": "#fbbf24"}


def mask_to_mesh(mask, spacing=(1.0, 1.0, 1.0), step=2, smooth=1.0, origin=None):
    from skimage.measure import marching_cubes

    m = np.asarray(mask, bool)
    if m.sum() < 30:
        return None
    f = gaussian_filter(m.astype(np.float32), smooth) if smooth else m.astype(np.float32)
    if step > 1:
        f = zoom(f, 1 / step, order=1)
    f = np.pad(f, 1)
    verts, faces, normals, _ = marching_cubes(f, 0.5, spacing=tuple(s * step for s in spacing))
    verts -= np.asarray(spacing) * step          # undo padding
    if origin is not None:
        verts -= origin
    return {"vertices": base64.b64encode(verts.astype(np.float32).tobytes()).decode(),
            "normals": base64.b64encode(normals.astype(np.float32).tobytes()).decode(),
            "faces": base64.b64encode(faces.astype(np.uint32).tobytes()).decode(),
            "n_vertices": int(len(verts)), "n_faces": int(len(faces))}


def patient_meshes(brain, labels, spacing=(1.0, 1.0, 1.0), step=2):
    """Brain surface + NCR / ED / ET surfaces, centred on the brain."""
    sp = np.asarray(spacing, float)
    origin = np.argwhere(brain).mean(0) * sp
    out = {"brain": mask_to_mesh(brain, spacing, step=max(step, 2), smooth=2.0, origin=origin)}
    if labels is not None:
        for k, m in label_masks(labels).items():
            out[k] = mask_to_mesh(m, spacing, step=step, smooth=0.8, origin=origin)
    return {k: ({**v, "color": COLORS[k]} if v else None) for k, v in out.items()}
