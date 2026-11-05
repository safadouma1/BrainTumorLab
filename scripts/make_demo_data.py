"""Create a small SYNTHETIC project folder (fake brains + tumors) to test the code and the app
without BraTS. Nothing here is real patient data.

    python scripts/make_demo_data.py /tmp/btl_demo --patients 12 --train
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def fake_patient(folder, pid, rng, shape=(176, 208, 150)):
    import nibabel as nib
    from scipy.ndimage import gaussian_filter

    os.makedirs(folder, exist_ok=True)
    X, Y, Z = shape
    g = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    r_brain = (g[0] / 0.8) ** 2 + (g[1] / 0.85) ** 2 + (g[2] / 0.75) ** 2
    brain = r_brain < 1
    vent = ((g[0] / 0.12) ** 2 + (g[1] / 0.35) ** 2 + ((g[2] - 0.05) / 0.25) ** 2 < 1)
    texture = gaussian_filter(rng.standard_normal(shape), 2.0)
    white = (r_brain < 0.55) & ~vent
    c = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.35)])
    rad = rng.uniform(0.18, 0.3)
    warp = gaussian_filter(rng.standard_normal(shape), 6) * 0.08
    d = np.sqrt(((g[0] - c[0]) / rad) ** 2 + ((g[1] - c[1]) / rad) ** 2 + ((g[2] - c[2]) / rad) ** 2) + warp
    seg = np.zeros(shape, np.int16)
    seg[(d < 1.0) & brain] = 2
    seg[(d < 0.55) & brain] = 4
    seg[(d < 0.38) & brain] = 1
    base = {"flair": (100, 85, 140, 260, 60), "t1": (110, 130, 30, 70, 50),
            "t1ce": (110, 130, 30, 220, 40), "t2": (90, 75, 250, 200, 230)}
    for m, (gm, wm, csf, et_v, ncr_v) in base.items():
        v = np.where(white, wm, gm).astype(np.float32)
        v[vent] = csf
        v[seg == 2] = (gm + 0.6 * (et_v - gm)) if m in ("flair", "t2") else gm * 0.9
        v[seg == 4] = et_v
        v[seg == 1] = ncr_v
        v = (v + 8 * texture + rng.normal(0, 4, shape)) * brain
        nib.save(nib.Nifti1Image(np.clip(v, 0, None).astype(np.float32), np.eye(4)),
                 os.path.join(folder, f"{pid}_{m}.nii.gz"))
    nib.save(nib.Nifti1Image(seg.astype(np.float32), np.eye(4)), os.path.join(folder, f"{pid}_seg.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--train", action="store_true", help="also train tiny models (CPU, minutes)")
    a = ap.parse_args()
    from braintumorlab.preprocessing import run_pipeline

    raw = os.path.join(a.out, "raw")
    rng = np.random.default_rng(0)
    ids = [f"BraTS20_Training_{i:03d}" for i in range(1, a.patients + 1)]
    for pid in ids:
        if not os.path.exists(os.path.join(raw, pid)):
            fake_patient(os.path.join(raw, pid), pid, rng)
    pd.DataFrame({"Grade": ["HGG", "HGG", "LGG"] * (len(ids) // 3 + 1),
                  "BraTS_2020_subject_ID": ids + [""] * 3}).head(len(ids)).to_csv(
        os.path.join(raw, "name_mapping.csv"), index=False)
    base = os.path.join(a.out, "miniProjetAI")
    data = os.path.join(a.out, "preprocessed")
    os.makedirs(base, exist_ok=True)
    if not os.path.isdir(data):
        df = run_pipeline(raw, data)
        df.to_csv(os.path.join(base, "patients_metadata_2020_fini.csv"), index=False)
    if a.train:
        import torch

        from braintumorlab.classification import MultiTaskGliomaClassifier, save_model
        from braintumorlab.ddpm_data import build_npz_dataset
        from braintumorlab.ddpm_train import DDPMConfig, train_ddpm
        from braintumorlab.segmentation import SegConfig, evaluate_model, prepare_cache, train_segmentation

        ddpm_root = os.path.join(base, "MedDDPM_dataset_T2")
        build_npz_dataset(data, ddpm_root, cond_modalities=("flair",))   # cond_ch = 4 like the real dataset
        train_ddpm(DDPMConfig(ddpm_root=ddpm_root, models_dir=os.path.join(ddpm_root, "MedDDPM2D_T2_models"),
                              epochs=2, batch_size=2, base=8, time_dim=32, num_timesteps=50, out_size=64,
                              num_workers=0, val_every=1))
        torch.manual_seed(0)
        save_model(MultiTaskGliomaClassifier(1), os.path.join(base, "demo_classifier.pt"))
        eval_dir = os.path.join(base, "results", "eval")
        for name in ("unet3d", "segresnet"):
            cfg = SegConfig(data_root=data, cache_dir=os.path.join(a.out, "seg_cache"),
                            out_dir=os.path.join(base, "models", "segmentation"), model_name=name,
                            roi=(64, 64, 64), epochs=2, num_workers=0, val_every=1, use_amp=False)
            prepare_cache(cfg)
            best, _ = train_segmentation(cfg)
            evaluate_model(best, cfg, "test", eval_dir)
        with open(os.path.join(eval_dir, "README.txt"), "w") as f:
            f.write("SYNTHETIC demo results (models trained 2 epochs on fake data). Not real results.\n")
    print(json.dumps({"BTL_BASE": base, "BTL_DATA_ROOT": data}, indent=1))


if __name__ == "__main__":
    main()
