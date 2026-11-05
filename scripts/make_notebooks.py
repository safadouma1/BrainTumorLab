"""Generate the Google Colab notebooks in notebooks/  (python scripts/make_notebooks.py)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP = '''
# --- Setup: mount Drive + install the BrainTumorLab code (run this cell first) ---
from google.colab import drive
drive.mount("/content/drive")

import os, sys
REPO_URL = ""   # e.g. "https://github.com/<your-account>/BrainTumorLab.git" ; empty = use the zip on Drive
ZIP_ON_DRIVE = "/content/drive/MyDrive/miniProjetAI/BrainTumorLab.zip"
if not os.path.isfile("/content/BrainTumorLab/pyproject.toml"):
    import shutil
    shutil.rmtree("/content/BrainTumorLab", ignore_errors=True)
    if REPO_URL:
        !git clone -q {REPO_URL} /content/BrainTumorLab
    else:
        !rm -rf /content/btl_tmp && unzip -q -o "{ZIP_ON_DRIVE}" -d /content/btl_tmp
        _src = "/content/btl_tmp/BrainTumorLab" if os.path.isfile("/content/btl_tmp/BrainTumorLab/pyproject.toml") else "/content/btl_tmp"
        shutil.move(_src, "/content/BrainTumorLab")
!pip install -q -e /content/BrainTumorLab
sys.path.insert(0, "/content/BrainTumorLab/src")

import torch
from braintumorlab.config import Paths, seed_everything
seed_everything(0)
paths = Paths("/content/drive/MyDrive/miniProjetAI")
DATA_ROOT = paths.ensure_data_extracted()     # unzips BraTS_2020_Data_preprocessing_nii_FIXED.zip once
print("data:", DATA_ROOT, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
'''


def nb(name, cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n")}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    doc = {"cells": out, "nbformat": 4, "nbformat_minor": 0,
           "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
    with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("written:", name)


M, C = "markdown", "code"

# ---------------------------------------------------------------- 00
nb("00_preprocessing.ipynb", [
    (M, """# 00 — Preprocessing BraTS 2020

Part A computes the tumor volume and localisation tables from the segmentations (about 5 minutes).
Part B (optional) rebuilds the whole preprocessed dataset from the raw BraTS archive (about 1-2 hours on Colab).
The preprocessed dataset `BraTS_2020_Data_preprocessing_nii_FIXED.zip` is used by every other notebook and by the app."""),
    (C, SETUP),
    (M, """## A. Tumor volume and localisation tables

Volumes are computed from the label map (1 = NCR, 2 = ED, 4 = ET; 1 voxel = 1 mm³). Localisation uses the
NIfTI orientation (affine) and the distance to the brain surface. Output: `recap_stats_volumes_2020_v2.csv`
and `tumor_localization_2020_v2.csv`."""),
    (C, """
from braintumorlab.analysis import volumes_table, localization_table
vol = volumes_table(DATA_ROOT)
vol.to_csv(paths.base / "recap_stats_volumes_2020_v2.csv", index=False)
vol[["vol_WT_ml", "vol_TC_ml", "vol_ET_ml", "vol_ED_ml", "vol_NCR_ml"]].describe().round(1)
"""),
    (C, """
loc = localization_table(DATA_ROOT)
loc.to_csv(paths.base / "tumor_localization_2020_v2.csv", index=False)
loc[["hemisphere", "anterior_posterior", "depth_category"]].value_counts().head(10)
"""),
    (M, """## B. (Optional) rebuild the preprocessed dataset

The z-score is computed **inside the brain** (background = 0), and a `brain_mask.nii.gz`
is saved for each patient. Needs `archive.zip` (Kaggle BraTS 2020) in `MyDrive/miniProjetAI/`."""),
    (C, """
from braintumorlab.preprocessing import ensure_brats2020_extracted, run_pipeline
RAW = ensure_brats2020_extracted(str(paths.base / "archive.zip"), "/content")
OUT = "/content/BraTS_2020_Data_preprocessing_nii_V5"
df_meta = run_pipeline(RAW, OUT, max_patients=None, overwrite=True)   # max_patients=5 for a quick test
df_meta.head()
"""),
    (C, """
import shutil
shutil.make_archive("/content/BraTS_2020_Data_preprocessing_nii_V5", "zip", OUT)
shutil.copy("/content/BraTS_2020_Data_preprocessing_nii_V5.zip", paths.base)
shutil.copy(f"{OUT}/patients_metadata_2020.csv", paths.base / "patients_metadata_2020_v5.csv")
print("To use V5 everywhere: set BTL_DATA_ZIP / BTL_DATA_ROOT, or rename the zip.")
"""),
])

# ---------------------------------------------------------------- 01
nb("01_data_analysis.ipynb", [
    (M, """# 01 — Data analysis and MRI modality comparison

Statistics per modality, histograms, threshold baseline + ablation over the 15 modality combinations,
tumor localisation and patient clustering."""),
    (C, SETUP),
    (C, """
from braintumorlab import analysis as A
from braintumorlab.volumes import list_patients, load_patient
SPLIT = "train"
pid, split, folder = list_patients(DATA_ROOT, SPLIT)[0]
p = load_patient(folder)          # background set to 0, z-score inside the brain
print(pid, "| brain voxels:", p["brain"].sum(), "| tumor voxels:", (p["seg"] > 0).sum())
"""),
    (C, "A.plot_modalities_with_seg(p)     # labels: 1 = NCR/NET (red), 2 = ED (green), 4 = ET (yellow)"),
    (C, "A.plot_histograms(p)"),
    (M, """### Statistics per modality
CNR = (mean tumor - mean healthy) / std healthy. The plain SNR (mean / std) is meaningless on z-scored data
(the mean is about 0), so the CNR is the number to compare between modalities."""),
    (C, "A.modality_table(p).round(3)"),
    (M, """### Threshold baseline + ablation (15 combinations)
Each modality is oriented so that the tumor is "higher" (the tumor is hypo-intense in T1). This is a
simple **baseline**, not an AI model: it measures how much tumor information each sequence carries."""),
    (C, """
abl = A.ablation_study(p, percentile=90)
abl.round(3)
"""),
    (C, """
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, m in zip(axes, ["f1", "recall", "auc"]):
    ax.barh(abl["scenario"], abl[m]); ax.set_xlim(0, 1); ax.set_title(m); ax.invert_yaxis()
plt.tight_layout(); plt.show()
print("Best combination:", abl.iloc[0]["scenario"], "| weakest:", abl.iloc[-1]["scenario"])
"""),
    (M, "### Ablation over several patients (more reliable than one patient)"),
    (C, """
import pandas as pd
rows = []
for pid_, _, f in list_patients(DATA_ROOT, SPLIT, max_patients=20):
    r = A.ablation_study(load_patient(f)); r["patient_id"] = pid_; rows.append(r)
abl_all = pd.concat(rows).groupby("scenario")[["f1", "recall", "auc"]].agg(["mean", "std"]).sort_values(("f1", "mean"), ascending=False)
abl_all.round(3)
"""),
    (M, "### Tumor localisation (uses the NIfTI orientation)"),
    (C, "A.tumor_localization(p['seg'], p['brain'], p['affine'])"),
    (M, """### Patient clustering
K-Means runs on the standardised features; UMAP (or PCA) is only used to draw the 2D map."""),
    (C, """
!pip install -q umap-learn
FEATURE_CASE = "all_modalities"      # see A.FEATURE_CASES
feats = A.features_table(DATA_ROOT, "train", max_patients=60, case=FEATURE_CASE)
clusters, emb = A.cluster_patients(feats, n_clusters=3)
plt.figure(figsize=(7, 6)); plt.scatter(emb[:, 0], emb[:, 1], c=clusters["cluster"], cmap="tab10")
plt.title(f"Patients - {FEATURE_CASE}"); plt.show()
clusters.groupby("cluster").mean(numeric_only=True).round(3)
"""),
    (M, """### Generation metrics
SSIM / PSNR / MAE are computed on images generated by Med-DDPM in notebook 04
(helper: `A.generation_metrics(real, generated, brain_mask)`)."""),
])

# ---------------------------------------------------------------- 02
nb("02_classification_3d.ipynb", [
    (M, """# 02 — 3D CNN classification: grade LGG / HGG + radiological pseudo-profile

Pseudo-profiles (LGG-DA / LGG-OLIGO / HGG-ANA / HGG-GBM) are **K-Means clusters of tumor volumes**, not
histological subtypes. The trained model is saved as `models/classification/cnn3d_grade_profile.pt`."""),
    (C, SETUP),
    (C, """
from braintumorlab import classification as C
from braintumorlab.analysis import volumes_table
import pandas as pd, os
vol_csv = paths.base / "recap_stats_volumes_2020_v2.csv"       # written by notebook 00, part A
volumes = pd.read_csv(vol_csv) if vol_csv.exists() else volumes_table(DATA_ROOT)
df = C.build_patient_table(paths.metadata_csv, volumes)
df = C.add_pseudo_profiles(df)            # K-Means fitted on the train split only
train_df, val_df, test_df = C.split_tables(df)
print(len(train_df), len(val_df), len(test_df))
df.groupby(["grade_str", "pseudo_profile"]).size()
"""),
    (C, """
MODALITIES = ["FLAIR", "T1ce", "T2"]
DOWNSAMPLE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_index = C.build_file_index(DATA_ROOT, MODALITIES)
datasets = [C.BraTS3DClassificationDataset(d, file_index, MODALITIES, DOWNSAMPLE) for d in (train_df, val_df, test_df)]
BALANCE = "sampler"                       # "sampler" OR "weights", never both
train_loader, val_loader, test_loader = C.make_loaders(*datasets, batch_size=2, balance=BALANCE)
crit_g, crit_s = C.make_criteria(train_df, balance=BALANCE, device=device)
model = C.MultiTaskGliomaClassifier(in_channels=len(MODALITIES)).to(device)
"""),
    (C, """
CLF_PATH = str(paths.base / "models" / "classification" / "cnn3d_grade_profile.pt")
hist = C.train_classifier(model, train_loader, val_loader, crit_g, crit_s, device, epochs=60, out_path=CLF_PATH)
hist.plot(x="epoch", y=["train_loss", "val_loss"]); hist.plot(x="epoch", y=["val_grade_bal_acc", "val_subtype_bal_acc"])
"""),
    (C, """
import json
model = C.load_classifier(CLF_PATH, device)
res = C.run_epoch(model, test_loader, crit_g, crit_s, device)
report = C.classification_report_dict(res)
report.update({"model": "cnn3d", "split": "test", "modalities": MODALITIES})
os.makedirs(paths.eval_dir, exist_ok=True)
json.dump(report, open(paths.eval_dir / "cls_cnn3d.json", "w"), indent=1)   # shown in the app
report
"""),
    (C, """
pid = test_df["patient_id"][0]
C.predict_grade(model, file_index[pid], MODALITIES, DOWNSAMPLE, device=device)
"""),
    (M, """### Use the classifier in the app
Notebook 05 loads `models/classification/cnn3d_grade_profile.pt` automatically when it exists."""),
])

# ---------------------------------------------------------------- 03
nb("03_segmentation_3d_unet.ipynb", [
    (M, """# 03 — 3D tumor segmentation with MONAI (new)

Input: FLAIR, T1, T1ce, T2. Output: the 3 BraTS regions (TC, WT, ET) -> labels 0/1/2/4.
Models: `unet3d` (3D U-Net), `attention_unet`, `segresnet`, `swin_unetr`.
Metrics: Dice, IoU, HD95 per region on the test split -> `results/eval/seg_<model>.json` (read by the app).

**Runtime → Change runtime type → T4 GPU.** About 1-2 min per epoch on a T4. Checkpoints are written to
Drive after every epoch, so you can resume if Colab disconnects."""),
    (C, SETUP),
    (C, """
from braintumorlab.segmentation import SegConfig, prepare_cache, train_segmentation, evaluate_model
cfg = SegConfig(data_root=str(DATA_ROOT), cache_dir="/content/seg_cache",
                out_dir=str(paths.seg_models), model_name="unet3d",   # or "segresnet", "attention_unet"
                roi=(128, 128, 128), epochs=60, batch_size=1, val_every=2, max_val=20)
prepare_cache(cfg)          # float16 copies on the local disk (once per Colab session, ~5 min)
"""),
    (C, """
best, hist = train_segmentation(cfg)
# resume after a disconnection:
# best, hist = train_segmentation(cfg, resume=f"{cfg.out_dir}/{cfg.model_name}/last.pt")
"""),
    (C, """
import pandas as pd
h = pd.DataFrame(hist)
h.plot(x="epoch", y="train_loss"); h.dropna().plot(x="epoch", y=["val_dice_WT", "val_dice_TC", "val_dice_ET"])
"""),
    (C, """
result = evaluate_model(f"{cfg.out_dir}/{cfg.model_name}/best.pt", cfg, split="test", eval_dir=str(paths.eval_dir))
pd.DataFrame({r: {k: v["mean"] for k, v in result["summary"][r].items() if v} for r in ("WT", "TC", "ET")}).round(3)
"""),
    (M, """### Compare models
Change `cfg.model_name` (e.g. `"segresnet"`), re-run training and evaluation: every model gets its own
JSON file and appears in the **Model Comparison** page of the app."""),
])

# ---------------------------------------------------------------- 04
nb("04_medddpm_generation.ipynb", [
    (M, """# 04 — Med-DDPM 2D: T2 generation + scenarios

Uses the dataset `MedDDPM_dataset_T2/` and the checkpoint `MedDDPM2D_T2_models/best_model_EMA.pt`.
The architecture is read from the checkpoint itself (no need to know BASE / cond_ch / T).

* **Synthesis** = T2 generated from pure noise, conditioned on the other channel + segmentation.
  This is the honest "missing modality" test: PSNR / SSIM against the real T2.
* **Scenarios G2 / G3 / G4** are simulations for teaching (no follow-up MRI exists in BraTS to validate them)."""),
    (C, SETUP),
    (C, """
from braintumorlab.generation import MedDDPMGenerator
import matplotlib.pyplot as plt
gen = MedDDPMGenerator(paths.ddpm_root, paths.ddpm_checkpoint)
print(gen.info, "\\n", len(gen.patients()), "patients")
if gen.warning: print("WARNING:", gen.warning)
"""),
    (C, """
pid = gen.patients()[0]["patient_id"]
out = gen.run(pid, "synthesis", steps=50, n_samples=4, seed=0)
fig, ax = plt.subplots(1, 5, figsize=(22, 4.5))
for a, (k, t) in zip(ax, [("condition", "condition"), ("real", "real T2"), ("generated", "generated T2"),
                          ("error", "|error|"), ("uncertainty", "uncertainty")]):
    a.imshow(out[k], cmap="gray" if k in ("condition", "real", "generated") else "magma"); a.set_title(t); a.axis("off")
plt.show(); out["metrics"]
"""),
    (M, "### Evaluation on the test split (synthesis) -> `results/eval/gen_medddpm.json`"),
    (C, """
import json, numpy as np, os
rows = gen.evaluate("test", steps=50)
summary = {k: float(np.mean([r[k] for r in rows])) for k in ("PSNR", "SSIM", "MAE")}
os.makedirs(paths.eval_dir, exist_ok=True)
json.dump({"task": "generation", "model": "medddpm_t2", "split": "test", "n": len(rows), **summary,
           "per_case": rows}, open(paths.eval_dir / "gen_medddpm.json", "w"), indent=1)
summary
"""),
    (M, "### Scenarios"),
    (C, """
for mode in ("postop", "progression", "response"):
    o = gen.run(pid, mode, strength=0.05, steps=60)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(o["real"], cmap="gray"); ax[0].set_title("before")
    ax[1].imshow(o["generated"], cmap="gray"); ax[1].set_title(f"{mode} (simulated)")
    ax[2].imshow(o["error"], cmap="magma"); ax[2].set_title("difference")
    [a.axis("off") for a in ax]; plt.show(); print(mode, o["metrics"])
"""),
    (M, """### (Optional) retrain Med-DDPM
New checkpoints go to `MedDDPM2D_T2_models_v2/` (the current model is kept).
`fix_background=True` sets a constant background in the npz files to 0."""),
    (C, """
from braintumorlab.ddpm_train import DDPMConfig, train_ddpm
# best, hist = train_ddpm(DDPMConfig(ddpm_root=str(paths.ddpm_root), epochs=43))
"""),
])

# ---------------------------------------------------------------- 05
nb("05_launch_app.ipynb", [
    (M, """# 05 — Launch the BrainTumorLab web app in Colab

Pages: Dashboard · MRI Analysis · AI Segmentation · 3D Visualization · Radiomics · MRI Generation ·
Model Comparison & Reports. A GPU runtime makes segmentation and generation much faster."""),
    (C, SETUP),
    (C, """
import os
# use the classifier trained in notebook 02 when it exists
CLF = paths.base / "models" / "classification" / "cnn3d_grade_profile.pt"
if CLF.exists():
    os.environ["BTL_CLASSIFIER"] = str(CLF)
os.environ["BTL_DATA_ROOT"] = str(DATA_ROOT)
from braintumorlab.webapp.server import launch_in_colab
server = launch_in_colab(port=8000)      # click the "Open BrainTumorLab" link
"""),
    (M, "If the link does not open (browser blocks third-party cookies), show the app inside the notebook:"),
    (C, """
from google.colab import output
output.serve_kernel_port_as_iframe(8000, height=900)
"""),
])
