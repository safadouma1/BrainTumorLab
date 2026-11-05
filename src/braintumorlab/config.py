"""Project paths and global settings.

Every path can be overridden with an environment variable, so the same code runs
in Google Colab (Drive mounted at /content/drive) and on a local machine.

Default Drive layout (folder ``MyDrive/miniProjetAI``)::

    archive.zip                                     raw BraTS 2020 (Kaggle archive)
    BraTS_2020_Data_preprocessing_nii_FIXED.zip     preprocessed NIfTI (train/val/test/<patient>/...)
    patients_metadata_2020_fini.csv                 grade (HGG/LGG) + split per patient
    MedDDPM_dataset_T2/index_MedDDPM_T2_ready.csv   Med-DDPM npz index
    MedDDPM_dataset_T2/MedDDPM2D_T2_models/best_model_EMA.pt
    best_multitask_glioma_classifier_2020_pickle_v2.pkl
    models/segmentation/<model_name>/best.pt        3D U-Net checkpoints (notebook 03)
    results/eval/*.json                             evaluation results shown in the app
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE_CANDIDATES = ["/content/drive/MyDrive/miniProjetAI", "./miniProjetAI", "."]


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_base_path() -> Path:
    env = os.environ.get("BTL_BASE")
    for p in ([env] if env else []) + BASE_CANDIDATES:
        if p and os.path.isdir(p):
            return Path(p).resolve()
    raise FileNotFoundError(
        "Project folder not found. Mount Google Drive or set the BTL_BASE environment variable."
    )


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass
class Paths:
    base: Path = field(default_factory=find_base_path)

    def __post_init__(self):
        b = Path(self.base)
        self.base = b
        # preprocessed NIfTI dataset (folder with train/ val/ test/)
        self.data_root = _env_path("BTL_DATA_ROOT", Path("/content/BraTS_2020_Data_preprocessing_nii_FIXED"))
        self.data_zip = _env_path("BTL_DATA_ZIP", b / "BraTS_2020_Data_preprocessing_nii_FIXED.zip")
        self.metadata_csv = _env_path("BTL_METADATA", b / "patients_metadata_2020_fini.csv")
        self.ddpm_root = _env_path("BTL_DDPM_ROOT", b / "MedDDPM_dataset_T2")
        self.ddpm_checkpoint = _env_path(
            "BTL_DDPM_CKPT", self.ddpm_root / "MedDDPM2D_T2_models" / "best_model_EMA.pt")
        self.classifier = _env_path("BTL_CLASSIFIER", b / "best_multitask_glioma_classifier_2020_pickle_v2.pkl")
        self.seg_models = _env_path("BTL_SEG_MODELS", b / "models" / "segmentation")
        self.eval_dir = _env_path("BTL_EVAL_DIR", b / "results" / "eval")
        self.work = _env_path("BTL_WORK", b / "app_data")   # uploads, predictions, cache, sqlite db

    def ensure_data_extracted(self) -> Path:
        """Unzip the preprocessed dataset from Drive to local disk (much faster than reading Drive)."""
        from .volumes import find_data_root

        marker = self.data_root / ".extraction_complete"
        if self.data_root.is_dir() and any(self.data_root.iterdir()):
            if marker.exists() or not self.data_zip.exists():
                self.data_root = Path(find_data_root(self.data_root))
                return self.data_root
            print("Incomplete extraction found (interrupted?) -> extracting again.")
            import shutil

            shutil.rmtree(self.data_root)
        if not self.data_zip.exists():
            raise FileNotFoundError(f"Preprocessed dataset not found: {self.data_root} / {self.data_zip}")
        import shutil
        import subprocess
        import zipfile

        print(f"Extracting {self.data_zip} -> {self.data_root} (5-10 min, do not stop the cell) ...")
        self.data_root.mkdir(parents=True, exist_ok=True)
        if shutil.which("unzip"):   # much faster than Python's zipfile
            subprocess.run(["unzip", "-q", "-o", str(self.data_zip), "-d", str(self.data_root)], check=True)
        else:
            with zipfile.ZipFile(self.data_zip) as z:
                z.extractall(self.data_root)
        marker.touch()
        self.data_root = Path(find_data_root(self.data_root))
        return self.data_root
