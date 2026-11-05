"""End-to-end tests on SYNTHETIC data (no BraTS needed).  pytest -q"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_demo_data import fake_patient  # noqa: E402

from braintumorlab import analysis as A  # noqa: E402
from braintumorlab.metrics import dice, hd95, region_metrics  # noqa: E402
from braintumorlab.preprocessing import one_hot, run_pipeline, tumor_volumes  # noqa: E402
from braintumorlab.volumes import labels_from_seg, list_patients, load_patient, remove_constant_background  # noqa: E402


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("btl")
    raw = root / "raw"
    rng = np.random.default_rng(1)
    ids = [f"BraTS20_Training_{i:03d}" for i in range(1, 9)]
    for pid in ids:
        fake_patient(str(raw / pid), pid, rng, shape=(120, 140, 110))
    pd.DataFrame({"Grade": ["HGG", "LGG"] * 4, "BraTS_2020_subject_ID": ids}).to_csv(raw / "name_mapping.csv", index=False)
    data = root / "pre"
    base = root / "miniProjetAI"
    base.mkdir()
    df = run_pipeline(str(raw), str(data))
    df.to_csv(base / "patients_metadata_2020_fini.csv", index=False)
    return {"root": root, "data": str(data), "base": base}


def test_conventions():
    seg = np.array([[[0, 1], [2, 4]]])
    assert (labels_from_seg(one_hot(seg)) == seg).all()
    v = tumor_volumes(seg)
    assert v["vol_WT"] == 3 and v["vol_TC"] == 2
    vol = np.full((6, 6, 6), -0.4, np.float32)
    vol[2:4, 2:4, 2:4] = 1
    out, fg = remove_constant_background(vol)
    assert out[0, 0, 0] == 0 and fg.sum() == 8
    a = np.zeros((20, 20, 20), bool)
    a[5:10, 5:10, 5:10] = True
    assert dice(a, a) == 1 and hd95(a, a) == 0
    assert np.isnan(hd95(a, np.zeros_like(a)))


def test_preprocessing_and_analysis(project):
    pid, _, folder = list_patients(project["data"], "train")[0]
    p = load_patient(folder)
    assert p["t2"][0, 0, 0] == 0 and abs(p["t2"][p["brain"]].mean()) < 1e-3
    meta = pd.read_csv(project["base"] / "patients_metadata_2020_fini.csv")
    assert (meta["vol_WT_ml"] < 200).all()          # no "whole image" volumes
    assert len(A.ablation_study(p)) == 15
    assert (A.modality_table(p)["CNR_tumor"].abs() > 0.5).all()
    loc = A.tumor_localization(p["seg"], p["brain"], p["affine"])
    assert loc["depth_to_surface_mm"] >= 0
    vol = A.volumes_table(project["data"])
    assert np.allclose(vol["vol_WT"], vol["vol_NCR"] + vol["vol_ED"] + vol["vol_ET"])


def test_radiomics_mesh(project):
    from braintumorlab.mesh import patient_meshes
    from braintumorlab.radiomics import compute_radiomics

    p = load_patient(list_patients(project["data"], "train")[0][2])
    r = compute_radiomics(p)
    assert 0 < r["shape"]["WT"]["sphericity"] <= 1.05
    assert r["texture"]["FLAIR"]
    m = patient_meshes(p["brain"], p["seg"], p["spacing"])
    assert m["brain"]["n_faces"] > 100


def test_segmentation_generation_api(project):
    from fastapi.testclient import TestClient

    from braintumorlab.config import Paths
    from braintumorlab.ddpm_data import build_npz_dataset
    from braintumorlab.ddpm_train import DDPMConfig, train_ddpm
    from braintumorlab.segmentation import SegConfig, evaluate_model, prepare_cache, train_segmentation
    from braintumorlab.webapp.server import create_app
    from braintumorlab.webapp.services import Lab

    base, data = project["base"], project["data"]
    ddpm_root = base / "MedDDPM_dataset_T2"
    build_npz_dataset(data, str(ddpm_root), ("flair",))
    train_ddpm(DDPMConfig(ddpm_root=str(ddpm_root), models_dir=str(ddpm_root / "MedDDPM2D_T2_models"), epochs=1,
                          batch_size=2, base=8, time_dim=16, num_timesteps=20, out_size=32, num_workers=0, val_every=1))
    cfg = SegConfig(data_root=data, cache_dir=str(project["root"] / "cache"), out_dir=str(base / "models" / "segmentation"),
                    model_name="segresnet", roi=(32, 32, 32), epochs=1, num_workers=0, use_amp=False, max_val=1)
    prepare_cache(cfg)
    best, _ = train_segmentation(cfg)
    res = evaluate_model(best, cfg, "test", str(base / "results" / "eval"), max_cases=1)
    assert set(res["summary"]) == {"WT", "TC", "ET"}

    os.environ["BTL_DATA_ROOT"] = data
    client = TestClient(create_app(Lab(Paths(base))))
    assert client.get("/").status_code == 200
    st = client.get("/api/status").json()
    assert st["segmentation_models"] and st["ddpm"]["cond_ch"] == 4
    pid = client.get("/api/patients").json()["patients"][0]["patient_id"]
    info = client.get(f"/api/patient/{pid}").json()
    r = client.get(f"/api/patient/{pid}/volume/flair")
    assert len(r.content) == np.prod(info["shape"])
    seg = client.post(f"/api/patient/{pid}/segment?model=segresnet").json()
    assert "WT" in seg["metrics"]
    assert len(client.get(f"/api/patient/{pid}/volume/pred:segresnet").content) == np.prod(info["shape"])
    assert client.get(f"/api/patient/{pid}/mesh").json()["brain"]
    assert client.get(f"/api/patient/{pid}/radiomics").json()["shape"]["WT"]["volume_ml"] > 0
    gp = client.get("/api/generation/patients").json()
    for mode in ("synthesis", "img2img", "postop", "progression", "response"):
        g = client.post("/api/generation/run", json={"patient_id": gp["patients"][0]["patient_id"], "mode": mode,
                                                     "steps": 3, "n_samples": 2}).json()
        assert g["images"]["generated"].startswith("data:image/png")
    assert client.get("/api/models/compare").json()["evaluations"]
    assert client.get("/api/dashboard").json()["n_patients"] == 8
    rep = client.get(f"/api/report/{pid}?model=segresnet").json()
    assert "segmentation" in rep and "radiomics" in rep
    assert client.get(f"/api/patient/{pid}/classify").status_code == 404   # no classifier in this project
    full = client.post(f"/api/patient/{pid}/analyze").json()
    assert "segmentation" in full and "tumor_detected" in full
