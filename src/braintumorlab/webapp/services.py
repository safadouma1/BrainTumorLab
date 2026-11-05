"""Application services: patient catalog, caches, models and analysis history.

Everything shown in the web app comes from here and is computed from the real data and the real
trained models. When a model has not been trained yet the service says so; it never invents numbers.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Paths
from ..volumes import MODALITIES, list_patients, load_patient, stack_modalities


class LRU(OrderedDict):
    def __init__(self, size=3):
        super().__init__()
        self.size = size

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = fn()
        self[key] = val
        if len(self) > self.size:
            self.popitem(last=False)
        return val


class Lab:
    def __init__(self, paths: Paths | None = None, device=None):
        self.paths = paths or Paths()
        try:
            self.paths.ensure_data_extracted()
        except FileNotFoundError as e:
            print("WARNING:", e)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.work = Path(self.paths.work)
        for d in ("uploads", "predictions", "cache"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.patients_cache = LRU(3)
        self.models_cache = LRU(2)
        self._classifier = None
        self._generator = None
        self._generator_error = None
        self._init_db()
        self.volumes = self._load_volume_cache()
        threading.Thread(target=self._compute_volumes, daemon=True).start()

    # ------------------------------------------------------------------ history
    def _init_db(self):
        self.db_path = str(self.work / "history.db")
        with sqlite3.connect(self.db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, patient_id TEXT, "
                      "kind TEXT, model TEXT, summary TEXT)")

    def log(self, pid, kind, model=None, summary=None):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO events (ts, patient_id, kind, model, summary) VALUES (?,?,?,?,?)",
                      (time.time(), pid, kind, model, json.dumps(summary or {})))

    def recent(self, n=12):
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT ts, patient_id, kind, model, summary FROM events ORDER BY id DESC LIMIT ?",
                             (n,)).fetchall()
        return [{"ts": r[0], "patient_id": r[1], "kind": r[2], "model": r[3], "summary": json.loads(r[4])}
                for r in rows]

    # ------------------------------------------------------------------ catalog
    def catalog(self):
        rows = []
        root = self.paths.data_root
        if root and Path(root).is_dir():
            rows += [{"patient_id": p, "split": s, "folder": f, "source": "brats"} for p, s, f in list_patients(root)]
        for d in sorted(glob.glob(str(self.work / "uploads" / "*"))):
            meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
            rows.append({"patient_id": os.path.basename(d), "split": "upload", "folder": d, "source": "upload",
                         "name": meta.get("name")})
        df = pd.DataFrame(rows, columns=["patient_id", "split", "folder", "source", "name"])
        meta = self.metadata()
        if meta is not None and len(df):
            df = df.merge(meta[["patient_id", "grade_name"]], on="patient_id", how="left")
        else:
            df["grade_name"] = None
        if self.volumes is not None and len(df):
            df = df.merge(self.volumes, on="patient_id", how="left")
        return df

    def metadata(self):
        p = Path(self.paths.metadata_csv)
        if not p.exists():
            alt = Path(self.paths.data_root) / "patients_metadata_2020.csv"
            p = alt if alt.exists() else None
        if p is None:
            return None
        m = pd.read_csv(p)
        if "grade_name" not in m.columns and "grade_label" in m.columns:
            m["grade_name"] = m["grade_label"].map({0: "LGG", 1: "HGG"})
        return m

    def folder(self, pid):
        cat = self.catalog()
        hit = cat[cat["patient_id"] == pid]
        if hit.empty:
            raise KeyError(f"Unknown patient {pid}")
        return hit.iloc[0]["folder"]

    def patient(self, pid):
        return self.patients_cache.get_or(pid, lambda: load_patient(self.folder(pid)))

    # ------------------------------------------------------------------ volumes (correct ones)
    def _load_volume_cache(self):
        p = self.work / "cache" / "volumes.csv"
        return pd.read_csv(p) if p.exists() else None

    def _compute_volumes(self):
        """Correct tumor volumes for every patient, computed once in the background."""
        from ..analysis import load_seg
        from ..preprocessing import tumor_volumes

        root = self.paths.data_root
        if not root or not Path(root).is_dir():
            return
        pats = list_patients(root)
        done = set(self.volumes["patient_id"]) if self.volumes is not None else set()
        rows = [] if self.volumes is None else self.volumes.to_dict("records")
        for pid, _, folder in pats:
            if pid in done:
                continue
            try:
                seg, _, sp = load_seg(folder)
                v = tumor_volumes(seg, float(np.prod(sp)))
                rows.append({"patient_id": pid, **{k: v[k] for k in ("vol_WT_ml", "vol_TC_ml", "vol_ET_ml",
                                                                    "vol_ED_ml", "vol_NCR_ml")}})
            except Exception as e:
                print("volume error", pid, e)
            if len(rows) % 20 == 0:
                self.volumes = pd.DataFrame(rows)
        self.volumes = pd.DataFrame(rows)
        if len(rows):
            self.volumes.to_csv(self.work / "cache" / "volumes.csv", index=False)

    # ------------------------------------------------------------------ models
    def segmentation_models(self):
        out = []
        for ck in sorted(glob.glob(str(Path(self.paths.seg_models) / "*" / "best.pt"))):
            name = Path(ck).parent.name
            info = {"name": name, "path": ck}
            hist = Path(ck).parent / "history.json"
            if hist.exists():
                h = json.load(open(hist))
                info["epochs_trained"] = len(h)
                vals = [r["val_dice_mean"] for r in h if "val_dice_mean" in r]
                info["best_val_dice"] = max(vals) if vals else None
            ev = Path(self.paths.eval_dir) / f"seg_{name}.json"
            if ev.exists():
                info["test"] = json.load(open(ev))["summary"]
            out.append(info)
        return out

    def seg_model(self, name):
        from ..segmentation import load_seg_model

        path = Path(self.paths.seg_models) / name / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Segmentation model '{name}' not trained (train it with notebook 03).")
        return self.models_cache.get_or(name, lambda: load_seg_model(str(path), self.device))

    def classifier(self):
        if self._classifier is None:
            from ..classification import load_classifier

            p = Path(self.paths.classifier)
            if not p.exists():
                return None
            self._classifier = (load_classifier(str(p), self.device), p.suffix == ".pkl")
        return self._classifier

    def generator(self):
        if self._generator is None and self._generator_error is None:
            try:
                from ..generation import MedDDPMGenerator

                self._generator = MedDDPMGenerator(self.paths.ddpm_root, self.paths.ddpm_checkpoint,
                                                   device=self.device)
            except Exception as e:
                self._generator_error = str(e)
        return self._generator

    def eval_results(self):
        out = []
        for f in sorted(glob.glob(str(Path(self.paths.eval_dir) / "*.json"))):
            try:
                out.append(json.load(open(f)))
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ analyses
    def prediction_path(self, model, pid):
        return self.work / "predictions" / model / f"{pid}.npz"

    def segment(self, pid, model_name):
        from ..metrics import region_metrics
        from ..preprocessing import tumor_volumes
        from ..segmentation import predict_volume

        out = self.prediction_path(model_name, pid)
        p = self.patient(pid)
        t0 = time.time()
        if out.exists():
            labels = np.load(out)["labels"]
        else:
            if p.get("missing"):
                raise ValueError(f"Missing modalities {p['missing']}: segmentation needs FLAIR, T1, T1ce and T2.")
            model, ck = self.seg_model(model_name)
            roi = tuple(ck.get("config", {}).get("roi", (128, 128, 128)))
            with self.lock:
                labels, _ = predict_volume(model, stack_modalities(p), roi, self.device)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, labels=labels)
        res = {"patient_id": pid, "model": model_name, "time_s": round(time.time() - t0, 2),
               "volumes_pred": tumor_volumes(labels, float(np.prod(p["spacing"])))}
        if p["seg"] is not None:
            res["metrics"] = region_metrics(labels, p["seg"], p["spacing"])
        self.log(pid, "segmentation", model_name,
                 {r: round(res["metrics"][r]["dice"], 4) for r in ("WT", "TC", "ET")} if "metrics" in res else {})
        return res, labels

    def labels(self, pid, source="gt"):
        if source in (None, "", "gt"):
            seg = self.patient(pid)["seg"]
            if seg is None:
                raise ValueError("No ground-truth segmentation for this case; run a segmentation model first.")
            return seg
        path = self.prediction_path(source, pid)
        if not path.exists():
            self.segment(pid, source)
        return np.load(path)["labels"]

    def add_upload(self, files: dict, name=None):
        from ..volumes import prepare_uploaded_case

        pid = "UPLOAD_" + uuid.uuid4().hex[:8].upper()
        d = self.work / "uploads" / pid
        try:
            prepare_uploaded_case(files, d)
        except Exception:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            raise
        json.dump({"name": name or pid, "created": time.time()}, open(d / "meta.json", "w"))
        self.log(pid, "upload", None, {"name": name or pid})
        return pid
