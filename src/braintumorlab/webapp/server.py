"""BrainTumorLab web server (FastAPI). Serves the single-page interface and the JSON / binary API.

Local:  python -m braintumorlab.webapp.server --port 8000
Colab:  from braintumorlab.webapp.server import launch_in_colab; launch_in_colab()
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..volumes import MODALITIES, to_uint8
from .services import Lab

STATIC = Path(__file__).parent / "static"


def clean(o):
    """Make numpy / NaN values JSON friendly."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def png_b64(img, cmap=None):
    from PIL import Image

    a = np.asarray(img, np.float32)
    if cmap:
        import matplotlib

        rgba = matplotlib.colormaps[cmap](np.clip(a, 0, 1))
        im = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    elif a.ndim == 3:
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    else:
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay_rgb(gray, seg3, alpha=0.45):
    rgb = np.stack([gray] * 3, -1)
    colors = [(0.96, 0.25, 0.37), (0.13, 0.83, 0.65), (0.98, 0.75, 0.14)]   # NCR, ED, ET
    for ch, col in zip(seg3, colors):
        m = ch > 0.5
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(col)
    return rgb


def create_app(lab: Lab | None = None) -> FastAPI:
    lab = lab or Lab()
    app = FastAPI(title="BrainTumorLab", version="1.0")
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.state.lab = lab

    def err(e, code=400):
        raise HTTPException(code, str(e))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # ---------------------------------------------------------------- overview
    @app.get("/api/status")
    def status():
        gen = lab.generator()
        clf = lab.classifier()
        return clean({"device": str(lab.device), "data_root": str(lab.paths.data_root),
                      "segmentation_models": lab.segmentation_models(),
                      "classifier": ({"path": str(lab.paths.classifier), "legacy": clf[1]} if clf else None),
                      "ddpm": (gen.info if gen else None), "ddpm_error": lab._generator_error,
                      "ddpm_warning": gen.warning if gen else None})

    @app.get("/api/dashboard")
    def dashboard():
        cat = lab.catalog()
        brats = cat[cat["source"] == "brats"]
        vols = {}
        if "vol_WT_ml" in brats and brats["vol_WT_ml"].notna().any():
            for k in ("WT", "TC", "ET", "ED", "NCR"):
                v = brats[f"vol_{k}_ml"].dropna()
                vols[k] = {"median": float(v.median()), "mean": float(v.mean()), "n": int(v.size)}
            hist, edges = np.histogram(brats["vol_WT_ml"].dropna(), bins=12)
            vols["WT_hist"] = {"counts": hist.tolist(), "edges": edges.round(1).tolist()}
            means = {k: float(brats[f"vol_{k}_ml"].dropna().mean()) for k in ("NCR", "ED", "ET")}
            vols["composition"] = means
        evals = lab.eval_results()
        best = None
        for e in evals:
            if e.get("task") == "segmentation" and e["summary"]["WT"]["dice"]:
                d = np.mean([e["summary"][r]["dice"]["mean"] for r in ("WT", "TC", "ET") if e["summary"][r]["dice"]])
                if best is None or d > best["mean_dice"]:
                    best = {"model": e.get("model_label", e["model"]), "mean_dice": float(d), "n": e["n_cases"],
                            "split": e["split"]}
        return clean({"n_patients": int(len(brats)), "n_uploads": int((cat["source"] == "upload").sum()),
                      "splits": brats["split"].value_counts().to_dict(),
                      "grades": brats["grade_name"].fillna("unknown").value_counts().to_dict(),
                      "volumes": vols, "volumes_ready": int(brats["vol_WT_ml"].notna().sum()) if "vol_WT_ml" in brats else 0,
                      "best_segmentation": best, "n_models": len(lab.segmentation_models()),
                      "recent": lab.recent(10)})

    @app.get("/api/patients")
    def patients(q: str = "", split: str = "", limit: int = 500):
        cat = lab.catalog()
        if q:
            cat = cat[cat["patient_id"].str.contains(q, case=False) | cat["name"].fillna("").str.contains(q, case=False)]
        if split:
            cat = cat[cat["split"] == split]
        cols = [c for c in ("patient_id", "split", "source", "name", "grade_name", "vol_WT_ml", "vol_TC_ml",
                            "vol_ET_ml") if c in cat]
        return clean({"total": int(len(cat)), "patients": cat[cols].head(limit).to_dict("records")})

    # ---------------------------------------------------------------- patient
    @app.get("/api/patient/{pid}")
    def patient_info(pid: str):
        from ..analysis import tumor_localization
        from ..preprocessing import tumor_volumes

        try:
            p = lab.patient(pid)
        except Exception as e:
            err(e, 404)
        out = {"patient_id": pid, "shape": list(p["flair"].shape), "spacing": list(p["spacing"]),
               "missing": p.get("missing", []), "has_seg": p["seg"] is not None,
               "brain_volume_ml": float(p["brain"].sum() * np.prod(p["spacing"]) / 1000)}
        cat = lab.catalog()
        row = cat[cat["patient_id"] == pid].iloc[0]
        out.update({"split": row["split"], "grade": row.get("grade_name"), "source": row["source"],
                    "name": row.get("name")})
        if p["seg"] is not None:
            out["volumes"] = tumor_volumes(p["seg"], float(np.prod(p["spacing"])))
            out["localization"] = tumor_localization(p["seg"], p["brain"], p["affine"], float(p["spacing"][0]))
            tz = (p["seg"] > 0).sum((0, 1))
            out["tumor_center"] = [int(x) for x in np.argwhere(p["seg"] > 0).mean(0).round()]
            out["best_axial"] = int(tz.argmax())
        else:
            out["tumor_center"] = [s // 2 for s in p["flair"].shape]
        out["predictions"] = [m["name"] for m in lab.segmentation_models() if lab.prediction_path(m["name"], pid).exists()]
        lab.log(pid, "view")
        return clean(out)

    @app.get("/api/patient/{pid}/volume/{name}")
    def volume(pid: str, name: str):
        try:
            p = lab.patient(pid)
            if name in MODALITIES:
                arr = to_uint8(p[name], p["brain"])
            elif name == "seg":
                arr = lab.labels(pid, "gt").astype(np.uint8)
            elif name == "brain":
                arr = p["brain"].astype(np.uint8)
            elif name.startswith("pred:"):
                arr = lab.labels(pid, name[5:]).astype(np.uint8)
            else:
                raise ValueError(f"unknown volume {name}")
        except FileNotFoundError as e:
            err(e, 404)
        except Exception as e:
            err(e)
        return Response(np.ascontiguousarray(arr).tobytes(), media_type="application/octet-stream",
                        headers={"X-Shape": ",".join(map(str, arr.shape))})

    @app.post("/api/patient/{pid}/segment")
    def segment(pid: str, model: str):
        try:
            res, _ = lab.segment(pid, model)
        except Exception as e:
            err(e)
        return clean(res)

    @app.get("/api/patient/{pid}/mesh")
    def mesh(pid: str, source: str = "gt", step: int = 2):
        from ..mesh import patient_meshes

        try:
            p = lab.patient(pid)
            lab_ = lab.labels(pid, source) if (source != "gt" or p["seg"] is not None) else None
            return clean(patient_meshes(p["brain"], lab_, p["spacing"], step))
        except Exception as e:
            err(e)

    @app.get("/api/patient/{pid}/radiomics")
    def radiomics(pid: str, source: str = "gt"):
        from ..radiomics import compute_radiomics

        try:
            p = lab.patient(pid)
            res = compute_radiomics(p, lab.labels(pid, source))
        except Exception as e:
            err(e)
        lab.log(pid, "radiomics", None if source == "gt" else source,
                {"WT_ml": round(res["shape"]["WT"].get("volume_ml", 0), 2)})
        return clean(res)

    @app.get("/api/patient/{pid}/classify")
    def classify(pid: str):
        from ..classification import predict_grade

        c = lab.classifier()
        if c is None:
            err("No classifier found (train it with notebook 02).", 404)
        model, legacy = c
        n_in = model.encoder.features[0].in_channels
        mods = ["FLAIR", "T1ce", "T2", "T1"][:n_in] if n_in > 1 else ["FLAIR"]
        folder = lab.folder(pid)
        from ..volumes import nii_path

        entry = {m: nii_path(folder, m.lower()) for m in mods}
        if not all(entry.values()):
            err("Missing modality for the classifier.")
        res = predict_grade(model, entry, mods, 4, legacy=legacy, device=lab.device)
        res["legacy_model"] = legacy
        lab.log(pid, "classification", "cnn3d", {"grade": res["grade"], "p_HGG": round(res["prob_HGG"], 3)})
        return clean(res)

    @app.post("/api/patient/{pid}/analyze")
    def analyze(pid: str, model: str = ""):
        """Full automatic analysis of a case (no manual segmentation needed):
        segmentation -> volumes -> localisation -> shape radiomics -> grade."""
        from ..analysis import tumor_localization
        from ..radiomics import shape_features

        models = [m["name"] for m in lab.segmentation_models()]
        if not model and models:
            scored = [m for m in lab.segmentation_models() if m.get("test")]
            model = (max(scored, key=lambda m: m["test"]["WT"]["dice"]["mean"])["name"] if scored else models[0])
        out = {"patient_id": pid, "model": model or None, "steps": []}
        p = lab.patient(pid)
        if model:
            try:
                seg, labels = lab.segment(pid, model)
                out["segmentation"] = seg
                out["tumor_detected"] = bool((labels > 0).sum() * np.prod(p["spacing"]) > 100)   # > 0.1 mL
                out["localization"] = tumor_localization(labels, p["brain"], p["affine"], float(p["spacing"][0]))
                out["shape"] = {r: shape_features(m, p["spacing"]) for r, m in
                                {"WT": labels > 0, "TC": (labels == 1) | (labels == 4), "ET": labels == 4}.items()}
                out["steps"].append("segmentation")
            except Exception as e:
                out["segmentation_error"] = str(e)
        else:
            out["segmentation_error"] = "No segmentation model trained yet (notebook 03)."
        try:
            out["classification"] = classify(pid)
            out["steps"].append("classification")
        except HTTPException as e:
            out["classification_error"] = e.detail
        lab.log(pid, "full analysis", model or None,
                {"tumor": out.get("tumor_detected"), "grade": out.get("classification", {}).get("grade")})
        return clean(out)

    @app.post("/api/upload")
    async def upload(name: str = Form(""), flair: UploadFile = File(...), t1: UploadFile | None = File(None),
                     t1ce: UploadFile | None = File(None), t2: UploadFile | None = File(None),
                     seg: UploadFile | None = File(None)):
        tmp = tempfile.mkdtemp()
        files = {}
        try:
            for key, f in {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2, "seg": seg}.items():
                if f is None or not f.filename:
                    continue
                suffix = ".zip" if f.filename.lower().endswith(".zip") else (
                    ".nii.gz" if f.filename.lower().endswith(".gz") else ".nii")
                path = os.path.join(tmp, key + suffix)
                with open(path, "wb") as out:
                    shutil.copyfileobj(f.file, out)
                files[key] = path
            pid = lab.add_upload(files, name or None)
        except Exception as e:
            err(e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {"patient_id": pid}

    # ---------------------------------------------------------------- generation
    @app.get("/api/generation/patients")
    def gen_patients():
        g = lab.generator()
        if g is None:
            return clean({"available": False, "error": lab._generator_error, "patients": []})
        return clean({"available": True, "info": g.info, "warning": g.warning, "patients": g.patients()})

    @app.get("/api/generation/slices/{pid}")
    def gen_slices(pid: str):
        g = lab.generator()
        if g is None:
            err(lab._generator_error or "Med-DDPM not available", 404)
        return {"n_slices": g.num_slices(pid)}

    @app.post("/api/generation/run")
    def gen_run(req: dict = Body(...)):
        g = lab.generator()
        if g is None:
            err(lab._generator_error or "Med-DDPM not available", 404)
        try:
            with lab.lock:
                r = g.run(req["patient_id"], req.get("mode", "synthesis"), req.get("slice_index"),
                          float(req.get("strength", 0.3)), int(req.get("steps", 50)), int(req.get("seed", 0)),
                          int(req.get("n_samples", 1)), float(req.get("residual", 0.25)), int(req.get("grow", 11)))
        except Exception as e:
            err(e)
        imgs = {"condition": png_b64(r["condition"]), "real": png_b64(r["real"]),
                "real_seg": png_b64(overlay_rgb(r["real"], r["seg"])),
                "generated": png_b64(r["generated"]),
                "generated_seg": png_b64(overlay_rgb(r["generated"], r["seg_after"])),
                "error": png_b64(r["error"] / max(float(r["error"].max()), 1e-6), "magma")}
        if "uncertainty" in r:
            imgs["uncertainty"] = png_b64(r["uncertainty"] / max(float(r["uncertainty"].max()), 1e-6), "viridis")
        lab.log(r["patient_id"], "generation", r["mode"], {k: round(v, 3) for k, v in r["metrics"].items()
                                                             if isinstance(v, float) and math.isfinite(v)})
        return clean({"slice_index": r["slice_index"], "mode": r["mode"], "metrics": r["metrics"], "images": imgs})

    # ---------------------------------------------------------------- comparison & reports
    @app.get("/api/models/compare")
    def compare():
        evals = lab.eval_results()
        hist = {}
        for m in lab.segmentation_models():
            h = Path(m["path"]).parent / "history.json"
            if h.exists():
                import json

                hist[m["name"]] = json.load(open(h))
        return clean({"evaluations": evals, "histories": hist})

    @app.get("/api/report/{pid}")
    def report(pid: str, model: str = ""):
        out = {"patient": patient_info(pid)}
        if model:
            try:
                out["segmentation"], _ = lab.segment(pid, model)
            except Exception as e:
                out["segmentation_error"] = str(e)
        try:
            out["radiomics"] = radiomics(pid, model or "gt")
        except HTTPException as e:
            out["radiomics_error"] = e.detail
        try:
            out["classification"] = classify(pid)
        except HTTPException as e:
            out["classification_error"] = e.detail
        lab.log(pid, "report", model or None)
        return clean(out)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(port=8000, host="0.0.0.0", lab=None, background=False):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(lab), host=host, port=port, log_level="warning"))
    if background:
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        return server
    server.run()


def launch_in_colab(port=8000, height=900):
    """Start the server in a background thread and open it through the Colab proxy."""
    import time

    server = serve(port, background=True)
    while not server.started:
        time.sleep(0.2)
    from google.colab import output  # type: ignore

    output.serve_kernel_port_as_window(port, anchor_text="Open BrainTumorLab")
    print("If the link does not open, run: output.serve_kernel_port_as_iframe(%d, height=%d)" % (port, height))
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    serve(a.port, a.host)


if __name__ == "__main__":
    main()
