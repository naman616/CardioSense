"""
CardioSense FastAPI backend.

Serves the frontend HTML and exposes 3 inference endpoints:
    POST /api/predict   — ECG classification
    POST /api/gradcam   — Grad-CAM heatmap
    GET  /api/reference — Reference waveforms for the gallery

Run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import io
import base64
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    PredictRequest, PredictResponse,
    GradCAMRequest, GradCAMResponse,
    ReferenceResponse,
)
from api.deps import get_model

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]

app = FastAPI(title="CardioSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")
app.mount("/results", StaticFiles(directory=str(PROJECT_ROOT / "results")), name="results")


@app.get("/")
def index():
    return FileResponse(str(PROJECT_ROOT / "frontend/index.html"))


@app.get("/research")
def research_page():
    return FileResponse(str(PROJECT_ROOT / "frontend/research.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.values) != 187:
        raise HTTPException(400, f"Expected 187 values, got {len(req.values)}")

    model = get_model()
    signal = np.array(req.values, dtype=np.float32)

    from app.utils.inference import run_inference
    result = run_inference(model, signal)

    probs = {CLASS_NAMES[i]: float(result["probabilities"][i]) for i in range(5)}
    return PredictResponse(
        predicted_class=result["class_name"],
        class_index=result["class_idx"],
        confidence=result["confidence"],
        probabilities=probs,
    )


@app.post("/api/gradcam", response_model=GradCAMResponse)
def gradcam(req: GradCAMRequest):
    if len(req.values) != 187:
        raise HTTPException(400, f"Expected 187 values, got {len(req.values)}")

    model = get_model()
    signal = np.array(req.values, dtype=np.float32)

    from src.data.preprocessor import normalize
    from src.explainability.gradcam import GradCAM1D
    from src.explainability.heatmap_overlay import overlay_heatmap

    x = normalize(signal.reshape(1, -1))
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        pred_class = int(logits.argmax(1).item())
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        confidence = float(probs[pred_class])

    target_layer = model.get_gradcam_target_layer()
    cam = GradCAM1D(model, target_layer)
    heatmap = cam.compute(x_tensor, target_class=pred_class)

    plt.style.use("dark_background")
    fig = overlay_heatmap(
        ecg_signal=signal,
        heatmap=heatmap,
        predicted_class=pred_class,
        true_class=pred_class,
        confidence=confidence,
    )
    fig.patch.set_facecolor("#0e131e")
    for ax in fig.axes:
        ax.set_facecolor("#1a202a")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return GradCAMResponse(heatmap=heatmap.tolist(), image_base64=image_b64)


@app.get("/api/reference", response_model=ReferenceResponse)
def reference():
    npz_path = PROJECT_ROOT / "data/processed/reference_waveforms.npz"
    if not npz_path.exists():
        raise HTTPException(
            404,
            "Reference waveforms not found. Run notebooks/01_eda.ipynb first.",
        )
    data = np.load(str(npz_path))
    waveforms = {
        CLASS_NAMES[i]: data[f"class_{i}"].tolist()
        for i in range(5)
        if f"class_{i}" in data
    }
    return ReferenceResponse(waveforms=waveforms)
