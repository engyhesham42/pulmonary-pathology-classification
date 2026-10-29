import io
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from guidance import build_follow_up
from llm_service import build_context, generate_explanation
from model_utils import (default_explain_class, gradcam_base64, load_bundle,
                         predict_probs, preprocess, review_priority)
from schemas import Finding, HealthResponse, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/run3_finetuned.keras")
CONFIG_PATH = os.getenv("CONFIG_PATH", "models/inference_config.json")
HIGH_RATIO = float(os.getenv("PRIORITY_HIGH_RATIO", "1.25"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

state = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
        state["bundle"] = load_bundle(MODEL_PATH, CONFIG_PATH)
    yield


app = FastAPI(title="Chest X-ray Multi-Label API (educational prototype)", lifespan=lifespan)


def _bundle():
    if state["bundle"] is None:
        raise HTTPException(503, f"Model/config not found ({MODEL_PATH}, {CONFIG_PATH}).")
    return state["bundle"]


@app.get("/health", response_model=HealthResponse)
def health():
    b = state["bundle"]
    return HealthResponse(status="ok", model_loaded=b is not None,
                          model_file=b.model_file if b else None)


@app.get("/classes")
def classes():
    return {"classes": _bundle().classes}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), explain: bool = True,
                  explain_class: Optional[str] = None,
                  llm_explain: bool = False):
    b = _bundle()
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Upload a PNG or JPEG image.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read image file.")
    if explain_class and explain_class not in b.classes:
        raise HTTPException(400, f"Unknown class '{explain_class}'.")

    x = preprocess(b, img)
    probs = predict_probs(b, x)
    findings = sorted(
        (Finding(disease=c, probability=float(p), threshold=float(t), detected=bool(p >= t))
         for c, p, t in zip(b.classes, probs, b.thresholds)),
        key=lambda f: -f.probability)

    target = explain_class or default_explain_class(b, probs)
    heatmap = gradcam_base64(b, x, target, img) if explain else None
    detected = [f.disease for f in findings if f.detected]

    follow_up = explanation = None
    if llm_explain:
        # Deterministic follow-up first; the LLM only explains it. Predictions are
        # never modified and any LLM failure leaves them untouched.
        follow_up = build_follow_up(detected)
        preds = [{"class": f.disease, "probability": f.probability,
                  "positive": f.detected, "threshold": f.threshold} for f in findings]
        ctx = build_context(preds, 0.5, target, heatmap is not None, follow_up)
        explanation = {"class": target, **await run_in_threadpool(generate_explanation, ctx)}

    return PredictionResponse(
        findings=findings,
        detected=detected,
        review_priority=review_priority(findings, HIGH_RATIO),
        explained_class=target if explain else None,
        heatmap_base64=heatmap,
        model_file=b.model_file,
        follow_up=follow_up,
        explanation=explanation,
    )
