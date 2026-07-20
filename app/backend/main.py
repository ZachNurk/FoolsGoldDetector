"""FastAPI inference service for the Gold/Pyrite classifier.

Usage:
    uvicorn backend.main:app --reload --port 8000
"""
import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "model"))
from common import load_model, predict  # noqa: E402

CHECKPOINT = REPO_ROOT / "model" / "checkpoints" / "model_best.pth.tar"

DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
EXTRA_ORIGINS = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]

app = FastAPI(title="Fools Gold Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + EXTRA_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, classes, resolution = load_model(str(CHECKPOINT))


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": classes}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image file")

    probs = predict(model, classes, resolution, img)
    top_class = max(probs, key=probs.get)

    return {
        "classes": classes,
        "probabilities": probs,
        "prediction": top_class,
    }
