"""Vercel serverless entrypoint: FastAPI + ONNX Runtime (no torch, to stay under function size limits)."""
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = REPO_ROOT / "model" / "checkpoints"
RESOLUTION = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES = (CHECKPOINTS / "labels.txt").read_text().split()
SESSION = ort.InferenceSession(str(CHECKPOINTS / "resnet18.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name

app = FastAPI(title="Fools Gold Detector API")


def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")

    # Resize (shorter side) + center crop, matching torchvision's Resize+CenterCrop(224).
    w, h = img.size
    scale = RESOLUTION / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left = (w - RESOLUTION) // 2
    top = (h - RESOLUTION) // 2
    img = img.crop((left, top, left + RESOLUTION, top + RESOLUTION))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW
    return arr.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image file")

    x = preprocess(img)
    logits = SESSION.run(None, {INPUT_NAME: x})[0][0]
    probs = softmax(logits)

    result = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    top_class = max(result, key=result.get)

    return {
        "classes": CLASSES,
        "probabilities": result,
        "prediction": top_class,
    }
