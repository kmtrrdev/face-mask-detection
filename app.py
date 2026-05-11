"""
Face Mask Detection — FastAPI App
Model: MobileNetV2 (Transfer Learning)
Classes: WithMask | WithoutMask
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import io
import uvicorn

MODEL_PATH = "mask_detector.pth"
CLASSES    = ["WithMask", "WithoutMask"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
model = None

def apply_clahe(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(apply_clahe),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.last_channel, 2))
    return m

@asynccontextmanager
async def lifespan(app):
    global model
    model = build_model()
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"WARNING: {MODEL_PATH} not found — using random weights.")
    model.to(DEVICE)
    model.eval()
    print(f"Device: {DEVICE}")
    yield

app = FastAPI(
    title="Face Mask Detection API",
    description="Detects whether a person is wearing a face mask using MobileNetV2.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "Face Mask Detection API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx   = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    return {
        "status":        "mask_on" if pred_class == "WithMask" else "mask_off",
        "action":        "Allow entry" if pred_class == "WithMask" else "Deny entry",
        "class":         pred_class,
        "confidence":    round(confidence, 4),
        "probabilities": {CLASSES[0]: round(float(probs[0]), 4), CLASSES[1]: round(float(probs[1]), 4)},
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)