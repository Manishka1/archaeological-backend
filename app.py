import io
import os
import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from soil_model import SoilClassifier
from veg_model import VegetationModel

# -------------------------------------------------
# APP INITIALIZATION
# -------------------------------------------------

app = FastAPI(
    title="Archaeological Site Mapping API",
    description="AI-powered soil classification and vegetation analysis",
    version="1.0.0"
)

# -------------------------------------------------
# CORS (Allow Vercel / Public Frontend)
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for public demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# LOAD MODELS (ON STARTUP)
# -------------------------------------------------

soil_model = SoilClassifier()
veg_model = VegetationModel()

# -------------------------------------------------
# HEALTH CHECK (IMPORTANT FOR RENDER)
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Archaeological Site Mapping API"
    }

@app.get("/health")
def health_check():
    return {"ok": True}

# -------------------------------------------------
# SOIL CLASSIFICATION ENDPOINT
# -------------------------------------------------

@app.post("/soil")
async def analyze_soil(file: UploadFile = File(...)):
    """
    Image-based soil classification.
    Returns soil type and confidence.
    """

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    soil_type, confidence = soil_model.predict(image)

    return {
        "soil_type": soil_type,
        "confidence": round(confidence * 100, 2)
    }

# -------------------------------------------------
# VEGETATION ANALYSIS ENDPOINT
# -------------------------------------------------

@app.post("/vegetation")
async def analyze_vegetation(file: UploadFile = File(...)):
    """
    Vegetation segmentation and coverage estimation.
    Returns vegetation vs non-vegetation percentage.
    """

    image_bytes = await file.read()
    image_rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

    veg_percent, non_veg_percent = veg_model.predict(image_bgr)

    return {
        "vegetation_percent": veg_percent,
        "non_vegetation_percent": non_veg_percent
    }

# -------------------------------------------------
# OPTIONAL: RENDER ENTRY POINT (SAFE)
# -------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)



@app.on_event("startup")
def warmup_models():
    import numpy as np
    import cv2

    # Dummy black image
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)

    try:
        veg_model.predict(dummy)
        print("✅ Vegetation model warmed up")
    except Exception as e:
        print("⚠️ Warmup failed:", e)