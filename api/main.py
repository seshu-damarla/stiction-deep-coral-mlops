# ============================================================
# api/main.py
# ============================================================

from io import BytesIO
from pathlib import Path
import json

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from PIL import Image

from src.inference import stictionpredictor


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "model_metadata.json"
)


# ============================================================
# 2. Create FastAPI application
# ============================================================

app = FastAPI(
    title="Stiction Deep CORAL API",
    description=(
        "API for control-valve stiction detection "
        "using OT images, Deep CORAL features, "
        "and Logistic Regression."
    ),
    version="1.0.0",
)


# ============================================================
# 3. Load predictor ONCE
# ============================================================

predictor = stictionpredictor()


# ============================================================
# 4. Load model metadata
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    model_metadata = json.load(
        file
    )


# ============================================================
# 5. Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Stiction Deep CORAL API is running",

        "docs":
            "/docs",
    }


# ============================================================
# 6. Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model_loaded":
            predictor is not None,
    }


# ============================================================
# 7. Model information endpoint
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_name":
            model_metadata[
                "model_name"
            ],

        "image_type":
            "OT image",

        "encoder":
            "Deep CORAL CNN encoder",

        "classifier":
            "StandardScaler + Logistic Regression",

        "input_size":
            [
                3,
                50,
                50,
            ],

        "feature_dimension":
            64,

        "threshold":
            model_metadata[
                "logistic_regression"
            ][
                "decision_threshold"
            ],
    }


# ============================================================
# 8. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check uploaded file type
    # --------------------------------------------------------

    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg",
    }


    if (
        file.content_type
        not in allowed_types
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PNG and JPEG images "
                "are supported."
            ),
        )


    # --------------------------------------------------------
    # Read uploaded bytes
    # --------------------------------------------------------

    try:

        file_bytes = await file.read()


        image = Image.open(
            BytesIO(
                file_bytes
            )
        )


        image = image.convert(
            "RGB"
        )


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from error


    # --------------------------------------------------------
    # Run model inference
    # --------------------------------------------------------

    try:

        result = predictor.predict(
            image
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error


    # --------------------------------------------------------
    # Return inference output
    # --------------------------------------------------------

    return result