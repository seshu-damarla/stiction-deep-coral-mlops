"""
Tests for model artifacts and src/model.py.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import torch

from src.model import load_CNNencoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ENCODER_PATH = ARTIFACT_DIR / "encoder.pt"
CLASSIFIER_PATH = ARTIFACT_DIR / "logistic_regression.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.json"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

def test_required_artifact_files_exist():
    assert ENCODER_PATH.exists()
    assert CLASSIFIER_PATH.exists()
    assert THRESHOLD_PATH.exists()
    assert METADATA_PATH.exists()

def test_encoder_loads_successfully():
    encoder, device = load_CNNencoder()
    assert encoder is not None
    assert isinstance(device, torch.device)

def test_encoder_is_in_evaluation_mode():
    encoder, _ = load_CNNencoder()
    assert encoder.training is False

def test_encoder_output_dimension_is_64():
    encoder, device = load_CNNencoder()
    dummy_image = torch.zeros(1, 3, 50, 50, dtype=torch.float32, device=device,)

    with torch.no_grad():
        features = encoder(dummy_image)
    assert features.shape == (1, 64)

def test_encoder_output_is_finite():
    encoder, device = load_CNNencoder()
    dummy_image = torch.ones( 1, 3, 50, 50,dtype=torch.float32,device=device,)

    with torch.no_grad():
        features = encoder(dummy_image)
    assert torch.isfinite(features).all()

def test_logistic_regression_pipeline_loads():
    classifier = joblib.load(CLASSIFIER_PATH)

    assert hasattr(classifier, "predict_proba")
    assert hasattr(classifier, "named_steps")
    assert "scaler" in classifier.named_steps
    assert "model" in classifier.named_steps

def test_classifier_accepts_64_features():
    classifier = joblib.load(CLASSIFIER_PATH)
    dummy_features = np.zeros((1, 64), dtype=np.float32,)
    probability = classifier.predict_proba(dummy_features)

    assert probability.shape == (1, 2)
    assert np.isfinite(probability).all()

def test_probability_sums_to_one():
    classifier = joblib.load(CLASSIFIER_PATH)
    dummy_features = np.zeros((1, 64), dtype=np.float32,)
    probability = classifier.predict_proba(dummy_features)
    assert np.isclose(probability.sum(), 1.0)

def test_threshold_is_0_point_5():
    with open(THRESHOLD_PATH,"r",encoding="utf-8",) as file:
        threshold = json.load(file)["threshold"]

    assert float(threshold) == 0.50
