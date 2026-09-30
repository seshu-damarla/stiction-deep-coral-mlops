"""
Tests for src/inference.py.
"""
import numpy as np
import pytest
from PIL import Image

from src.inference import stictionpredictor

# pytest is the standard testing framework for Python because it makes writing, running,
# and scaling tests significantly easier than built-in or legacy options (like unittest).

@pytest.fixture(scope="module")
def predictor():
    return stictionpredictor()

def make_test_image():
    return Image.new("RGB",(80, 80), color=(120, 80, 40),)

def test_predictor_initializes(predictor):
    assert predictor is not None

def test_extract_features_returns_numpy_array(predictor):
    features = predictor.extract_features(make_test_image())
    assert isinstance(features, np.ndarray)

def test_extract_features_shape_is_1_by_64(predictor):
    features = predictor.extract_features(make_test_image())
    assert features.shape == (1, 64)

def test_extract_features_are_finite(predictor):
    features = predictor.extract_features(make_test_image())
    assert np.isfinite(features).all()

def test_predict_returns_dictionary(predictor):
    result = predictor.predict(make_test_image())
    assert isinstance(result, dict)

def test_predict_returns_required_keys(predictor):
    result = predictor.predict(make_test_image())
    expected_keys = {"prediction", "diagnosis", "stiction_probability", "threshold", "inference_time_seconds",}
    assert expected_keys.issubset(result.keys())

def test_prediction_is_binary(predictor):
    result = predictor.predict(make_test_image())
    assert result["prediction"] in (0, 1)


def test_diagnosis_text_is_valid(predictor):
    result = predictor.predict(make_test_image())
    assert result["diagnosis"] in {"Stiction", "No Stiction"}

def test_probability_is_between_zero_and_one(predictor):
    result = predictor.predict(make_test_image())
    probability = result["stiction_probability"]
    assert 0.0 <= probability <= 1.0

def test_threshold_equals_deployment_threshold(predictor):
    result = predictor.predict(make_test_image())
    assert result["threshold"] == 0.50

def test_inference_time_is_non_negative(predictor):
    result = predictor.predict(make_test_image())
    assert result["inference_time_seconds"] >= 0.0

def test_prediction_matches_probability_threshold(predictor):
    result = predictor.predict(make_test_image())
    expected_prediction = int(result["stiction_probability"]>= result["threshold"])
    assert result["prediction"] == expected_prediction
