"""
Tests for src/preprocessing.py
"""
import numpy as np
import torch
from PIL import Image
from src.preprocessing import preprocess_image

def test_preprocess_returns_torch_tensor():
    image = Image.new("RGB", (100, 80), color=(255, 0, 0))
    tensor = preprocess_image(image)
    assert isinstance(tensor, torch.Tensor)

def test_preprocess_output_shape():
    image = Image.new("RGB", (120, 90), color=(100, 150, 200))
    tensor = preprocess_image(image)
    assert tensor.shape == (1, 3, 50, 50)

def test_preprocess_dtype_is_float32():
    image = Image.new("RGB", (50, 50), color=(20, 30, 40))
    tensor = preprocess_image(image)
    assert tensor.dtype == torch.float32

def test_preprocess_pixel_range():
    image = Image.new("RGB", (50, 50), color=(255, 128, 0))
    tensor = preprocess_image(image)
    assert torch.min(tensor).item() >= 0.0
    assert torch.max(tensor).item() <= 1.0

def test_preprocess_contains_no_nan_or_inf():
    image = Image.new("RGB", (73, 91), color=(10, 100, 240))
    tensor = preprocess_image(image)
    assert torch.isfinite(tensor).all()

def test_preprocess_preserves_rgb_channel_order():
    image = Image.new("RGB", (50, 50), color=(255, 0, 0))
    tensor = preprocess_image(image)

    red_value = tensor[0, 0, 0, 0].item()
    green_value = tensor[0, 1, 0, 0].item()
    blue_value = tensor[0, 2, 0, 0].item()

    assert np.isclose(red_value, 1.0)
    assert np.isclose(green_value, 0.0)
    assert np.isclose(blue_value, 0.0)
