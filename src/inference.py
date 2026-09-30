# this inference files works in the following way
#     ├── call preprocessing.py
#     │
#     ├── call trained encoder
#     │
#     ├── obtain learned features
#     │
#     ├── call logistic regression
#     │
#     ├── apply threshold
#     │
#     └── return diagnosis

from pathlib import Path
import json
import time
import joblib
import numpy as np
import torch
from PIL import Image

from src.model import load_CNNencoder
from src.preprocessing import preprocess_image

class stictionpredictor:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        artifacts_dir = project_root / "artifacts"

        # loading CNN encoder
        self.encoder, self.device = load_CNNencoder()
        # load logistic regression classifier
        logistic_model_path = project_root / "artifacts" / "logistic_regression.joblib"
        self.classifier = joblib.load(logistic_model_path)

        threshold_path = artifacts_dir / "threshold.json"

        with open(threshold_path, "r") as file:
            threshold_data = json.load(file)

        self.threshold = float(threshold_data["threshold"])

        print("Stiction predictor initialized.")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold}")

    def extract_features(self, image:Image.Image) -> np.ndarray:
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # this won't compute gradients, just feature extraction from trained encoder
        with torch.no_grad():
            features = self.encoder(image_tensor)

        #  Move features back to CPU
        features = features.cpu()
        # Convert PyTorch tensor into NumPy array
        features = features.numpy()

        return features

    def predict(self, image:Image.Image) -> dict:
        '''
        This function returns prediction, diagnosis, probability, threshold, inference time
        '''
        start_time = time.perf_counter()
        features = self.extract_features(image)

        probabilities = (self.classifier.predict_proba(features))
        stiction_probability = float(probabilities[0, 1])

        if stiction_probability >= self.threshold:
            prediction = 1
            diagnosis = "Stiction"
        else:
            prediction = 0
            diagnosis = "No Stiction"

        end_time = time.perf_counter()

        inference_time_seconds = end_time - start_time

        result_dict = {
            "prediction": prediction,
            "diagnosis": diagnosis,
            "stiction_probability": stiction_probability,
            "threshold": self.threshold,
            "inference_time_seconds": inference_time_seconds}

        return result_dict

# manually testing if this inference code provides predictions
if __name__ == "__main__":

    predictor = stictionpredictor()
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "demo_data" / "CHEM13.png"

    image = Image.open(image_path)
    result = predictor.predict(image)

    print("\nPrediction Result")
    print("-----------------------------")
    print("Diagnosis:", result["diagnosis"])
    print("Stiction probability:", round(result["stiction_probability"], 4))
    print("Threshold:", result["threshold"])
    print("Inference time:", round(result["inference_time_seconds"], 3), "ms")




















