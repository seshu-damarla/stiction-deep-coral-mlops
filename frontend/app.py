# ============================================================
# frontend/app.py
# ============================================================

import os
import requests
import streamlit as st
from PIL import Image

# ============================================================
# 1. FastAPI address
# ============================================================

def get_fastapi_url():

    # 1. Environment variable
    # Used by Docker Compose / Kubernetes
    api_url = os.getenv("FASTAPI_URL")

    if api_url:
        return api_url.rstrip("/")

    # 2. Streamlit Community Cloud secret
    try:
        api_url = st.secrets["FASTAPI_URL"]
        return api_url.rstrip("/")
    except Exception:
        pass

    # 3. Local development fallback
    return "http://127.0.0.1:8000"


FASTAPI_URL = get_fastapi_url()
# ============================================================
# 2. Page configuration
# ============================================================
st.set_page_config(
    page_title="Control Valve Stiction Detection",
    page_icon="⚙️",
    layout="centered",
)

# ============================================================
# 3. Page title
# ============================================================
st.title("Control Valve Stiction Detection")
st.write("Deep CORAL based control-valve stiction detection using Optimal Transport images.")

# ============================================================
# 4. Display API address
# ============================================================
st.caption(f"FastAPI service: {FASTAPI_URL}")

# ============================================================
# 5. Helper function: check API health
# ============================================================
def check_api_health():
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

# ============================================================
# 6. Helper function: get model information
# ============================================================
def get_model_info():
    try:
        response = requests.get(f"{FASTAPI_URL}/model-info", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

# ============================================================
# 7. Helper function: send image for prediction
# ============================================================
def predict_image(uploaded_file):
    # Move file pointer back to beginning.
    uploaded_file.seek(0)

    # Prepare multipart/form-data request.
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    # Send image to FastAPI.
    response = requests.post(f"{FASTAPI_URL}/predict", files=files, timeout=30)
    response.raise_for_status()
    return response.json()

# ============================================================
# 8. Check FastAPI health
# ============================================================
health = check_api_health()

if health is None:
    st.error("FastAPI is not available. Start the FastAPI server before using the frontend.")
else:
    status = health.get("status", "unknown")
    model_loaded = health.get("model_loaded", False)

    if status == "healthy" and model_loaded:
        st.success("FastAPI is healthy and the model is loaded.")
    else:
        st.warning("FastAPI is running, but the model may not be ready.")

    # ========================================================
    # 9. Model information
    # ========================================================
    model_info = get_model_info()

    if model_info is not None:
        with st.expander("Model Information"):
            st.write("**Model:**", model_info.get("model_name", "Unknown"))
            st.write("**Input representation:**", model_info.get("image_type", "Unknown"))
            st.write("**Encoder:**", model_info.get("encoder", "Unknown"))
            st.write("**Classifier:**", model_info.get("classifier", "Unknown"))
            st.write("**Input size:**", model_info.get("input_size", "Unknown"))
            st.write("**Feature dimension:**", model_info.get("feature_dimension", "Unknown"))
            st.write("**Decision threshold:**", model_info.get("threshold", "Unknown"))

# ============================================================
# 10. Image uploader
# ============================================================
st.subheader("Upload OT Image")

uploaded_file = st.file_uploader(
    "Choose a PNG or JPEG OT image",
    type=["png", "jpg", "jpeg"],
)

# ============================================================
# 11. Image preview
# ============================================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded OT image", width=350)
    except Exception:
        st.error("The selected file could not be read as an image.")
        st.stop()

    # ========================================================
    # 12. Prediction button
    # ========================================================
    if st.button("Detect Stiction", type="primary", use_container_width=True):
        if health is None:
            st.error("Prediction cannot be performed because FastAPI is unavailable.")
        else:
            with st.spinner("Running Deep CORAL inference..."):
                try:
                    result = predict_image(uploaded_file)
                except requests.RequestException as error:
                    st.error("Prediction request failed.")
                    st.write(error)
                    st.stop()

            # =================================================
            # 13. Read API result
            # =================================================
            prediction = result["prediction"]
            diagnosis = result["diagnosis"]
            probability = result["stiction_probability"]
            threshold = result["threshold"]
            inference_time = result["inference_time_seconds"]

            # =================================================
            # 14. Display diagnosis
            # =================================================
            st.subheader("Diagnosis")

            if prediction == 1:
                st.error("STICTION DETECTED")
            else:
                st.success("NO STICTION DETECTED")

            # =================================================
            # 15. Display important numerical results
            # =================================================
            column1, column2, column3 = st.columns(3)
            column1.metric("Diagnosis", diagnosis)
            column2.metric("Stiction Probability", f"{probability:.4f}")
            column3.metric("Decision Threshold", f"{threshold:.2f}")

            # =================================================
            # 16. Probability progress bar
            # =================================================
            st.write("### Stiction Probability")
            st.progress(float(probability))

            # =================================================
            # 17. Display inference time
            # =================================================
            st.write("### Inference Information")
            st.write(f"Inference time: {inference_time:.6f} seconds")
            st.write(f"Predicted class: {prediction}")

            # =================================================
            # 18. Show raw response for learning/debugging
            # =================================================
            with st.expander("Raw API Response"):
                st.json(result)

# ============================================================
# 19. Footer
# ============================================================
st.divider()
st.caption("OT Image → Deep CORAL Encoder → Logistic Regression → Stiction Diagnosis")

#FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
#FASTAPI_URL = FASTAPI_URL.rstrip("/")

# one codebase now works in all four environments:
# Local Python
#     ↓
# http://127.0.0.1:8000
#
# Docker Compose
#     ↓
# http://api:8000
#
# Kubernetes
#     ↓
# http://stiction-api-service:8000
#
# Streamlit Community Cloud
#     ↓
# https://stiction-deep-coral-api.onrender.com
