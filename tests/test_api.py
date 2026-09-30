# this files verifies if the endpoint is working
from io import BytesIO
from PIL import Image
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
test_image = "CHEM13.png"

def make_png_bytes():
    image = Image.new("RGB", (80, 80), color = (90, 120, 180))
    buffer = BytesIO()
    image.save(buffer, format = "PNG")
    buffer.seek(0)

    return buffer.getvalue()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200 # 200 OK is the standard status code for a successful HTTP GET request.
    '''
    200 OK: Standard response for successful GET, PUT, or POST requests fetching or updating data.

    201 Created: Typically returned when a new resource is successfully created (usually via POST).

    404 Not Found: Returned if the route / isn't defined in your FastAPI/Flask app.

    500 Internal Server Error: Returned if an unhandled exception occurs inside the route function.
    '''

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200

    data = response.json()
    expected_keys = {"model_name",
                     "image_type",
                     "encoder",
                     "classifier",
                     "input_size"}
    assert expected_keys.issubset(data.keys())

def test_predict_endpoint_with_valid_png():
    response = client.post("/predict", files={"file": (test_image, make_png_bytes(), "image/png",)},)
    assert response.status_code == 200

def test_predict_response_contains_required_fields():
    response = client.post("/predict",files={"file": (test_image,make_png_bytes(),"image/png",)},)
    data = response.json()

    expected_keys = {
        "prediction",
        "diagnosis",
        "stiction_probability",
        "threshold",
        "inference_time_seconds",}

    assert expected_keys.issubset(data.keys())

def test_predict_response_probability_range():
    response = client.post("/predict",files={"file": (test_image,make_png_bytes(),"image/png",)},)
    probability = response.json()["stiction_probability"]
    assert 0.0 <= probability <= 1.0

def test_predict_response_prediction_is_binary():
    response = client.post("/predict",files={"file": (test_image,make_png_bytes(),"image/png",)},)
    prediction = response.json()["prediction"]
    assert prediction in (0, 1)

def test_predict_response_diagnosis_is_valid():
    response = client.post("/predict",files={"file": (test_image,make_png_bytes(),"image/png",)},)
    diagnosis = response.json()["diagnosis"]
    assert diagnosis in {"Stiction", "No Stiction",}

def test_predict_rejects_non_image_file():
    response = client.post("/predict",files={"file": ("not_an_image.txt",b"this is not an image","text/plain",)},)
    assert response.status_code == 400

def test_prediction_matches_threshold_logic():
    response = client.post("/predict",files={"file": (test_image,make_png_bytes(),"image/png",)},)
    data = response.json()
    expected_prediction = int(data["stiction_probability"]>= data["threshold"])
    assert data["prediction"] == expected_prediction