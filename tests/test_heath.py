"""
Tests for GET /health.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_response_is_json():
    response = client.get("/health")
    data = response.json()
    assert isinstance(data, dict)

def test_health_response_contains_status():
    response = client.get("/health")
    data = response.json()
    assert "status" in data

def test_health_status_is_healthy():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"

def test_health_reports_model_loaded():
    response = client.get("/health")
    data = response.json()

    assert "model_loaded" in data
    assert data["model_loaded"] is True
