import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Locate the FastAPI application
API_DIR = Path(__file__).resolve().parents[1] / "src" / "18_api"
sys.path.insert(0, str(API_DIR))

from importlib import import_module

app = import_module("01_fastapi_app").app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
def test_diabetes_invalid_input():
    response = client.post(
        "/predict/diabetes",
        json={}
    )

    assert response.status_code == 422