import io
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Kedro Road Sign API" in response.data

def test_run_pipeline_no_file(client):
    response = client.post("/run-pipeline", data={})
    assert response.status_code == 400
    assert b"No file part in request" in response.data

def test_run_pipeline_empty_file(client):
    data = {
        "file": (io.BytesIO(b""), ""),
        "pipeline_name": "predict"
    }
    response = client.post("/run-pipeline", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No selected file" in response.data

def test_run_pipeline_missing_pipeline_name(client):
    data = {
        "file": (io.BytesIO(b"fake image"), "test.jpg")
    }
    response = client.post("/run-pipeline", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Missing pipeline_name" in response.data

def test_run_pipeline_file_type_not_allowed(client):
    data = {
        "file": (io.BytesIO(b"fake image"), "test.txt"),
        "pipeline_name": "predict"
    }
    response = client.post("/run-pipeline", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"File type not allowed" in response.data