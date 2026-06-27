from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Advanced Testing Pipeline API is running"


def test_predict_valid_input():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_missing_field():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_datatype():
    payload = {
        "sepal_length": "wrong",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_value():
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422