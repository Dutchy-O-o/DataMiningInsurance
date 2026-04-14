"""Integration tests for Flask endpoints."""
import sys
import pathlib
from io import BytesIO
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    from webapp import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_stats(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["dataset"]["total_rows"] == 1338
    assert len(data["models"]) == 5


def test_predict_valid_input(client):
    payload = {"age": 35, "sex": "male", "bmi": 27.5,
               "children": 1, "smoker": "no", "region": "southeast"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cost"] > 0
    assert len(data["shap_bars"]) > 0


def test_predict_smoker_costs_more(client):
    base = {"age": 45, "sex": "female", "bmi": 30.0, "children": 2, "region": "northwest"}
    ns = client.post("/predict", json={**base, "smoker": "no"}).get_json()
    sm = client.post("/predict", json={**base, "smoker": "yes"}).get_json()
    assert sm["cost"] > 2 * ns["cost"]


def test_batch_predict_valid_csv(client):
    csv = b"age,sex,bmi,children,smoker,region\n30,male,26.5,0,no,southeast\n45,female,32.1,2,yes,northwest\n"
    resp = client.post(
        "/batch_predict",
        data={"file": (BytesIO(csv), "test.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["summary"]["successful"] == 2


def test_batch_predict_missing_columns(client):
    csv = b"age,bmi\n30,26.5\n"
    resp = client.post(
        "/batch_predict",
        data={"file": (BytesIO(csv), "test.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "Missing columns" in resp.get_json()["error"]


def test_batch_predict_no_file(client):
    resp = client.post("/batch_predict")
    assert resp.status_code == 400


def test_similar_endpoint_returns_five(client):
    payload = {"age": 40, "sex": "male", "bmi": 29.0,
               "children": 2, "smoker": "no", "region": "southeast"}
    resp = client.post("/similar", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["similar"]) == 5
    assert data["summary"]["count"] == 5


def test_similar_patients_match_smoker_status(client):
    """All 5 returned patients must share the query's smoker status."""
    payload = {"age": 30, "sex": "male", "bmi": 25.0,
               "children": 1, "smoker": "no", "region": "northwest"}
    data = client.post("/similar", json=payload).get_json()
    assert all(p["smoker"] == "no" for p in data["similar"])


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()
