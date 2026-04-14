"""Integration tests for Flask web app endpoints."""
import sys
import os
import json
from io import BytesIO
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DataSet'))


@pytest.fixture(scope='module')
def client():
    """Create a Flask test client."""
    # Must change to DataSet directory so app.py finds its relative paths
    original_cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'DataSet'))
    try:
        import app as app_module
        # Reload in case of cached state
        app_module.app.config['TESTING'] = True
        with app_module.app.test_client() as c:
            yield c
    finally:
        os.chdir(original_cwd)


def test_api_stats_endpoint(client):
    """GET /api/stats should return dataset and model stats."""
    resp = client.get('/api/stats')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'dataset' in data
    assert 'models' in data
    assert data['dataset']['total_rows'] == 1338


def test_predict_endpoint_valid_input(client):
    """POST /predict with valid payload should return a prediction."""
    payload = {
        'age': 35, 'sex': 'male', 'bmi': 27.5,
        'children': 1, 'smoker': 'no', 'region': 'southeast',
    }
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'cost' in data
    assert data['cost'] > 0
    assert 'shap_bars' in data
    assert len(data['shap_bars']) > 0


def test_predict_endpoint_smoker_higher_cost(client):
    """Smoker should get higher predicted cost than non-smoker."""
    base = {'age': 45, 'sex': 'female', 'bmi': 30.0, 'children': 2, 'region': 'northwest'}
    non_smoker = {**base, 'smoker': 'no'}
    smoker = {**base, 'smoker': 'yes'}

    r1 = client.post('/predict', json=non_smoker).get_json()
    r2 = client.post('/predict', json=smoker).get_json()

    assert r2['cost'] > r1['cost'], "Smoker should cost more than non-smoker"
    assert r2['cost'] > 2 * r1['cost'], "Smoker cost should be significantly higher"


def test_batch_predict_valid_csv(client):
    """Batch endpoint should process a valid CSV."""
    csv_data = b"age,sex,bmi,children,smoker,region\n30,male,26.5,0,no,southeast\n45,female,32.1,2,yes,northwest\n"
    resp = client.post('/batch_predict', data={'file': (BytesIO(csv_data), 'test.csv')},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['summary']['total'] == 2
    assert data['summary']['successful'] == 2


def test_batch_predict_missing_columns(client):
    """Batch endpoint should error on missing columns."""
    csv_data = b"age,bmi\n30,26.5\n"  # missing required columns
    resp = client.post('/batch_predict', data={'file': (BytesIO(csv_data), 'test.csv')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'Missing columns' in resp.get_json()['error']


def test_batch_predict_no_file(client):
    """Batch endpoint without file should return 400."""
    resp = client.post('/batch_predict')
    assert resp.status_code == 400


def test_similar_endpoint(client):
    """Similar patients endpoint should return 5 matches with summary."""
    payload = {
        'age': 40, 'sex': 'male', 'bmi': 29.0,
        'children': 2, 'smoker': 'no', 'region': 'southeast',
    }
    resp = client.post('/similar', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'similar' in data
    assert len(data['similar']) == 5
    assert 'summary' in data
    assert data['summary']['count'] == 5
    # First match should be highly similar
    assert data['similar'][0]['similarity'] >= 50


def test_index_page_loads(client):
    """The main / route should return HTML."""
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'<html' in resp.data.lower()
    assert b'insurance' in resp.data.lower()
