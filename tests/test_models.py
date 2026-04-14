"""Tests for model inference and sanity checks."""
import sys
import os
import numpy as np
import pandas as pd
import joblib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DataSet'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'DataSet', 'saved_models')


@pytest.fixture(scope='module')
def xgb_model():
    """Load XGBoost model once per module."""
    return joblib.load(os.path.join(MODEL_DIR, 'xgboost.joblib'))


@pytest.fixture(scope='module')
def feature_names():
    return joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))


@pytest.fixture
def valid_feature_row(feature_names):
    """A plausible input feature row with interaction features."""
    data = {
        'age': 0.0, 'sex': 1, 'bmi': 0.0, 'children': 0.0, 'smoker': 0,
        'region_northwest': 0, 'region_southeast': 0, 'region_southwest': 0,
        'smoker_bmi': 0.0, 'smoker_age': 0.0,
    }
    return pd.DataFrame([data], columns=feature_names)


def test_xgboost_model_loads(xgb_model):
    """Model should load successfully."""
    assert xgb_model is not None
    assert hasattr(xgb_model, 'predict')


def test_feature_names_match(feature_names):
    """Feature names file must have the expected 10 features."""
    assert len(feature_names) == 10
    assert 'smoker_bmi' in feature_names
    assert 'smoker_age' in feature_names


def test_prediction_is_positive(xgb_model, valid_feature_row):
    """Prediction should be a reasonable positive number."""
    pred = xgb_model.predict(valid_feature_row)[0]
    assert pred > 0, f"Prediction should be positive, got {pred}"
    assert pred < 100000, f"Prediction should be under $100K for avg profile, got {pred}"


def test_smoker_increases_prediction(xgb_model, feature_names):
    """A smoker should have a higher predicted cost than a non-smoker, all else equal."""
    base = {f: 0.0 for f in feature_names}
    non_smoker = base.copy(); non_smoker['smoker'] = 0; non_smoker['smoker_bmi'] = 0; non_smoker['smoker_age'] = 0
    smoker = base.copy(); smoker['smoker'] = 1; smoker['smoker_bmi'] = 0; smoker['smoker_age'] = 0

    p_ns = xgb_model.predict(pd.DataFrame([non_smoker], columns=feature_names))[0]
    p_s = xgb_model.predict(pd.DataFrame([smoker], columns=feature_names))[0]

    assert p_s > p_ns, f"Smoker ({p_s}) should cost more than non-smoker ({p_ns})"
    # Based on EDA, smokers pay ~3.8x more on average
    assert p_s > 2 * p_ns, "Smoker cost should be meaningfully higher (at least 2x)"


def test_all_models_load():
    """All 5 models should load without error."""
    models = ['xgboost', 'lightgbm', 'gradient_boosting', 'ridge_regression', 'linear_regression']
    for m in models:
        path = os.path.join(MODEL_DIR, f'{m}.joblib')
        assert os.path.exists(path), f"Model file missing: {m}"
        loaded = joblib.load(path)
        assert hasattr(loaded, 'predict'), f"Model {m} has no predict method"


def test_model_reproducibility(xgb_model, valid_feature_row):
    """The same input should always produce the same prediction."""
    p1 = xgb_model.predict(valid_feature_row)[0]
    p2 = xgb_model.predict(valid_feature_row)[0]
    assert p1 == p2, "Model should be deterministic"


def test_batch_prediction(xgb_model, feature_names):
    """Model should handle multiple rows at once."""
    rows = [{f: 0.0 for f in feature_names} for _ in range(5)]
    df = pd.DataFrame(rows, columns=feature_names)
    preds = xgb_model.predict(df)
    assert len(preds) == 5, "Batch prediction should return 5 values"
    assert all(p >= 0 or p > -100 for p in preds), "All predictions should be reasonable"
