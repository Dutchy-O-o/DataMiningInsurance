"""Tests for model loading and inference sanity checks."""
import sys
import pathlib
import joblib
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, FEATURE_NAMES_FILE


@pytest.fixture(scope="module")
def xgb_model():
    return joblib.load(MODELS_DIR / "xgboost.joblib")


@pytest.fixture(scope="module")
def feature_names():
    return joblib.load(FEATURE_NAMES_FILE)


@pytest.fixture
def blank_row(feature_names):
    data = {f: 0.0 for f in feature_names}
    data["sex"] = 1
    return pd.DataFrame([data], columns=feature_names)


def test_model_has_predict(xgb_model):
    assert hasattr(xgb_model, "predict")


def test_feature_names_has_interactions(feature_names):
    assert len(feature_names) == 10
    assert "smoker_bmi" in feature_names
    assert "smoker_age" in feature_names


def test_prediction_is_reasonable(xgb_model, blank_row):
    pred = xgb_model.predict(blank_row)[0]
    assert 0 < pred < 100000


def test_smoker_costs_more_than_nonsmoker(xgb_model, feature_names):
    base = {f: 0.0 for f in feature_names}
    non_smoker = pd.DataFrame([{**base, "smoker": 0, "smoker_bmi": 0, "smoker_age": 0}], columns=feature_names)
    smoker =     pd.DataFrame([{**base, "smoker": 1, "smoker_bmi": 0, "smoker_age": 0}], columns=feature_names)
    p_ns = xgb_model.predict(non_smoker)[0]
    p_s  = xgb_model.predict(smoker)[0]
    assert p_s > 2 * p_ns


def test_all_models_load():
    models = ["xgboost", "lightgbm", "gradient_boosting", "ridge_regression", "linear_regression"]
    for name in models:
        path = MODELS_DIR / f"{name}.joblib"
        assert path.exists(), f"Missing model file: {name}"
        assert hasattr(joblib.load(path), "predict")


def test_model_is_deterministic(xgb_model, blank_row):
    assert xgb_model.predict(blank_row)[0] == xgb_model.predict(blank_row)[0]


def test_model_accepts_batch(xgb_model, feature_names):
    rows = [{f: 0.0 for f in feature_names} for _ in range(5)]
    preds = xgb_model.predict(pd.DataFrame(rows, columns=feature_names))
    assert len(preds) == 5
