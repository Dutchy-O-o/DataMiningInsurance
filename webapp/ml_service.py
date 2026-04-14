"""Core ML prediction, SHAP explanations, and counterfactual scenarios."""
import numpy as np
import joblib
import shap

from src.config import MODELS_DIR, XGBOOST_MODEL, DISPLAY_NAMES
from webapp.features import FEATURE_NAMES, build_features

# Primary model + SHAP explainer loaded once at import time.
model = joblib.load(XGBOOST_MODEL)
explainer = shap.TreeExplainer(model)

# Ensemble models used for confidence intervals.
_ENSEMBLE_NAMES = ["xgboost", "lightgbm", "gradient_boosting"]
_ensemble_cache = {}


def _load_ensemble(name):
    if name not in _ensemble_cache:
        _ensemble_cache[name] = joblib.load(MODELS_DIR / f"{name}.joblib")
    return _ensemble_cache[name]


def predict_cost(age, sex, bmi, children, smoker, region) -> float:
    """Single-model XGBoost prediction, clipped to non-negative."""
    X = build_features(age, sex, bmi, children, smoker, region)
    return max(float(model.predict(X)[0]), 0)


def predict_confidence(age, sex, bmi, children, smoker, region):
    """Return low/mid/high from 3 boosting models. None if <2 models load."""
    preds = []
    for name in _ENSEMBLE_NAMES:
        try:
            m = _load_ensemble(name)
            X = build_features(age, sex, bmi, children, smoker, region)
            preds.append(max(float(m.predict(X)[0]), 0))
        except Exception:
            continue
    if len(preds) >= 2:
        return {
            "low":  round(min(preds)),
            "mid":  round(float(np.mean(preds))),
            "high": round(max(preds)),
        }
    return None


def compute_shap(age, sex, bmi, children, smoker, region):
    """Return (shap_bars, top_features) for the current prediction."""
    X = build_features(age, sex, bmi, children, smoker, region)
    values = explainer(X).values[0]
    order = np.argsort(np.abs(values))[::-1]
    shap_bars = [
        {"name": DISPLAY_NAMES.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
         "value": round(float(values[i]), 2)}
        for i in order
    ]
    top = []
    for item in shap_bars[:2]:
        top.append({
            "name": item["name"],
            "direction": "increases" if item["value"] > 0 else "decreases",
            "impact": abs(item["value"]),
        })
    return shap_bars, top


def what_if_scenarios(age, sex, bmi, children, smoker, region, current_cost):
    """Compute counterfactual savings for 'quit smoking' and 'reach BMI 25'."""
    scenarios = {}
    if smoker == "yes":
        new_cost = predict_cost(age, sex, bmi, children, "no", region)
        savings = current_cost - new_cost
        if savings > 0:
            scenarios["quit_smoking"] = {
                "current": round(current_cost),
                "new":     round(new_cost),
                "savings": round(savings),
            }
    if bmi > 25:
        new_cost = predict_cost(age, sex, 25.0, children, smoker, region)
        savings = current_cost - new_cost
        if savings > 0:
            scenarios["healthy_bmi"] = {
                "current": round(current_cost),
                "new":     round(new_cost),
                "savings": round(savings),
            }
    return scenarios
