"""Welcome-dashboard stats + model leaderboard, precomputed at startup."""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

from src.config import INSURANCE_CSV, MODELS_DIR, PROCESSED_DIR

DATASET_STATS = {}
MODEL_SCORES = []

# ---- Raw dataset stats ----
try:
    raw = pd.read_csv(INSURANCE_CSV)
    DATASET_STATS = {
        "total_rows":    int(len(raw)),
        "avg_age":       round(float(raw["age"].mean()), 1),
        "avg_bmi":       round(float(raw["bmi"].mean()), 1),
        "avg_cost":      round(float(raw["charges"].mean())),
        "max_cost":      round(float(raw["charges"].max())),
        "min_cost":      round(float(raw["charges"].min())),
        "smoker_pct":    round(float((raw["smoker"] == "yes").mean() * 100), 1),
        "male_pct":      round(float((raw["sex"] == "male").mean() * 100), 1),
        "smoker_avg":    round(float(raw[raw["smoker"] == "yes"]["charges"].mean())),
        "nonsmoker_avg": round(float(raw[raw["smoker"] == "no"]["charges"].mean())),
    }
except Exception as e:
    print(f"Dataset stats failed: {e}")


# ---- 5-model leaderboard (recomputed on test set at startup) ----
try:
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    # Interaction features weren't saved in preprocessed CSV — add them here.
    X_test["smoker_bmi"] = X_test["smoker"] * X_test["bmi"]
    X_test["smoker_age"] = X_test["smoker"] * X_test["age"]

    models = [
        ("XGBoost",           "xgboost",           False),
        ("LightGBM",          "lightgbm",          False),
        ("GradientBoosting",  "gradient_boosting", False),
        ("Ridge Regression",  "ridge_regression",  True),
        ("Linear Regression", "linear_regression", True),
    ]
    for display_name, file_name, uses_log in models:
        m = joblib.load(MODELS_DIR / f"{file_name}.joblib")
        preds = m.predict(X_test)
        if uses_log:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        MODEL_SCORES.append({
            "name":    display_name,
            "r2":      round(float(r2_score(y_test, preds)), 4),
            "mae":     round(float(mean_absolute_error(y_test, preds))),
            "rmse":    round(float(np.sqrt(np.mean((y_test - preds) ** 2)))),
            "is_best": file_name == "xgboost",
        })
    MODEL_SCORES.sort(key=lambda x: x["r2"], reverse=True)
except Exception as e:
    print(f"Model leaderboard failed: {e}")
