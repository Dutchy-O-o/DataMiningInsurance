"""KNN-based 'similar patients' module, stratified by smoker status.

Design:
  - Training data is split into smokers / non-smokers at import time.
  - Two separate KNN models are fitted — queries are always routed to the
    user's own smoker group, guaranteeing an exact match on the most
    important feature.
  - Within each group, features use weighted Euclidean distance so that
    BMI and age dominate similarity (matching their predictive power).
  - Similarity % uses exp(-d / scale) normalised by the group's 95th-percentile
    distance, giving a realistic 0-100% score.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import PROCESSED_DIR, INSURANCE_CSV, SCALER
from webapp.features import scale

# Feature order used internally by the KNN distance metric.
KNN_FEATURES = [
    "age", "bmi", "children", "sex",
    "region_northwest", "region_southeast", "region_southwest",
]
# Weights chosen to reflect XGBoost feature importance:
#   BMI (×2.5) and age (×2.0) carry the most weight.
#   Children / sex / region are secondary.
KNN_WEIGHTS = np.array([2.0, 2.5, 0.8, 0.5, 0.4, 0.4, 0.4])


def _weighted(df):
    return df[KNN_FEATURES].values * np.sqrt(KNN_WEIGHTS)


def _reconstruct_raw(row):
    """Decode a preprocessed feature row back to human-readable values."""
    age = round(row["age"] * SCALER["age"]["std"] + SCALER["age"]["mean"])
    bmi = round(row["bmi"] * SCALER["bmi"]["std"] + SCALER["bmi"]["mean"], 1)
    children = round(row["children"] * SCALER["children"]["std"] + SCALER["children"]["mean"])
    if row.get("region_northwest", 0) == 1:
        region = "northwest"
    elif row.get("region_southeast", 0) == 1:
        region = "southeast"
    elif row.get("region_southwest", 0) == 1:
        region = "southwest"
    else:
        region = "northeast"
    return {
        "age": int(age), "sex": "male" if row["sex"] == 1 else "female",
        "bmi": float(bmi), "children": int(children),
        "smoker": "yes" if row["smoker"] == 1 else "no",
        "region": region,
    }


def _estimate_scale(knn_model, X):
    if len(X) < 2:
        return 10.0
    sample = X[:min(200, len(X))]
    d, _ = knn_model.kneighbors(sample, n_neighbors=min(5, len(X)))
    return float(np.percentile(d[:, -1], 95)) or 1.0


# ---- Fit two KNN models at module import ----
KNN_SMOKER = KNN_NONSMOKER = None
TRAIN_RAW_SMOKER = TRAIN_RAW_NONSMOKER = None
TRAIN_Y_SMOKER = TRAIN_Y_NONSMOKER = None
SCALE_SMOKER = SCALE_NONSMOKER = 1.0

try:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    is_smoker = X_train["smoker"] == 1
    smoker_rows = X_train[is_smoker].reset_index(drop=True)
    nonsmoker_rows = X_train[~is_smoker].reset_index(drop=True)

    KNN_SMOKER = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(_weighted(smoker_rows))
    KNN_NONSMOKER = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(_weighted(nonsmoker_rows))

    TRAIN_RAW_SMOKER = pd.DataFrame([_reconstruct_raw(r) for _, r in smoker_rows.iterrows()])
    TRAIN_RAW_NONSMOKER = pd.DataFrame([_reconstruct_raw(r) for _, r in nonsmoker_rows.iterrows()])

    TRAIN_Y_SMOKER = y_train.values[is_smoker.values]
    TRAIN_Y_NONSMOKER = y_train.values[~is_smoker.values]

    SCALE_SMOKER = _estimate_scale(KNN_SMOKER, _weighted(smoker_rows))
    SCALE_NONSMOKER = _estimate_scale(KNN_NONSMOKER, _weighted(nonsmoker_rows))
    print(f"KNN ready: {len(smoker_rows)} smokers, {len(nonsmoker_rows)} non-smokers")
except Exception as e:
    print(f"KNN init failed: {e}")


def find_similar_patients(age, sex, bmi, children, smoker, region):
    """Return 5 most similar training patients within the user's smoker group."""
    if smoker == "yes":
        model, raw, y_vals, dist_scale = KNN_SMOKER, TRAIN_RAW_SMOKER, TRAIN_Y_SMOKER, SCALE_SMOKER
    else:
        model, raw, y_vals, dist_scale = KNN_NONSMOKER, TRAIN_RAW_NONSMOKER, TRAIN_Y_NONSMOKER, SCALE_NONSMOKER

    if model is None or raw is None or len(raw) == 0:
        return None

    vec = np.array([[
        scale(age, "age"),
        scale(bmi, "bmi"),
        scale(children, "children"),
        1 if sex == "male" else 0,
        1 if region == "northwest" else 0,
        1 if region == "southeast" else 0,
        1 if region == "southwest" else 0,
    ]]) * np.sqrt(KNN_WEIGHTS)

    distances, indices = model.kneighbors(vec)
    out = []
    for dist, idx in zip(distances[0], indices[0]):
        row = raw.iloc[int(idx)]
        sim_pct = max(5.0, min(100.0, 100.0 * np.exp(-float(dist) / max(dist_scale, 0.1))))
        out.append({
            "age": int(row["age"]),
            "sex": row["sex"],
            "bmi": round(float(row["bmi"]), 1),
            "children": int(row["children"]),
            "smoker": row["smoker"],
            "region": row["region"],
            "actual_charge": round(float(y_vals[int(idx)])),
            "similarity": round(float(sim_pct), 1),
            "distance": round(float(dist), 3),
        })
    return out


def similar_with_summary(age, sex, bmi, children, smoker, region):
    """Wrapper that returns {patients, summary} or None."""
    patients = find_similar_patients(age, sex, bmi, children, smoker, region)
    if not patients:
        return None
    charges = [p["actual_charge"] for p in patients]
    return {
        "patients": patients,
        "summary": {
            "count":  len(patients),
            "min":    min(charges),
            "max":    max(charges),
            "mean":   round(float(np.mean(charges))),
            "median": round(float(np.median(charges))),
        },
    }
