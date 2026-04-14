"""Feature scaling and engineering — mirrors the training-time pipeline."""
import joblib
import pandas as pd

from src.config import SCALER, FEATURE_NAMES_FILE

FEATURE_NAMES = joblib.load(FEATURE_NAMES_FILE)


def scale(value: float, feature: str) -> float:
    """Apply StandardScaler transform using fixed train-time statistics."""
    return (value - SCALER[feature]["mean"]) / SCALER[feature]["std"]


def build_features(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Turn raw user input into the 10-column feature row the model expects.

    Columns: age, sex, bmi, children, smoker, region_northwest, region_southeast,
             region_southwest, smoker_bmi, smoker_age
    Interaction features use SCALED age/bmi to match training.
    """
    age_s = scale(age, "age")
    bmi_s = scale(bmi, "bmi")
    children_s = scale(children, "children")
    sex_enc = 1 if sex == "male" else 0
    smoker_enc = 1 if smoker == "yes" else 0
    row = {
        "age": age_s,
        "sex": sex_enc,
        "bmi": bmi_s,
        "children": children_s,
        "smoker": smoker_enc,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "smoker_bmi": smoker_enc * bmi_s,
        "smoker_age": smoker_enc * age_s,
    }
    return pd.DataFrame([row], columns=FEATURE_NAMES)
