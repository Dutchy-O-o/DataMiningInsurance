"""Central configuration: paths and preprocessing constants."""
from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

INSURANCE_CSV = DATA_DIR / "insurance.csv"
XGBOOST_MODEL = MODELS_DIR / "xgboost.joblib"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.joblib"

# ---- StandardScaler statistics (fitted on training set) ----
# Hard-coded to avoid a joblib scaler dependency in the web layer.
SCALER = {
    "age":      {"mean": 39.21, "std": 14.04},
    "bmi":      {"mean": 30.65, "std": 6.05},
    "children": {"mean": 1.09,  "std": 1.21},
}

# ---- Display names for UI ----
DISPLAY_NAMES = {
    "age": "Age", "sex": "Sex", "bmi": "BMI", "children": "Children",
    "smoker": "Smoker", "region_northwest": "Region: NW",
    "region_southeast": "Region: SE", "region_southwest": "Region: SW",
    "smoker_bmi": "Smoker x BMI", "smoker_age": "Smoker x Age",
}
