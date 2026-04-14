"""
Insurance Claim Costs - Model Training V2.0
=============================================
Enhancements over V1:
  - Feature Engineering: smoker*bmi and smoker*age interaction features
  - Log Transform: applied ONLY to Linear/Ridge (fixes skewness for linear models)
  - Boosting models train on raw target (trees handle skewness natively)
  - Expanded hyperparameter search (30 iterations)
"""

import pandas as pd
import numpy as np
import pathlib
import time
import joblib
import warnings

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR as MODEL_DIR  # noqa: E402
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATA LOADING + FEATURE ENGINEERING
# ============================================================

def load_and_engineer(verbose=True):
    """
    Loads preprocessed data and adds interaction features:
      - smoker_bmi:  smoker * bmi  (captures high-cost smoker+obese cluster)
      - smoker_age:  smoker * age  (captures age amplification for smokers)
    """
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    for df in [X_train, X_test]:
        df["smoker_bmi"] = df["smoker"] * df["bmi"]
        df["smoker_age"] = df["smoker"] * df["age"]

    if verbose:
        print(f"Data loaded -> Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Features: {list(X_train.columns)}")
        print(f"New interaction features: smoker_bmi, smoker_age")

    return X_train, X_test, y_train, y_test


# ============================================================
# 2. EVALUATION
# ============================================================

def evaluate_model(name, model, X_test, y_test, log_target=False):
    """Evaluates model. If log_target, applies expm1 to predictions."""
    y_pred = model.predict(X_test)
    if log_target:
        y_pred = np.expm1(y_pred)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ============================================================
# 3. BASE MODELS (with log target - fixes skewness for linear)
# ============================================================

def train_linear(X_train, y_train_log):
    model = LinearRegression()
    model.fit(X_train, y_train_log)
    return model


def train_ridge(X_train, y_train_log):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train_log)
    return model


# ============================================================
# 4. BOOSTING MODELS (raw target - trees handle skewness)
# ============================================================

COMMON_PARAM_GRID = {
    "n_estimators": [200, 300, 500, 800, 1000],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
}


def tune_xgboost(X_train, y_train):
    param_grid = {
        **COMMON_PARAM_GRID,
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 1.5, 2, 3],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_distributions=param_grid,
        n_iter=40,
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(X_train, y_train):
    param_grid = {
        **COMMON_PARAM_GRID,
        "num_leaves": [15, 31, 50, 70, 100],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 1.5, 2, 3],
    }
    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        param_distributions=param_grid,
        n_iter=40,
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_gradient_boosting(X_train, y_train):
    param_grid = {
        **COMMON_PARAM_GRID,
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=40,
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


# ============================================================
# 5. SAVE
# ============================================================

def save_model(obj, name):
    filepath = MODEL_DIR / f"{name}.joblib"
    joblib.dump(obj, filepath)
    return filepath


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main():
    X_train, X_test, y_train, y_test = load_and_engineer()

    # Log target for linear models only
    y_train_log = np.log1p(y_train)

    print(f"\nStrategy:")
    print(f"  Linear/Ridge  -> log1p(charges) target (skew {y_train.skew():.2f} -> {y_train_log.skew():.2f})")
    print(f"  Boosting      -> raw charges target (trees handle skewness natively)")

    save_model(list(X_train.columns), "feature_names")

    results = []
    models = {}

    # --- Linear Models (log target) ---
    print("\n" + "=" * 60)
    print("LINEAR MODELS (log1p target + interaction features)")
    print("=" * 60)

    print("\n[1/5] Training Linear Regression...")
    lr = train_linear(X_train, y_train_log)
    models["linear_regression"] = lr
    results.append(evaluate_model("Linear Regression", lr, X_test, y_test, log_target=True))
    print(f"      R2: {results[-1]['R2']:.4f} | RMSE: {results[-1]['RMSE']:.2f}")

    print("[2/5] Training Ridge Regression...")
    ridge = train_ridge(X_train, y_train_log)
    models["ridge_regression"] = ridge
    results.append(evaluate_model("Ridge Regression", ridge, X_test, y_test, log_target=True))
    print(f"      R2: {results[-1]['R2']:.4f} | RMSE: {results[-1]['RMSE']:.2f}")

    # --- Boosting Models (raw target + interactions) ---
    print("\n" + "=" * 60)
    print("BOOSTING MODELS (raw target + interactions, 5-Fold CV, 40 iter)")
    print("=" * 60)

    print("\n[3/5] XGBoost - Hyperparameter search started...")
    t0 = time.time()
    xgb_best, xgb_params, xgb_cv = tune_xgboost(X_train, y_train)
    models["xgboost"] = xgb_best
    results.append(evaluate_model("XGBoost", xgb_best, X_test, y_test))
    print(f"      Time: {time.time() - t0:.1f}s | CV R2: {xgb_cv:.4f}")
    print(f"      Test R2: {results[-1]['R2']:.4f} | RMSE: {results[-1]['RMSE']:.2f}")
    print(f"      Best params: {xgb_params}")

    print("\n[4/5] LightGBM - Hyperparameter search started...")
    t0 = time.time()
    lgbm_best, lgbm_params, lgbm_cv = tune_lightgbm(X_train, y_train)
    models["lightgbm"] = lgbm_best
    results.append(evaluate_model("LightGBM", lgbm_best, X_test, y_test))
    print(f"      Time: {time.time() - t0:.1f}s | CV R2: {lgbm_cv:.4f}")
    print(f"      Test R2: {results[-1]['R2']:.4f} | RMSE: {results[-1]['RMSE']:.2f}")
    print(f"      Best params: {lgbm_params}")

    print("\n[5/5] GradientBoosting - Hyperparameter search started...")
    t0 = time.time()
    gb_best, gb_params, gb_cv = tune_gradient_boosting(X_train, y_train)
    models["gradient_boosting"] = gb_best
    results.append(evaluate_model("GradientBoosting", gb_best, X_test, y_test))
    print(f"      Time: {time.time() - t0:.1f}s | CV R2: {gb_cv:.4f}")
    print(f"      Test R2: {results[-1]['R2']:.4f} | RMSE: {results[-1]['RMSE']:.2f}")
    print(f"      Best params: {gb_params}")

    # --- Results ---
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE (V2.0)")
    print("=" * 60)
    df_results = pd.DataFrame(results).sort_values("R2", ascending=False)
    df_results.index = range(1, len(df_results) + 1)
    df_results.index.name = "Rank"
    print(df_results.to_string())

    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    for name, model in models.items():
        path = save_model(model, name)
        print(f"  -> {path.name}")
    print(f"\nLocation: {MODEL_DIR}")

    best = df_results.iloc[0]
    print(f"\nBEST MODEL: {best['Model']} "
          f"(R2={best['R2']:.4f}, RMSE={best['RMSE']:.2f})")


if __name__ == "__main__":
    main()
