"""
Insurance Claim Costs - Data Preprocessing
===========================================
This script prepares the insurance.csv dataset for machine learning models.

Steps:
1. Load data
2. Outlier detection and clipping (IQR method)
3. Categorical encoding (Label & One-Hot)
4. Continuous variable scaling (StandardScaler)
5. Train/Test split (80/20)
6. Save processed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pathlib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA LOADING
# ============================================================

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, PROCESSED_DIR, INSURANCE_CSV as RAW_CSV  # noqa: E402


def load_data(path=RAW_CSV):
    """Loads raw data and prints basic information."""
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


# ============================================================
# 2. OUTLIER DETECTION AND CLIPPING (IQR)
# ============================================================

def detect_outliers_iqr(df, column):
    """Calculates outlier boundaries using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df[column] < lower) | (df[column] > upper)).sum()
    return lower, upper, outlier_count


def clip_outliers(df, columns):
    """
    Clips outlier values in specified columns using the IQR method.
    Outliers are pulled to lower/upper bounds, not deleted.
    """
    df_clipped = df.copy()
    print("\n--- Outlier Clipping (IQR) ---")
    for col in columns:
        lower, upper, count = detect_outliers_iqr(df_clipped, col)
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
        print(f"  {col:>10s}: {count:3d} outliers clipped "
              f"[{lower:.2f}, {upper:.2f}]")
    return df_clipped


# ============================================================
# 3. CATEGORICAL ENCODING
# ============================================================

def encode_categoricals(df):
    """
    - sex, smoker: Label Encoding (binary - 2 classes)
    - region: One-Hot Encoding (4 classes, drop_first=True)
    """
    df_enc = df.copy()
    print("\n--- Categorical Encoding ---")

    # Label Encoding (binary variables)
    label_mappings = {}
    for col in ["sex", "smoker"]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        label_mappings[col] = mapping
        print(f"  {col}: Label Encoding -> {mapping}")

    # One-Hot Encoding (region)
    df_enc = pd.get_dummies(df_enc, columns=["region"], drop_first=True, dtype=int)
    ohe_cols = [c for c in df_enc.columns if c.startswith("region_")]
    print(f"  region: One-Hot Encoding -> {ohe_cols}")

    return df_enc, label_mappings


# ============================================================
# 4. CONTINUOUS VARIABLE SCALING
# ============================================================

def scale_features(df, target_col="charges"):
    """
    Scales continuous variables (excluding target) with StandardScaler.
    Also returns the scaler object (for test data).
    """
    continuous_cols = ["age", "bmi", "children"]
    scaler = StandardScaler()

    df_scaled = df.copy()
    df_scaled[continuous_cols] = scaler.fit_transform(df_scaled[continuous_cols])

    print("\n--- Scaling (StandardScaler) ---")
    for i, col in enumerate(continuous_cols):
        print(f"  {col:>10s}: mean={scaler.mean_[i]:.2f}, "
              f"std={scaler.scale_[i]:.2f}")

    return df_scaled, scaler, continuous_cols


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(df, target_col="charges", test_size=0.20, random_state=42):
    """Splits data into 80% train, 20% test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n--- Train/Test Split (test={test_size:.0%}) ---")
    print(f"  Train: {X_train.shape[0]} rows")
    print(f"  Test : {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def preprocess_pipeline(save=True):
    """
    Runs all preprocessing steps sequentially.
    If save=True, saves processed data as CSV files.
    """
    # 1) Load
    df = load_data()

    # 2) Clip outliers (continuous features only)
    numeric_cols = ["age", "bmi", "children"]
    df = clip_outliers(df, numeric_cols)

    # 3) Categorical encoding
    df, label_mappings = encode_categoricals(df)

    # 4) Scaling
    df, scaler, continuous_cols = scale_features(df)

    # 5) Train/Test split
    X_train, X_test, y_train, y_test = split_data(df)

    # 6) Save
    if save:
        output_dir = PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)
        df.to_csv(output_dir / "insurance_preprocessed.csv", index=False)

        print(f"\n--- Files Saved ---")
        print(f"  Location: {output_dir}")
        for f in sorted(output_dir.glob("*.csv")):
            print(f"  -> {f.name}")

    print("\n--- Final Data Structure ---")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  Target: charges")
    print(f"  Total features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, label_mappings


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, mappings = preprocess_pipeline()
