"""Tests for data preprocessing functions."""
import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add DataSet directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DataSet'))

from preprocessing import detect_outliers_iqr, clip_outliers


@pytest.fixture
def sample_df():
    """Fixture providing a small sample dataframe matching insurance.csv schema."""
    return pd.DataFrame({
        'age': [25, 30, 45, 60, 18, 100],  # 100 is an outlier
        'bmi': [22.5, 28.0, 31.5, 35.0, 45.0, 75.0],  # 75 is an outlier
        'children': [0, 1, 2, 3, 0, 1],
        'charges': [5000, 8000, 15000, 25000, 12000, 60000],
    })


def test_detect_outliers_iqr_finds_outliers(sample_df):
    """IQR detection should find the extreme values."""
    lower, upper, count = detect_outliers_iqr(sample_df, 'bmi')
    assert count >= 1, "Should detect at least one BMI outlier"
    assert lower < upper, "Lower bound must be less than upper bound"


def test_detect_outliers_iqr_no_outliers():
    """Perfectly uniform data should have zero outliers."""
    df = pd.DataFrame({'val': [10, 10, 10, 10, 10, 10]})
    lower, upper, count = detect_outliers_iqr(df, 'val')
    assert count == 0, "No outliers expected in uniform data"


def test_clip_outliers_preserves_row_count(sample_df):
    """Clipping must preserve all rows (no deletion)."""
    original_count = len(sample_df)
    clipped = clip_outliers(sample_df, ['bmi'])
    assert len(clipped) == original_count, "Clipping should preserve row count"


def test_clip_outliers_constrains_values(sample_df):
    """After clipping, no values should exceed the IQR bounds."""
    clipped = clip_outliers(sample_df, ['bmi'])
    lower, upper, _ = detect_outliers_iqr(sample_df, 'bmi')
    assert clipped['bmi'].max() <= upper + 0.001, "Max value should not exceed upper bound"
    assert clipped['bmi'].min() >= lower - 0.001, "Min value should not go below lower bound"


def test_clip_outliers_preserves_non_target_columns(sample_df):
    """Clipping BMI should not modify other columns."""
    original_ages = sample_df['age'].tolist()
    clipped = clip_outliers(sample_df, ['bmi'])
    assert clipped['age'].tolist() == original_ages, "Age column should be untouched"


def test_clip_outliers_empty_column_list(sample_df):
    """Passing no columns should return a copy of the original."""
    clipped = clip_outliers(sample_df, [])
    pd.testing.assert_frame_equal(clipped, sample_df)


def test_insurance_csv_structure():
    """The insurance.csv should have expected columns and dtypes."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'DataSet', 'insurance.csv')
    df = pd.read_csv(csv_path)

    required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) == 1338, "Expected exactly 1,338 records"
    assert df.isnull().sum().sum() == 0, "Dataset should have no missing values"
    assert df['smoker'].isin(['yes', 'no']).all(), "Smoker must be yes/no"
    assert df['sex'].isin(['male', 'female']).all(), "Sex must be male/female"
