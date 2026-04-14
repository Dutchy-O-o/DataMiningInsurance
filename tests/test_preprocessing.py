"""Tests for data preprocessing functions."""
import sys
import pathlib
import pandas as pd
import pytest

# Repo root on sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import detect_outliers_iqr, clip_outliers
from src.config import INSURANCE_CSV


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":      [25, 30, 45, 60, 18, 100],
        "bmi":      [22.5, 28.0, 31.5, 35.0, 45.0, 75.0],
        "children": [0, 1, 2, 3, 0, 1],
        "charges":  [5000, 8000, 15000, 25000, 12000, 60000],
    })


def test_detect_outliers_finds_extreme_values(sample_df):
    _, _, count = detect_outliers_iqr(sample_df, "bmi")
    assert count >= 1


def test_detect_outliers_uniform_data():
    df = pd.DataFrame({"val": [10] * 6})
    _, _, count = detect_outliers_iqr(df, "val")
    assert count == 0


def test_clip_outliers_preserves_row_count(sample_df):
    assert len(clip_outliers(sample_df, ["bmi"])) == len(sample_df)


def test_clip_outliers_constrains_values(sample_df):
    clipped = clip_outliers(sample_df, ["bmi"])
    lower, upper, _ = detect_outliers_iqr(sample_df, "bmi")
    assert clipped["bmi"].max() <= upper + 0.001
    assert clipped["bmi"].min() >= lower - 0.001


def test_clip_outliers_leaves_other_columns_alone(sample_df):
    ages = sample_df["age"].tolist()
    assert clip_outliers(sample_df, ["bmi"])["age"].tolist() == ages


def test_clip_outliers_empty_column_list(sample_df):
    pd.testing.assert_frame_equal(clip_outliers(sample_df, []), sample_df)


def test_insurance_csv_structure():
    df = pd.read_csv(INSURANCE_CSV)
    for col in ["age", "sex", "bmi", "children", "smoker", "region", "charges"]:
        assert col in df.columns
    assert len(df) == 1338
    assert df.isnull().sum().sum() == 0
    assert df["smoker"].isin(["yes", "no"]).all()
    assert df["sex"].isin(["male", "female"]).all()
