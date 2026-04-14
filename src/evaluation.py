"""
Insurance Claim Costs - Model Evaluation
==========================================
Evaluates the best model (XGBoost) on the test set:
  1. Test set prediction
  2. Feature Importance chart
  3. Actual vs Predicted scatter plot
  4. Residuals distribution histogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pathlib

sns.set_theme(style="whitegrid")

# ============================================================
# PATHS
# ============================================================

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR as MODEL_DIR, RESULTS_DIR  # noqa: E402
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD DATA AND MODEL
# ============================================================

print("=" * 60)
print("MODEL EVALUATION - XGBoost")
print("=" * 60)

X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
model = joblib.load(MODEL_DIR / "xgboost.joblib")

print(f"\nTest data: {X_test.shape[0]} rows, {X_test.shape[1]} features")
print(f"Model loaded: {MODEL_DIR / 'xgboost.joblib'}")

# ============================================================
# 2. PREDICTION
# ============================================================

y_pred = model.predict(X_test)
residuals = y_test - y_pred

mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))
r2 = 1 - np.sum(residuals ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f"\n--- Test Metrics ---")
print(f"  MAE  : {mae:>10.2f} $")
print(f"  RMSE : {rmse:>10.2f} $")
print(f"  R2   : {r2:>10.4f}")

# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

importances = model.feature_importances_
feature_names = X_test.columns.tolist()
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)
fi_df["Percentage"] = (fi_df["Importance"] / fi_df["Importance"].sum()) * 100

print(f"\n--- Feature Importance Ranking ---")
for _, row in fi_df.sort_values("Percentage", ascending=False).iterrows():
    bar = "#" * int(row["Percentage"] / 2)
    print(f"  {row['Feature']:>20s}  {row['Percentage']:5.1f}%  {bar}")

# Chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(fi_df["Feature"], fi_df["Percentage"], color="steelblue", edgecolor="black")
ax.set_xlabel("Importance (%)", fontsize=12)
ax.set_title("XGBoost - Feature Importance", fontsize=14)
for bar, pct in zip(bars, fi_df["Percentage"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
plt.close()
print(f"\n  -> feature_importance.png saved")

# ============================================================
# 4. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="black",
           linewidth=0.5, s=40, label="Predictions")

# y=x reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
        label="Perfect Prediction (y=x)")

ax.set_xlabel("Actual Values ($)", fontsize=12)
ax.set_ylabel("Predicted Values ($)", fontsize=12)
ax.set_title("Actual vs Predicted Values", fontsize=14)
ax.legend(fontsize=11)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=200)
plt.close()
print("  -> actual_vs_predicted.png saved")

# ============================================================
# 5. RESIDUALS DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(residuals, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
ax.axvline(x=residuals.mean(), color="orange", linestyle="-", linewidth=2,
           label=f"Mean Error ({residuals.mean():.0f} $)")
ax.set_xlabel("Error (Actual - Predicted) ($)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Residuals Distribution", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "residuals_distribution.png", dpi=200)
plt.close()
print("  -> residuals_distribution.png saved")

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'=' * 60}")
print(f"COMPLETED")
print(f"{'=' * 60}")
print(f"  Charts: {RESULTS_DIR}")
print(f"    - feature_importance.png")
print(f"    - actual_vs_predicted.png")
print(f"    - residuals_distribution.png")
print(f"\n  XGBoost R2 = {r2:.4f} -> Model explains {r2*100:.1f}% of variance")
