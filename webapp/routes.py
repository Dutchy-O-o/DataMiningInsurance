"""Flask routes — REST API endpoints."""
import io
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, render_template, Response

from webapp.features import FEATURE_NAMES, build_features
from webapp.ml_service import (
    model, explainer, predict_cost, predict_confidence,
    compute_shap, what_if_scenarios,
)
from webapp.similarity import similar_with_summary
from webapp.ai_service import get_ai_advice
from webapp.dashboard import DATASET_STATS, MODEL_SCORES
from webapp.openapi import OPENAPI_SPEC
from src.config import DISPLAY_NAMES

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/stats")
def api_stats():
    return jsonify({"dataset": DATASET_STATS, "models": MODEL_SCORES})


@bp.route("/predict", methods=["POST"])
def api_predict():
    d = request.json
    age, sex, bmi = int(d["age"]), d["sex"], float(d["bmi"])
    children, smoker, region = int(d["children"]), d["smoker"], d["region"]

    cost = predict_cost(age, sex, bmi, children, smoker, region)
    confidence = predict_confidence(age, sex, bmi, children, smoker, region)
    scenarios = what_if_scenarios(age, sex, bmi, children, smoker, region, cost)
    shap_bars, shap_top = compute_shap(age, sex, bmi, children, smoker, region)
    advice = get_ai_advice(age, bmi, smoker, cost, scenarios, shap_top)
    similar = similar_with_summary(age, sex, bmi, children, smoker, region)

    return jsonify({
        "cost":       round(cost, 2),
        "confidence": confidence,
        "scenarios":  scenarios,
        "shap_bars":  shap_bars,
        "shap_top":   shap_top,
        "advice":     advice,
        "similar":    similar,
    })


@bp.route("/similar", methods=["POST"])
def api_similar():
    d = request.json
    result = similar_with_summary(
        int(d["age"]), d["sex"], float(d["bmi"]),
        int(d["children"]), d["smoker"], d["region"],
    )
    if result is None:
        return jsonify({"error": "KNN model not available"}), 503
    return jsonify({"similar": result["patients"], "summary": result["summary"]})


@bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    """CSV upload -> per-row predictions + top SHAP feature."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        df_in = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    required = ["age", "sex", "bmi", "children", "smoker", "region"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    results = []
    for _, row in df_in.iterrows():
        try:
            age = int(row["age"])
            sex = str(row["sex"]).lower()
            bmi = float(row["bmi"])
            children = int(row["children"])
            smoker = str(row["smoker"]).lower()
            region = str(row["region"]).lower()

            X = build_features(age, sex, bmi, children, smoker, region)
            cost = max(float(model.predict(X)[0]), 0)
            values = explainer(X).values[0]
            top_idx = int(np.argmax(np.abs(values)))
            results.append({
                "age": age, "sex": sex, "bmi": bmi,
                "children": children, "smoker": smoker, "region": region,
                "predicted_cost": round(cost, 2),
                "top_shap_feature": DISPLAY_NAMES.get(FEATURE_NAMES[top_idx], FEATURE_NAMES[top_idx]),
                "top_shap_impact": round(float(values[top_idx]), 2),
            })
        except Exception as e:
            results.append({"age": row.get("age", None), "error": str(e)})

    # CSV export?
    if request.args.get("format") == "csv":
        csv_buf = io.StringIO()
        pd.DataFrame(results).to_csv(csv_buf, index=False)
        return Response(
            csv_buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"},
        )

    valid = [r for r in results if "predicted_cost" in r]
    summary = {
        "total":           len(results),
        "successful":      len(valid),
        "failed":          len(results) - len(valid),
        "mean_prediction": round(float(np.mean([r["predicted_cost"] for r in valid])), 2) if valid else 0,
        "min_prediction":  round(min(r["predicted_cost"] for r in valid), 2) if valid else 0,
        "max_prediction":  round(max(r["predicted_cost"] for r in valid), 2) if valid else 0,
    }
    return jsonify({"results": results, "summary": summary})


@bp.route("/openapi.json")
def openapi_json():
    """Serve the OpenAPI 3.0 spec as JSON for tooling integration."""
    return jsonify(OPENAPI_SPEC)


@bp.route("/docs")
def api_docs():
    """Swagger UI documentation page (served from CDN, no dependency)."""
    return render_template("swagger.html")


@bp.route("/report", methods=["GET"])
def print_report():
    """Standalone printable report page. Data is pulled from sessionStorage
    on the client side (set by the main UI before opening this route)."""
    return render_template("report_print.html")


@bp.route("/download-report", methods=["POST"])
def download_report():
    d = request.json
    content = f"""# Smart Insurance Advisor - Personal Report

**Estimated Annual Cost:** ${d.get('cost', 0):,.0f}

---

## AI Advisor Analysis

{d.get('advice', 'No report available.')}

---

## Profile Summary
- Age: {d.get('age', '-')}
- BMI: {d.get('bmi', '-')}
- Smoker: {d.get('smoker', '-')}
- Region: {d.get('region', '-')}

---
*Generated by Smart Insurance Advisor V2.0 | XGBoost ML + SHAP + Claude AI*
"""
    return Response(
        content, mimetype="text/markdown",
        headers={"Content-Disposition": "attachment; filename=insurance_report.md"},
    )
