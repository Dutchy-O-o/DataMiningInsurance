"""
Smart Insurance Advisor V2.0 - Flask Web Application
======================================================
- XGBoost V2 (interaction features + raw target for boosting)
- SHAP Explainability (frontend animated bars)
- Claude Haiku AI with SHAP-informed personalized advice
- Dataset Overview dashboard on welcome screen
- Model Comparison chart (5 models)
- Prediction Confidence interval
- Report download as markdown
"""

import os
import pathlib
import numpy as np
import pandas as pd
import joblib
import shap
import anthropic
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, Response

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "saved_models" / "xgboost.joblib"
FEATURES_PATH = BASE_DIR / "saved_models" / "feature_names.joblib"
CSV_PATH = BASE_DIR / "insurance.csv"
PROCESSED_DIR = BASE_DIR / "processed"

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = joblib.load(FEATURES_PATH)
explainer = shap.TreeExplainer(model)

# Read API key from .env
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Preprocessing constants
SCALER = {
    "age": {"mean": 39.21, "std": 14.04},
    "bmi": {"mean": 30.65, "std": 6.05},
    "children": {"mean": 1.09, "std": 1.21},
}

DISPLAY_NAMES = {
    "age": "Age", "sex": "Sex", "bmi": "BMI", "children": "Children",
    "smoker": "Smoker", "region_northwest": "Region: NW",
    "region_southeast": "Region: SE", "region_southwest": "Region: SW",
    "smoker_bmi": "Smoker x BMI", "smoker_age": "Smoker x Age",
}

# ---- Precompute dataset stats for welcome screen ----
DATASET_STATS = {}
MODEL_SCORES = []
try:
    raw = pd.read_csv(CSV_PATH)
    DATASET_STATS = {
        "total_rows": int(len(raw)),
        "avg_age": round(float(raw["age"].mean()), 1),
        "avg_bmi": round(float(raw["bmi"].mean()), 1),
        "avg_cost": round(float(raw["charges"].mean())),
        "max_cost": round(float(raw["charges"].max())),
        "min_cost": round(float(raw["charges"].min())),
        "smoker_pct": round(float((raw["smoker"] == "yes").mean() * 100), 1),
        "male_pct": round(float((raw["sex"] == "male").mean() * 100), 1),
        "smoker_avg": round(float(raw[raw["smoker"] == "yes"]["charges"].mean())),
        "nonsmoker_avg": round(float(raw[raw["smoker"] == "no"]["charges"].mean())),
    }
except Exception:
    pass

# Model comparison scores (from models.py V2 output)
try:
    model_names = ["XGBoost", "LightGBM", "GradientBoosting", "Ridge Regression", "Linear Regression"]
    model_files = ["xgboost", "lightgbm", "gradient_boosting", "ridge_regression", "linear_regression"]
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    # Add interaction features
    X_test["smoker_bmi"] = X_test["smoker"] * X_test["bmi"]
    X_test["smoker_age"] = X_test["smoker"] * X_test["age"]

    for name, fname in zip(model_names, model_files):
        m = joblib.load(BASE_DIR / "saved_models" / f"{fname}.joblib")
        preds = m.predict(X_test)
        # Linear/Ridge were trained on log target
        if fname in ("linear_regression", "ridge_regression"):
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        MODEL_SCORES.append({
            "name": name, "r2": round(r2, 4),
            "mae": round(mae), "rmse": round(rmse),
            "is_best": fname == "xgboost",
        })
    MODEL_SCORES.sort(key=lambda x: x["r2"], reverse=True)
except Exception:
    pass


# ============================================================
# ML FUNCTIONS
# ============================================================

def scale(value, feature):
    return (value - SCALER[feature]["mean"]) / SCALER[feature]["std"]


def build_features(age, sex, bmi, children, smoker, region):
    age_s = scale(age, "age")
    bmi_s = scale(bmi, "bmi")
    children_s = scale(children, "children")
    sex_enc = 1 if sex == "male" else 0
    smoker_enc = 1 if smoker == "yes" else 0
    data = {
        "age": age_s, "sex": sex_enc, "bmi": bmi_s,
        "children": children_s, "smoker": smoker_enc,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "smoker_bmi": smoker_enc * bmi_s,
        "smoker_age": smoker_enc * age_s,
    }
    return pd.DataFrame([data], columns=FEATURE_NAMES)


def predict_cost(age, sex, bmi, children, smoker, region):
    X = build_features(age, sex, bmi, children, smoker, region)
    return max(float(model.predict(X)[0]), 0)


def predict_confidence(age, sex, bmi, children, smoker, region):
    """Returns low/mid/high estimate using all 3 boosting models."""
    preds = []
    for fname in ["xgboost", "lightgbm", "gradient_boosting"]:
        try:
            m = joblib.load(BASE_DIR / "saved_models" / f"{fname}.joblib")
            X = build_features(age, sex, bmi, children, smoker, region)
            preds.append(max(float(m.predict(X)[0]), 0))
        except Exception:
            pass
    if len(preds) >= 2:
        return {"low": round(min(preds)), "mid": round(np.mean(preds)), "high": round(max(preds))}
    return None


# ============================================================
# SHAP ANALYSIS
# ============================================================

def compute_shap(age, sex, bmi, children, smoker, region):
    X = build_features(age, sex, bmi, children, smoker, region)
    sv = explainer(X)
    values = sv.values[0]
    order = np.argsort(np.abs(values))[::-1]
    shap_bars = []
    for idx in order:
        fname = FEATURE_NAMES[idx]
        shap_bars.append({
            "name": DISPLAY_NAMES.get(fname, fname),
            "value": round(float(values[idx]), 2),
        })
    top_features = []
    for item in shap_bars[:2]:
        direction = "increases" if item["value"] > 0 else "decreases"
        top_features.append({"name": item["name"], "direction": direction,
                             "impact": abs(item["value"])})
    return shap_bars, top_features


# ============================================================
# CLAUDE AI
# ============================================================

def get_ai_advice(age, bmi, smoker, cost, scenarios, shap_top):
    if not API_KEY or API_KEY == "your-api-key-here":
        return None
    lines = ""
    if "quit_smoking" in scenarios:
        s = scenarios["quit_smoking"]
        lines += f"- Quitting smoking: cost drops from ${s['current']:,.0f} to ${s['new']:,.0f}, saving ${s['savings']:,.0f}/year.\n"
    if "healthy_bmi" in scenarios:
        s = scenarios["healthy_bmi"]
        lines += f"- Reaching BMI 25: cost drops from ${s['current']:,.0f} to ${s['new']:,.0f}, saving ${s['savings']:,.0f}/year.\n"
    shap_text = ""
    for f in shap_top:
        shap_text += f"- '{f['name']}' {f['direction']} their cost significantly.\n"

    prompt = f"""You are a professional, empathetic insurance health advisor writing directly to a client.

Client profile: Age {age}, BMI {bmi:.1f}, Smoker: {smoker}, Annual cost: ${cost:,.0f}

SHAP model analysis reveals the top factors driving this client's cost:
{shap_text}

Savings scenarios:
{lines if lines else "- No major risk factors detected. The client is in good shape."}

Write a personalized "Insurance & Health Optimization Report":
1. Start by referencing the SHAP analysis: "Our AI analysis shows that [top factor] is the primary driver of your insurance cost..."
2. Empathetic, encouraging, not judgmental
3. Include specific dollar amounts from the savings scenarios
4. Provide 2-3 actionable health tips
5. Max 200 words, professional warm tone, in English
6. Use markdown formatting with headers
7. IMPORTANT: Do NOT use placeholder names like [Client], [Your Name], or [Name]. Address the reader directly as "you". Do NOT sign off with a name - end with an encouraging closing sentence.
8. Only mention scenarios where savings are positive. If none listed, congratulate and give general wellness tips."""

    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI advisor unavailable: {e}"


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/stats")
def api_stats():
    return jsonify({"dataset": DATASET_STATS, "models": MODEL_SCORES})


@app.route("/predict", methods=["POST"])
def api_predict():
    d = request.json
    age, sex, bmi = int(d["age"]), d["sex"], float(d["bmi"])
    children, smoker, region = int(d["children"]), d["smoker"], d["region"]

    cost = predict_cost(age, sex, bmi, children, smoker, region)
    confidence = predict_confidence(age, sex, bmi, children, smoker, region)

    scenarios = {}
    if smoker == "yes":
        ns_cost = predict_cost(age, sex, bmi, children, "no", region)
        savings = cost - ns_cost
        if savings > 0:
            scenarios["quit_smoking"] = {"current": round(cost), "new": round(ns_cost), "savings": round(savings)}
    if bmi > 25:
        hb_cost = predict_cost(age, sex, 25.0, children, smoker, region)
        savings = cost - hb_cost
        if savings > 0:
            scenarios["healthy_bmi"] = {"current": round(cost), "new": round(hb_cost), "savings": round(savings)}

    shap_bars, shap_top = compute_shap(age, sex, bmi, children, smoker, region)
    advice = get_ai_advice(age, bmi, smoker, cost, scenarios, shap_top)

    return jsonify({
        "cost": round(cost, 2), "confidence": confidence,
        "scenarios": scenarios, "shap_bars": shap_bars,
        "shap_top": shap_top, "advice": advice,
    })


@app.route("/download-report", methods=["POST"])
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
    return Response(content, mimetype="text/markdown",
                    headers={"Content-Disposition": "attachment; filename=insurance_report.md"})


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Insurance Advisor V2.0</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0a0e1a;--surface:#111827;--surface2:#1a2236;--border:rgba(255,255,255,0.06);
    --accent:#3b82f6;--accent2:#8b5cf6;--green:#10b981;--red:#ef4444;--orange:#f59e0b;
    --cyan:#06b6d4;--text:#f1f5f9;--text2:#94a3b8;--text3:#64748b;--glass:rgba(17,24,39,0.7);
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
  body::before{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 20% 50%,rgba(59,130,246,0.08) 0%,transparent 50%),radial-gradient(ellipse at 80% 20%,rgba(139,92,246,0.06) 0%,transparent 50%),radial-gradient(ellipse at 50% 80%,rgba(16,185,129,0.04) 0%,transparent 50%);animation:bgMove 20s ease-in-out infinite;z-index:0;}
  @keyframes bgMove{0%,100%{transform:translate(0,0)}50%{transform:translate(-2%,-1%)}}
  .app{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:40px 24px;}

  .header{text-align:center;margin-bottom:48px;}
  .header .badge{display:inline-flex;align-items:center;gap:6px;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:20px;padding:6px 16px;font-size:12px;font-weight:500;color:var(--accent);letter-spacing:0.5px;text-transform:uppercase;margin-bottom:20px;}
  .badge .dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .header h1{font-size:44px;font-weight:800;letter-spacing:-1.5px;background:linear-gradient(135deg,#f1f5f9,#94a3b8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;}
  .header p{font-size:16px;color:var(--text2);max-width:560px;margin:0 auto;line-height:1.6;}

  .layout{display:grid;grid-template-columns:380px 1fr;gap:32px;align-items:start;}
  @media(max-width:900px){.layout{grid-template-columns:1fr;}}

  .card{background:var(--glass);backdrop-filter:blur(20px);border:1px solid var(--border);border-radius:20px;padding:32px;}
  .card-title{font-size:14px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:28px;display:flex;align-items:center;gap:8px;}
  .card-title .icon{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;}

  .form-group{margin-bottom:20px;}
  .form-group label{display:block;font-size:13px;font-weight:500;color:var(--text2);margin-bottom:8px;}
  .form-group input,.form-group select{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:12px 16px;color:var(--text);font-size:15px;font-family:inherit;outline:none;transition:all 0.2s;}
  .form-group input:focus,.form-group select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(59,130,246,0.1);}
  .form-group select{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%2394a3b8' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;padding-right:36px;}
  .form-group select option{background:var(--surface);}
  .form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;}

  .range-wrap{position:relative;padding-top:4px;}
  .range-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
  .range-value{font-size:14px;font-weight:600;color:var(--accent);background:rgba(59,130,246,0.1);padding:2px 10px;border-radius:6px;}
  .range-wrap input[type=range]{-webkit-appearance:none;width:100%;height:6px;background:var(--surface2);border-radius:3px;outline:none;border:none;padding:0;}
  .range-wrap input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;background:var(--accent);border-radius:50%;cursor:pointer;box-shadow:0 0 10px rgba(59,130,246,0.4);transition:transform 0.15s;}
  .range-wrap input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2);}

  .btn{width:100%;padding:14px;border:none;border-radius:14px;font-family:inherit;font-size:15px;font-weight:600;cursor:pointer;transition:all 0.3s;margin-top:8px;position:relative;overflow:hidden;}
  .btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:white;}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(59,130,246,0.3);}
  .btn-primary:active{transform:translateY(0);}
  .btn-primary.loading{pointer-events:none;opacity:0.8;}
  .btn-download{background:var(--surface2);border:1px solid var(--border);color:var(--text2);padding:10px 20px;border-radius:10px;font-size:13px;font-weight:500;cursor:pointer;transition:all 0.2s;display:inline-flex;align-items:center;gap:6px;width:auto;margin-top:16px;}
  .btn-download:hover{border-color:var(--accent);color:var(--text);}

  .results{display:none;}
  .results.show{display:block;animation:fadeUp 0.5s ease;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

  .cost-hero{text-align:center;padding:40px 32px;background:linear-gradient(135deg,rgba(59,130,246,0.1),rgba(139,92,246,0.1));border:1px solid rgba(59,130,246,0.15);border-radius:20px;margin-bottom:24px;}
  .cost-hero .label{font-size:13px;font-weight:500;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
  .cost-hero .amount{font-size:56px;font-weight:800;letter-spacing:-2px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .cost-hero .sub{font-size:14px;color:var(--text3);margin-top:4px;}
  .cost-hero .confidence{margin-top:12px;font-size:13px;color:var(--text3);display:flex;justify-content:center;gap:24px;}
  .cost-hero .confidence span{display:flex;align-items:center;gap:4px;}
  .cost-hero .confidence .c-val{color:var(--text2);font-weight:600;}

  .scenarios{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
  @media(max-width:700px){.scenarios{grid-template-columns:1fr;}}
  .scenario-card{background:var(--surface2);border:1px solid var(--border);border-radius:16px;padding:24px;transition:all 0.3s;}
  .scenario-card:hover{border-color:rgba(16,185,129,0.3);transform:translateY(-2px);}
  .scenario-card .sc-icon{font-size:24px;margin-bottom:12px;}
  .scenario-card .sc-title{font-size:13px;color:var(--text2);font-weight:500;margin-bottom:4px;}
  .scenario-card .sc-new{font-size:28px;font-weight:700;color:var(--text);}
  .scenario-card .sc-savings{display:inline-flex;align-items:center;gap:4px;margin-top:8px;padding:4px 10px;background:rgba(16,185,129,0.1);border-radius:8px;font-size:13px;font-weight:600;color:var(--green);}

  .shap-section{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:32px;margin-bottom:24px;position:relative;overflow:hidden;}
  .shap-section::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#f59e0b,#ef4444,#8b5cf6);}
  .shap-section .shap-header{display:flex;align-items:center;gap:12px;margin-bottom:24px;}
  .shap-section .shap-icon{width:40px;height:40px;background:linear-gradient(135deg,#f59e0b,#ef4444);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;}
  .shap-section .shap-header div h3{font-size:16px;font-weight:600;}
  .shap-section .shap-header div p{font-size:12px;color:var(--text3);}

  .shap-row{display:grid;grid-template-columns:120px 1fr 80px;align-items:center;gap:12px;margin-bottom:10px;}
  .shap-label{font-size:13px;font-weight:500;color:var(--text2);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .shap-bar-track{height:28px;position:relative;background:rgba(255,255,255,0.03);border-radius:6px;overflow:hidden;}
  .shap-bar-fill{position:absolute;top:2px;bottom:2px;border-radius:4px;transition:width 1s cubic-bezier(0.22,1,0.36,1);width:0;}
  .shap-bar-fill.positive{background:linear-gradient(90deg,#8b5cf6,#a78bfa);left:50%;}
  .shap-bar-fill.negative{background:linear-gradient(270deg,#06b6d4,#22d3ee);right:50%;}
  .shap-bar-track .center-line{position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.1);}
  .shap-val{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;text-align:left;}
  .shap-val.positive{color:#a78bfa;}
  .shap-val.negative{color:#22d3ee;}
  .shap-legend{display:flex;gap:20px;margin-top:16px;padding-top:14px;border-top:1px solid var(--border);}
  .shap-legend span{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text3);}
  .shap-legend .dot-up{width:10px;height:10px;border-radius:3px;background:linear-gradient(135deg,#8b5cf6,#a78bfa);}
  .shap-legend .dot-down{width:10px;height:10px;border-radius:3px;background:linear-gradient(135deg,#06b6d4,#22d3ee);}

  .ai-report{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:32px;position:relative;overflow:hidden;}
  .ai-report::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2),var(--green));}
  .ai-report .report-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;}
  .ai-report .report-header .ai-icon{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;}
  .ai-report .report-header div h3{font-size:16px;font-weight:600;}
  .ai-report .report-header div p{font-size:12px;color:var(--text3);}
  .ai-report .report-body{font-size:14px;line-height:1.8;color:var(--text2);}
  .ai-report .report-body h1,.ai-report .report-body h2,.ai-report .report-body h3{color:var(--text);margin:16px 0 8px;font-size:15px;}
  .ai-report .report-body strong{color:var(--text);}
  .ai-report .report-body ul{padding-left:20px;}
  .ai-report .report-body li{margin-bottom:6px;}
  .report-body p{margin-bottom:8px;}

  .profile-bar{margin-top:24px;padding:16px 24px;background:var(--surface2);border-radius:12px;display:flex;flex-wrap:wrap;gap:20px;font-size:13px;color:var(--text3);}
  .profile-bar span{display:flex;align-items:center;gap:6px;}
  .profile-bar .pval{color:var(--text2);font-weight:500;}

  .spinner{width:24px;height:24px;border:3px solid rgba(255,255,255,0.2);border-top-color:white;border-radius:50%;animation:spin 0.6s linear infinite;display:inline-block;vertical-align:middle;}
  @keyframes spin{to{transform:rotate(360deg)}}

  .bmi-display{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px 16px;font-size:13px;color:var(--text2);margin-bottom:20px;display:flex;align-items:center;gap:10px;}
  .bmi-display strong{color:var(--text);font-size:15px;}
  .bmi-tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;text-transform:uppercase;letter-spacing:0.5px;}
  .bmi-tag.underweight{background:rgba(59,130,246,0.15);color:#60a5fa;}
  .bmi-tag.normal{background:rgba(16,185,129,0.15);color:#34d399;}
  .bmi-tag.overweight{background:rgba(245,158,11,0.15);color:#fbbf24;}
  .bmi-tag.obese{background:rgba(239,68,68,0.15);color:#f87171;}

  /* Welcome Dashboard */
  .welcome-dash{animation:fadeUp 0.6s ease;}
  .welcome-title{font-size:20px;font-weight:700;margin-bottom:6px;color:var(--text);}
  .welcome-sub{font-size:13px;color:var(--text3);margin-bottom:24px;}
  .stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px;}
  @media(max-width:700px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
  .stat-card{background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:18px;text-align:center;transition:all 0.3s;}
  .stat-card:hover{border-color:rgba(59,130,246,0.2);transform:translateY(-2px);}
  .stat-card .s-val{font-size:24px;font-weight:700;margin-bottom:4px;}
  .stat-card .s-label{font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;}
  .stat-card.blue .s-val{color:var(--accent);}
  .stat-card.purple .s-val{color:var(--accent2);}
  .stat-card.green .s-val{color:var(--green);}
  .stat-card.orange .s-val{color:var(--orange);}
  .stat-card.cyan .s-val{color:var(--cyan);}
  .stat-card.red .s-val{color:var(--red);}

  .model-section{background:var(--surface2);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px;}
  .model-section h3{font-size:14px;font-weight:600;color:var(--text2);margin-bottom:16px;display:flex;align-items:center;gap:8px;}
  .model-row{display:grid;grid-template-columns:160px 1fr 70px;align-items:center;gap:12px;margin-bottom:8px;}
  .model-name{font-size:13px;font-weight:500;color:var(--text2);}
  .model-name.best{color:var(--accent);font-weight:600;}
  .model-bar-track{height:22px;background:rgba(255,255,255,0.03);border-radius:6px;overflow:hidden;position:relative;}
  .model-bar-fill{height:100%;border-radius:6px;transition:width 1.2s cubic-bezier(0.22,1,0.36,1);width:0;}
  .model-r2{font-size:13px;font-weight:600;color:var(--text2);font-variant-numeric:tabular-nums;}

  .insight-box{background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.12);border-radius:12px;padding:16px;font-size:13px;color:var(--text2);line-height:1.6;}
  .insight-box strong{color:var(--text);}

  .welcome-arrow{text-align:center;margin-top:8px;font-size:13px;color:var(--text3);}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="badge"><span class="dot"></span> XGBoost ML + SHAP XAI + Claude AI</div>
    <h1>Smart Insurance Advisor</h1>
    <p>AI-powered cost prediction with explainable analysis and personalized health optimization</p>
  </div>

  <div class="layout">
    <!-- LEFT: Form -->
    <div class="card" style="position:sticky;top:24px;">
      <div class="card-title"><div class="icon" style="background:rgba(59,130,246,0.1);color:var(--accent);">&#9881;</div> Your Profile</div>

      <div class="form-group"><div class="range-wrap"><div class="range-header"><label>Age</label><span class="range-value" id="ageVal">30</span></div><input type="range" id="age" min="18" max="64" value="30" oninput="document.getElementById('ageVal').textContent=this.value"></div></div>

      <div class="form-row">
        <div class="form-group"><label>Sex</label><select id="sex"><option value="male">Male</option><option value="female">Female</option></select></div>
        <div class="form-group"><label>Children</label><select id="children"><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option></select></div>
      </div>

      <div class="form-row">
        <div class="form-group"><label>Height (cm)</label><input type="number" id="height" min="100" max="220" value="170" step="1" oninput="updateBmi()"></div>
        <div class="form-group"><label>Weight (kg)</label><input type="number" id="weight" min="30" max="200" value="72" step="1" oninput="updateBmi()"></div>
      </div>
      <div class="bmi-display" id="bmiDisplay">BMI: <strong>24.9</strong> <span class="bmi-tag normal">Normal</span></div>

      <div class="form-row">
        <div class="form-group"><label>Smoker</label><select id="smoker"><option value="no">No</option><option value="yes">Yes</option></select></div>
        <div class="form-group"><label>Region</label><select id="region"><option value="northeast">Northeast</option><option value="northwest">Northwest</option><option value="southeast">Southeast</option><option value="southwest">Southwest</option></select></div>
      </div>

      <button class="btn btn-primary" id="predictBtn" onclick="runPredict()">Calculate My Cost</button>
    </div>

    <!-- RIGHT -->
    <div>
      <!-- Welcome Dashboard -->
      <div id="welcome" class="welcome-dash">
        <div class="welcome-title">Dataset Overview</div>
        <div class="welcome-sub">Trained on real insurance data - explore the insights below</div>
        <div class="stat-grid" id="statGrid"></div>

        <div class="model-section">
          <h3>&#127942; Model Performance Comparison</h3>
          <div id="modelBars"></div>
        </div>

        <div class="insight-box">
          <strong>Key Finding:</strong> Smoking is the #1 cost driver (82.6% feature importance). Smokers pay on average <strong>3.8x more</strong> than non-smokers. The XGBoost model with interaction features captures this non-linear relationship accurately.
        </div>
        <div class="welcome-arrow">&#8592; Fill in your profile and click Calculate to get your personalized analysis</div>
      </div>

      <!-- Results -->
      <div id="results" class="results">
        <div class="cost-hero">
          <div class="label">Estimated Annual Cost</div>
          <div class="amount" id="costAmount">$0</div>
          <div class="sub">Predicted by XGBoost V2.0 (Interaction Features + SHAP)</div>
          <div class="confidence" id="confidence" style="display:none;">
            <span>Low: <span class="c-val" id="confLow">-</span></span>
            <span>Avg: <span class="c-val" id="confMid">-</span></span>
            <span>High: <span class="c-val" id="confHigh">-</span></span>
          </div>
        </div>

        <div class="scenarios" id="scenarios"></div>

        <div class="shap-section" id="shapSection" style="display:none;">
          <div class="shap-header"><div class="shap-icon">&#9881;</div><div><h3>Cost Breakdown - SHAP Analysis</h3><p>Explainable AI: see exactly what drives your cost up or down</p></div></div>
          <div id="shapBars"></div>
          <div class="shap-legend"><span><span class="dot-up"></span> Increases cost</span><span><span class="dot-down"></span> Decreases cost</span></div>
        </div>

        <div class="ai-report" id="aiReport" style="display:none;">
          <div class="report-header"><div class="ai-icon">&#9733;</div><div><h3>AI Advisor Report</h3><p>Powered by Claude AI + SHAP Insights</p></div></div>
          <div class="report-body" id="reportBody"></div>
          <button class="btn-download" onclick="downloadReport()">&#11015; Download Report</button>
        </div>

        <div class="profile-bar" id="profileBar"></div>
      </div>
    </div>
  </div>
</div>

<script>
let lastResult=null, lastData=null;

// Load welcome dashboard on page load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    const ds = data.dataset;
    if (ds && ds.total_rows) {
      document.getElementById('statGrid').innerHTML = `
        <div class="stat-card blue"><div class="s-val">${ds.total_rows.toLocaleString()}</div><div class="s-label">Total Records</div></div>
        <div class="stat-card purple"><div class="s-val">$${ds.avg_cost.toLocaleString()}</div><div class="s-label">Avg Annual Cost</div></div>
        <div class="stat-card orange"><div class="s-val">${ds.smoker_pct}%</div><div class="s-label">Smokers</div></div>
        <div class="stat-card red"><div class="s-val">$${ds.smoker_avg.toLocaleString()}</div><div class="s-label">Avg Smoker Cost</div></div>
        <div class="stat-card green"><div class="s-val">$${ds.nonsmoker_avg.toLocaleString()}</div><div class="s-label">Avg Non-Smoker</div></div>
        <div class="stat-card cyan"><div class="s-val">${ds.avg_age}</div><div class="s-label">Avg Age</div></div>`;
    }
    if (data.models && data.models.length) {
      const mb = document.getElementById('modelBars');
      mb.innerHTML = '';
      data.models.forEach((m, i) => {
        const pct = (m.r2 / 1.0) * 100;
        const colors = ['linear-gradient(90deg,#3b82f6,#8b5cf6)','linear-gradient(90deg,#8b5cf6,#a78bfa)','linear-gradient(90deg,#06b6d4,#22d3ee)','linear-gradient(90deg,#64748b,#94a3b8)','linear-gradient(90deg,#64748b,#94a3b8)'];
        const best = m.is_best ? ' best' : '';
        const crown = m.is_best ? ' &#11088;' : '';
        mb.innerHTML += `<div class="model-row"><div class="model-name${best}">${m.name}${crown}</div><div class="model-bar-track"><div class="model-bar-fill" id="mbar${i}" style="background:${colors[i]};width:0;"></div></div><div class="model-r2">${(m.r2*100).toFixed(1)}%</div></div>`;
        setTimeout(()=>{document.getElementById('mbar'+i).style.width=pct+'%';}, 200+i*100);
      });
    }
  } catch(e) {}
});

function calcBmi(){const h=parseFloat(document.getElementById('height').value)/100;const w=parseFloat(document.getElementById('weight').value);if(h>0&&w>0)return w/(h*h);return 25;}
function updateBmi(){const bmi=calcBmi();let tag,cls;if(bmi<18.5){tag='Underweight';cls='underweight';}else if(bmi<25){tag='Normal';cls='normal';}else if(bmi<30){tag='Overweight';cls='overweight';}else{tag='Obese';cls='obese';}document.getElementById('bmiDisplay').innerHTML=`BMI: <strong>${bmi.toFixed(1)}</strong> <span class="bmi-tag ${cls}">${tag}</span>`;}

async function runPredict(){
  const btn=document.getElementById('predictBtn');
  btn.classList.add('loading');
  btn.innerHTML='<span class="spinner"></span>&nbsp; Analyzing...';
  const bmi=calcBmi();
  lastData={age:document.getElementById('age').value,sex:document.getElementById('sex').value,bmi:bmi.toFixed(1),children:document.getElementById('children').value,smoker:document.getElementById('smoker').value,region:document.getElementById('region').value};
  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastData)});
    lastResult=await res.json();
    document.getElementById('welcome').style.display='none';
    const results=document.getElementById('results');results.classList.remove('show');void results.offsetWidth;results.classList.add('show');
    animateNumber('costAmount',lastResult.cost);

    // Confidence
    if(lastResult.confidence){
      const c=lastResult.confidence;
      document.getElementById('confLow').textContent='$'+c.low.toLocaleString();
      document.getElementById('confMid').textContent='$'+c.mid.toLocaleString();
      document.getElementById('confHigh').textContent='$'+c.high.toLocaleString();
      document.getElementById('confidence').style.display='flex';
    }

    // Scenarios
    const sc=document.getElementById('scenarios');sc.innerHTML='';
    if(lastResult.scenarios.quit_smoking){const s=lastResult.scenarios.quit_smoking;sc.innerHTML+=`<div class="scenario-card"><div class="sc-icon">&#128708;</div><div class="sc-title">If You Quit Smoking</div><div class="sc-new">$${s.new.toLocaleString()}</div><div class="sc-savings">&#9660; Save $${s.savings.toLocaleString()}/year</div></div>`;}
    if(lastResult.scenarios.healthy_bmi){const s=lastResult.scenarios.healthy_bmi;sc.innerHTML+=`<div class="scenario-card"><div class="sc-icon">&#127939;</div><div class="sc-title">If BMI Reaches 25.0</div><div class="sc-new">$${s.new.toLocaleString()}</div><div class="sc-savings">&#9660; Save $${s.savings.toLocaleString()}/year</div></div>`;}

    // SHAP
    if(lastResult.shap_bars&&lastResult.shap_bars.length){
      const container=document.getElementById('shapBars');container.innerHTML='';
      const maxAbs=Math.max(...lastResult.shap_bars.map(b=>Math.abs(b.value)));
      lastResult.shap_bars.forEach((bar,i)=>{
        const pct=(Math.abs(bar.value)/maxAbs)*48;const isPos=bar.value>=0;const cls=isPos?'positive':'negative';const sign=isPos?'+':'-';
        const row=document.createElement('div');row.className='shap-row';
        row.innerHTML=`<div class="shap-label">${bar.name}</div><div class="shap-bar-track"><div class="center-line"></div><div class="shap-bar-fill ${cls}" id="sf${i}" style="width:0%;"></div></div><div class="shap-val ${cls}">${sign}$${Math.abs(bar.value).toLocaleString(undefined,{maximumFractionDigits:0})}</div>`;
        container.appendChild(row);
        setTimeout(()=>{document.getElementById('sf'+i).style.width=pct+'%';},100+i*60);
      });
      document.getElementById('shapSection').style.display='block';
    }

    // AI Report
    const aiDiv=document.getElementById('aiReport');
    if(lastResult.advice){document.getElementById('reportBody').innerHTML=markdownToHtml(lastResult.advice);aiDiv.style.display='block';}else{aiDiv.style.display='none';}

    // Profile
    const sx=lastData.sex==='male'?'Male':'Female',sm=lastData.smoker==='yes'?'Yes':'No',rg=lastData.region.charAt(0).toUpperCase()+lastData.region.slice(1),h=document.getElementById('height').value,w=document.getElementById('weight').value;
    document.getElementById('profileBar').innerHTML=`<span>&#128100; Age <span class="pval">${lastData.age}</span></span><span>&#9878; Sex <span class="pval">${sx}</span></span><span>&#9878; ${h}cm/${w}kg <span class="pval">BMI ${parseFloat(lastData.bmi).toFixed(1)}</span></span><span>&#128118; Children <span class="pval">${lastData.children}</span></span><span>&#128684; Smoker <span class="pval">${sm}</span></span><span>&#127760; Region <span class="pval">${rg}</span></span>`;
  }catch(e){alert('Error: '+e.message);}
  btn.classList.remove('loading');btn.innerHTML='Calculate My Cost';
}

async function downloadReport(){if(!lastResult||!lastResult.advice)return;const res=await fetch('/download-report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cost:lastResult.cost,advice:lastResult.advice,age:lastData.age,bmi:lastData.bmi,smoker:lastData.smoker,region:lastData.region})});const blob=await res.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='insurance_report.md';a.click();URL.revokeObjectURL(url);}

function animateNumber(id,target){const el=document.getElementById(id);const duration=1000;const start=performance.now();function update(now){const p=Math.min((now-start)/duration,1);const ease=1-Math.pow(1-p,3);el.textContent='$'+Math.round(target*ease).toLocaleString();if(p<1)requestAnimationFrame(update);}requestAnimationFrame(update);}

function markdownToHtml(md){return md.replace(/### (.*)/g,'<h3>$1</h3>').replace(/## (.*)/g,'<h2>$1</h2>').replace(/# (.*)/g,'<h1>$1</h1>').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>').replace(/^- (.*)/gm,'<li>$1</li>').replace(/(<li>.*<\/li>)/s,'<ul>$1</ul>').replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>');}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import webbrowser, threading
    print("\n  Smart Insurance Advisor V2.0")
    print("  http://localhost:5000\n")
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000)
