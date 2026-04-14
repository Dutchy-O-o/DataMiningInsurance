"""OpenAPI 3.0 specification for the Smart Insurance Advisor REST API."""

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Smart Insurance Advisor API",
        "version": "2.0.0",
        "description": (
            "REST API for predicting annual insurance costs with Explainable AI. "
            "Combines XGBoost + SHAP + KNN + Claude AI to deliver interpretable, "
            "actionable predictions.\n\n"
            "**Best model:** XGBoost — R² = 0.88, RMSE = $4,295, MAE = $2,498"
        ),
        "contact": {"url": "https://github.com/Dutchy-O-o/DataMiningInsurance"},
        "license": {"name": "MIT"},
    },
    "servers": [
        {"url": "http://localhost:5000", "description": "Local development"},
    ],
    "tags": [
        {"name": "Prediction",   "description": "Cost prediction endpoints"},
        {"name": "Batch",        "description": "Bulk CSV processing"},
        {"name": "Similarity",   "description": "KNN-based patient lookup"},
        {"name": "Meta",         "description": "Dataset stats & model leaderboard"},
        {"name": "Reports",      "description": "Exportable reports"},
    ],
    "paths": {
        "/api/stats": {
            "get": {
                "tags": ["Meta"],
                "summary": "Dataset statistics and model leaderboard",
                "description": "Returns precomputed stats about the training data and R²/MAE/RMSE of all 5 models.",
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/StatsResponse"}}},
                    }
                },
            }
        },
        "/predict": {
            "post": {
                "tags": ["Prediction"],
                "summary": "Full prediction with SHAP, confidence, similar patients, AI advice",
                "description": (
                    "Generates a complete analysis for a single policyholder profile: "
                    "XGBoost prediction, 3-model confidence interval, SHAP feature "
                    "contributions, what-if scenarios (quit smoking, reach BMI 25), "
                    "5 similar patients from training data, and an AI-written report "
                    "(if ANTHROPIC_API_KEY is configured)."
                ),
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Profile"}}},
                },
                "responses": {
                    "200": {
                        "description": "Complete prediction response",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PredictResponse"}}},
                    }
                },
            }
        },
        "/similar": {
            "post": {
                "tags": ["Similarity"],
                "summary": "5 nearest patients from training data",
                "description": (
                    "K-Nearest Neighbors lookup, stratified by smoker status. "
                    "Always returns 5 patients with the SAME smoker status as the query "
                    "(enforced via pre-filtering). Distance uses weighted Euclidean "
                    "where BMI (×2.5) and age (×2.0) dominate."
                ),
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Profile"}}},
                },
                "responses": {
                    "200": {
                        "description": "5 similar patients with actual charges",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SimilarResponse"}}},
                    },
                    "503": {"description": "KNN model not available"},
                },
            }
        },
        "/batch_predict": {
            "post": {
                "tags": ["Batch"],
                "summary": "Bulk predictions from uploaded CSV",
                "description": (
                    "Upload a CSV with columns: age, sex, bmi, children, smoker, region. "
                    "Returns per-row predicted cost and top SHAP feature. "
                    "Pass `?format=csv` to receive results as a downloadable CSV instead of JSON."
                ),
                "parameters": [
                    {
                        "name": "format",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["json", "csv"]},
                        "description": "Output format (default: json)",
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "CSV with headers: age, sex, bmi, children, smoker, region",
                                    }
                                },
                                "required": ["file"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Predictions for each row",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BatchResponse"}}},
                    },
                    "400": {"description": "Missing columns or invalid CSV"},
                },
            }
        },
        "/download-report": {
            "post": {
                "tags": ["Reports"],
                "summary": "Download prediction report as markdown",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ReportRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Markdown file download",
                        "content": {"text/markdown": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
        "/report": {
            "get": {
                "tags": ["Reports"],
                "summary": "Printable HTML report (PDF-ready)",
                "description": "Returns a print-styled HTML page. Open in a new window and use browser's Print to save as PDF.",
                "responses": {
                    "200": {
                        "description": "HTML page",
                        "content": {"text/html": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "Profile": {
                "type": "object",
                "required": ["age", "sex", "bmi", "children", "smoker", "region"],
                "properties": {
                    "age":      {"type": "integer", "minimum": 18, "maximum": 64, "example": 35},
                    "sex":      {"type": "string", "enum": ["male", "female"], "example": "male"},
                    "bmi":      {"type": "number", "format": "float", "minimum": 15, "maximum": 55, "example": 27.5},
                    "children": {"type": "integer", "minimum": 0, "maximum": 5, "example": 1},
                    "smoker":   {"type": "string", "enum": ["yes", "no"], "example": "no"},
                    "region":   {"type": "string", "enum": ["northeast", "northwest", "southeast", "southwest"], "example": "southeast"},
                },
            },
            "PredictResponse": {
                "type": "object",
                "properties": {
                    "cost":       {"type": "number", "description": "Primary XGBoost prediction ($)", "example": 4500.50},
                    "confidence": {"$ref": "#/components/schemas/Confidence"},
                    "scenarios":  {"$ref": "#/components/schemas/Scenarios"},
                    "shap_bars":  {"type": "array", "items": {"$ref": "#/components/schemas/ShapBar"}},
                    "shap_top":   {"type": "array", "items": {"$ref": "#/components/schemas/ShapTop"}},
                    "advice":     {"type": "string", "nullable": True, "description": "Claude AI report (null if API key not set)"},
                    "similar":    {"$ref": "#/components/schemas/SimilarWithSummary"},
                },
            },
            "Confidence": {
                "type": "object",
                "properties": {
                    "low":  {"type": "integer", "description": "Minimum across 3 boosting models"},
                    "mid":  {"type": "integer", "description": "Mean across 3 boosting models"},
                    "high": {"type": "integer", "description": "Maximum across 3 boosting models"},
                },
            },
            "Scenarios": {
                "type": "object",
                "description": "What-if counterfactual scenarios (only shown if savings > 0)",
                "properties": {
                    "quit_smoking": {"$ref": "#/components/schemas/Scenario"},
                    "healthy_bmi":  {"$ref": "#/components/schemas/Scenario"},
                },
            },
            "Scenario": {
                "type": "object",
                "properties": {
                    "current": {"type": "integer"},
                    "new":     {"type": "integer"},
                    "savings": {"type": "integer"},
                },
            },
            "ShapBar": {
                "type": "object",
                "properties": {
                    "name":  {"type": "string", "example": "Smoker"},
                    "value": {"type": "number", "description": "Dollar contribution (positive = increases cost)"},
                },
            },
            "ShapTop": {
                "type": "object",
                "properties": {
                    "name":      {"type": "string"},
                    "direction": {"type": "string", "enum": ["increases", "decreases"]},
                    "impact":    {"type": "number"},
                },
            },
            "SimilarResponse": {
                "type": "object",
                "properties": {
                    "similar": {"type": "array", "items": {"$ref": "#/components/schemas/SimilarPatient"}},
                    "summary": {"$ref": "#/components/schemas/SimilarSummary"},
                },
            },
            "SimilarWithSummary": {
                "type": "object",
                "properties": {
                    "patients": {"type": "array", "items": {"$ref": "#/components/schemas/SimilarPatient"}},
                    "summary":  {"$ref": "#/components/schemas/SimilarSummary"},
                },
            },
            "SimilarPatient": {
                "type": "object",
                "properties": {
                    "age":           {"type": "integer"},
                    "sex":           {"type": "string"},
                    "bmi":           {"type": "number"},
                    "children":      {"type": "integer"},
                    "smoker":        {"type": "string"},
                    "region":        {"type": "string"},
                    "actual_charge": {"type": "integer"},
                    "similarity":    {"type": "number", "description": "0-100% match score"},
                    "distance":      {"type": "number"},
                },
            },
            "SimilarSummary": {
                "type": "object",
                "properties": {
                    "count":  {"type": "integer"},
                    "min":    {"type": "integer"},
                    "max":    {"type": "integer"},
                    "mean":   {"type": "integer"},
                    "median": {"type": "integer"},
                },
            },
            "BatchResponse": {
                "type": "object",
                "properties": {
                    "results": {"type": "array", "items": {"type": "object"}},
                    "summary": {
                        "type": "object",
                        "properties": {
                            "total":           {"type": "integer"},
                            "successful":      {"type": "integer"},
                            "failed":          {"type": "integer"},
                            "mean_prediction": {"type": "number"},
                            "min_prediction":  {"type": "number"},
                            "max_prediction":  {"type": "number"},
                        },
                    },
                },
            },
            "StatsResponse": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "object",
                        "properties": {
                            "total_rows":    {"type": "integer", "example": 1338},
                            "avg_cost":      {"type": "integer"},
                            "smoker_pct":    {"type": "number"},
                            "smoker_avg":    {"type": "integer"},
                            "nonsmoker_avg": {"type": "integer"},
                        },
                    },
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":    {"type": "string"},
                                "r2":      {"type": "number"},
                                "mae":     {"type": "integer"},
                                "rmse":    {"type": "integer"},
                                "is_best": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "ReportRequest": {
                "type": "object",
                "properties": {
                    "cost":   {"type": "number"},
                    "advice": {"type": "string"},
                    "age":    {"type": "integer"},
                    "bmi":    {"type": "number"},
                    "smoker": {"type": "string"},
                    "region": {"type": "string"},
                },
            },
        }
    },
}
