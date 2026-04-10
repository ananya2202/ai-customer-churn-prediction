
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS  = os.path.join(ROOT, "models")
DATA    = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
FRONTEND = os.path.join(ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND)
CORS(app)

model           = joblib.load(os.path.join(MODELS, "model.pkl"))
scaler          = joblib.load(os.path.join(MODELS, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS, "feature_columns.pkl"))

print(f"Model loaded     : {MODELS}/model.pkl")
print(f"Features loaded  : {len(feature_columns)} columns")
print(f"Data path        : {DATA}")


def build_feature_vector(data: dict) -> pd.DataFrame:
   
    row = {col: 0 for col in feature_columns}

    tenure          = float(data.get("tenure", 0))
    monthly_charges = float(data.get("monthlyCharges", 0))
    total_charges   = monthly_charges * tenure

    row["tenure"]          = tenure
    row["MonthlyCharges"]  = monthly_charges
    row["TotalCharges"]    = total_charges
    row["AvgMonthlySpend"] = total_charges / (tenure + 1)

    bool_map = {
        "gender":           "gender",
        "seniorCitizen":    "SeniorCitizen",
        "partner":          "Partner",
        "dependents":       "Dependents",
        "phoneService":     "PhoneService",
        "paperlessBilling": "PaperlessBilling",
    }
    for form_key, col_name in bool_map.items():
        if col_name in row:
            val = data.get(form_key, False)
            row[col_name] = 1 if (val is True or val == "Yes" or val == "true") else 0

    contract = data.get("contract", "Month-to-month")
    if "Contract_One year" in row: row["Contract_One year"] = 1 if contract == "One year"  else 0
    if "Contract_Two year" in row: row["Contract_Two year"] = 1 if contract == "Two year"  else 0

    internet = data.get("internetService", "DSL")
    if "InternetService_Fiber optic" in row: row["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    if "InternetService_No"          in row: row["InternetService_No"]          = 1 if internet == "No"          else 0

    support = data.get("techSupport", "No")
    if "TechSupport_Yes"                 in row: row["TechSupport_Yes"]                 = 1 if support == "Yes" else 0
    if "TechSupport_No internet service" in row: row["TechSupport_No internet service"] = 1 if internet == "No" else 0

    payment = data.get("paymentMethod", "Bank transfer (automatic)")
    for method in ["Credit card (automatic)", "Electronic check", "Mailed check"]:
        col = f"PaymentMethod_{method}"
        if col in row:
            row[col] = 1 if payment == method else 0

    mlines = data.get("multipleLines", "No")
    if "MultipleLines_No phone service" in row: row["MultipleLines_No phone service"] = 1 if mlines == "No phone service" else 0
    if "MultipleLines_Yes"              in row: row["MultipleLines_Yes"]              = 1 if mlines == "Yes"              else 0

    addon_map = [
        ("onlineSecurity",   "OnlineSecurity_Yes",   "OnlineSecurity_No internet service"),
        ("onlineBackup",     "OnlineBackup_Yes",      "OnlineBackup_No internet service"),
        ("deviceProtection", "DeviceProtection_Yes",  "DeviceProtection_No internet service"),
        ("streamingTV",      "StreamingTV_Yes",        "StreamingTV_No internet service"),
        ("streamingMovies",  "StreamingMovies_Yes",    "StreamingMovies_No internet service"),
    ]
    for form_key, col_yes, col_no_svc in addon_map:
        val = data.get(form_key, "No")
        if col_yes    in row: row[col_yes]    = 1 if val == "Yes" else 0
        if col_no_svc in row: row[col_no_svc] = 1 if internet == "No" else 0

    return pd.DataFrame([row], columns=feature_columns)


def get_risk_level(prob: float) -> dict:
    if prob >= 0.60:
        return {
            "level":  "High Risk",
            "color":  "high",
            "action": "Initiate a retention call within 24h and offer a loyalty discount or contract upgrade to customergit.",
        }
    if prob >= 0.35:
        return {
            "level":  "Medium Risk",
            "color":  "medium",
            "action": "Send a personalised engagement email highlighting service value and support options.",
        }
    return {
        "level":  "Low Risk",
        "color":  "low",
        "action": "Continue standard service. Flag for review if monthly charges increase significantly.",
    }


def get_factors(data: dict) -> list:
    factors  = []
    tenure   = float(data.get("tenure", 0))
    charges  = float(data.get("monthlyCharges", 0))
    internet = data.get("internetService", "DSL")
    support  = data.get("techSupport", "No")
    payment  = data.get("paymentMethod", "")
    contract = data.get("contract", "Month-to-month")

    if tenure < 12:
        factors.append("Low tenure — customer is still evaluating the service")
    if contract == "Month-to-month":
        factors.append("Month-to-month contract — no long-term commitment")
    if charges > 75:
        factors.append("High monthly charges — may seek cheaper alternatives")
    if support == "No" and charges > 70:
        factors.append("High charges with no tech support — unmet expectations")
    if internet == "Fiber optic" and support == "No":
        factors.append("Fiber optic without tech support — issues likely unresolved")
    if payment == "Electronic check":
        factors.append("Electronic check payment — historically associated with higher churn")

    return factors if factors else ["Customer profile appears stable with no major churn signals"]


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "churn_dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data           = request.get_json(force=True)
        feature_df     = build_feature_vector(data)
        feature_scaled = scaler.transform(feature_df)
        prob           = float(model.predict_proba(feature_scaled)[0][1])
        risk           = get_risk_level(prob)
        factors        = get_factors(data)

        return jsonify({
            "churnProbability":    round(prob * 100, 1),
            "retainProbability":   round((1 - prob) * 100, 1),
            "willChurn":           prob >= 0.5,
            "riskLevel":           risk["level"],
            "riskColor":           risk["color"],
            "recommendedAction":   risk["action"],
            "contributingFactors": factors,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        df            = pd.read_csv(DATA)
        total         = len(df)
        churn_count   = (df["Churn"] == "Yes").sum()
        churn_rate    = round(churn_count / total * 100, 1)
        avg_charges   = round(df["MonthlyCharges"].mean(), 2)
        high_risk_est = min(int(total * (churn_rate / 100) * 1.4), total)

        return jsonify({
            "totalCustomers":    total,
            "churnRate":         churn_rate,
            "highRisk":          high_risk_est,
            "avgMonthlyCharges": avg_charges,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
