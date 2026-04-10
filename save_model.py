import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data",   "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
mask = df["TotalCharges"].isna()
df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]

df["Churn"] = (df["Churn"] == "Yes").astype(int)
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[col] = (df[col] == "Yes").astype(int)
df["gender"] = (df["gender"] == "Male").astype(int)

ohe_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_columns = X.columns.tolist()
print(f"Features: {len(feature_columns)}")

print("Scaling and applying SMOTE...")
scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

smote        = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print(f"After SMOTE: Class 0: {(y_res==0).sum():,} | Class 1: {(y_res==1).sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5, eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nTest Results:")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(model,           os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler,          os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

print(f"\nSaved to models/: model.pkl | scaler.pkl | feature_columns.pkl")
print("Run next: python backend/app.py")
