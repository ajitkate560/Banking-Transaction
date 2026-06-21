# ============================================================
# RISK-BASED DIGITAL BANKING AUTHENTICATION SYSTEM
# USING RANDOM FOREST + BANK TRANSACTION DATASET
# ============================================================

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

import joblib

# ===============================
# 2. LOAD DATASET
# ===============================
# Keep bank_transactions.csv in same folder

data = pd.read_csv("bank_transactions.csv")

print("\n✅ Dataset Loaded Successfully")

print("\n📌 Columns in Dataset:\n")
print(data.columns)

# ===============================
# 3. HANDLE MISSING VALUES
# ===============================

data = data.ffill()

# ===============================
# 4. FEATURE ENGINEERING
# ===============================

# ------------------------------------------------
# A = Amount Anomaly
# ------------------------------------------------
data['A'] = data['amount'] / data['amount'].max()

# ------------------------------------------------
# L = Location Risk
# ------------------------------------------------
data['L'] = (
    data['location_radius_km'] /
    data['location_radius_km'].max()
)

# ------------------------------------------------
# D = Device / Authentication Anomaly
# ------------------------------------------------
data['D'] = (
    data['is_new_beneficiary'] +
    data['failed_auth_attempts']
)

data['D'] = data['D'] / data['D'].max()

# ------------------------------------------------
# T = Transaction Type Risk
# ------------------------------------------------
data['T'] = data['transaction_type'].astype('category').cat.codes

data['T'] = data['T'] / data['T'].max()

# ------------------------------------------------
# F = Transaction Frequency / Velocity
# ------------------------------------------------
data['F'] = (
    data['transaction_velocity'] /
    data['transaction_velocity'].max()
)

# ===============================
# 5. RISK SCORE CALCULATION
# (From Research Paper)
# ===============================

data['R'] = (
    0.31 * data['A'] +
    0.24 * data['L'] +
    0.21 * data['D'] +
    0.14 * data['T'] +
    0.10 * data['F']
)

# ===============================
# 6. CREATE RISK LABELS
# ===============================

def classify_risk(r):

    if r < 0.30:
        return "Low"

    elif r < 0.60:
        return "Medium"

    else:
        return "High"

data['risk_label'] = data['R'].apply(classify_risk)

print("\n📊 Risk Distribution BEFORE Balancing:\n")
print(data['risk_label'].value_counts())

# ===============================
# 7. HANDLE CLASS IMBALANCE
# ===============================

low = data[data['risk_label'] == 'Low']
medium = data[data['risk_label'] == 'Medium']
high = data[data['risk_label'] == 'High']

# Upsample High Risk

high_upsampled = resample(
    high,
    replace=True,
    n_samples=3000,
    random_state=42
)

# Combine data

data = pd.concat([
    low,
    medium,
    high_upsampled
])

print("\n📊 Risk Distribution AFTER Balancing:\n")
print(data['risk_label'].value_counts())

# ===============================
# 8. PREPARE TRAINING DATA
# ===============================

X = data[['A', 'L', 'D', 'T', 'F']]

y = data['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 9. TRAIN RANDOM FOREST MODEL
# ===============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Training Completed")

# ===============================
# 10. MODEL EVALUATION
# ===============================

y_pred = model.predict(X_test)

print("\n✅ Model Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:\n")

print(classification_report(y_test, y_pred))

# ===============================
# 11. FEATURE IMPORTANCE
# ===============================

importance = model.feature_importances_

print("\n📌 Feature Importance:\n")

for i, col in enumerate(X.columns):

    print(f"{col}: {importance[i]:.4f}")

# ===============================
# 12. SAVE MODEL
# ===============================

joblib.dump(model, "risk_model.pkl")

print("\n✅ Model saved as 'risk_model.pkl'")

# ===============================
# 13. SAMPLE MANUAL PREDICTION
# ===============================

def predict_risk(A, L, D, T, F):

    score = (
        0.31 * A +
        0.24 * L +
        0.21 * D +
        0.14 * T +
        0.10 * F
    )

    if score < 0.30:
        risk = "Low"

    elif score < 0.60:
        risk = "Medium"

    else:
        risk = "High"

    return score, risk

# Example Transaction

score, risk = predict_risk(
    A=0.8,
    L=0.7,
    D=1,
    T=1,
    F=0.6
)

print("\n🔍 Sample Manual Prediction:")

print("Risk Score:", round(score, 3))
print("Risk Level:", risk)

# ===============================
# 14. REAL DATASET PREDICTION
# ===============================

sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\n🏦 Real Dataset Prediction:")

print("Predicted Risk:", prediction[0])

# ===============================
# 15. COMPLETE
# ===============================

print("\n✅ COMPLETE SYSTEM EXECUTED SUCCESSFULLY")