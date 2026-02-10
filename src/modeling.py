from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "generated"

# ============================================================
# LOAD ALL LABELED PATIENT FILES
# ============================================================
dfs = []
for file_path in DATA_DIR.glob("patient_*_labeled.csv"):
    df = pd.read_csv(file_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# ============================================================
# TARGET DEFINITION (DISTRESS VS NON-DISTRESS)
# ============================================================
data["target"] = (data["event"] == "distress").astype(int)

# ============================================================
# BASE FEATURES
# ============================================================
features = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "motion"
]

X = data[features]
y = data["target"]

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]
data = data.loc[X.index].reset_index(drop=True)

# ============================================================
# PART 2A — ANOMALY DETECTION (EARLY WARNING SIGNALS)
# ============================================================
WINDOW = 30  # 30-second rolling window

for col in ["heart_rate", "spo2", "systolic_bp", "diastolic_bp"]:
    data[f"{col}_mean"] = data[col].rolling(WINDOW).mean()
    data[f"{col}_std"] = data[col].rolling(WINDOW).std()
    data[f"{col}_z"] = (data[col] - data[f"{col}_mean"]) / data[f"{col}_std"]

# Signal-level anomalies
data["hr_anomaly"] = data["heart_rate_z"].abs() > 3
data["spo2_anomaly"] = data["spo2_z"] < -3
data["bp_anomaly"] = (
    (data["systolic_bp_z"].abs() > 3) |
    (data["diastolic_bp_z"].abs() > 3)
)

# Combined anomaly flag
data["anomaly"] = (
    data["hr_anomaly"] |
    data["spo2_anomaly"] |
    data["bp_anomaly"]
)

# ============================================================
# PART 2B — RISK SCORE & CONFIDENCE LOGIC
# ============================================================
def compute_risk_score(row):
    score = 0

    # Physiological risk
    if row["heart_rate"] > 120:
        score += 30
    if row["spo2"] < 92:
        score += 40
    if row["systolic_bp"] < 90:
        score += 20

    # Trend-based anomaly risk
    if row["anomaly"]:
        score += 20

    # Motion-aware suppression
    if row["motion"] > 1.5:
        score -= 15

    return np.clip(score, 0, 100)

data["risk_score"] = data.apply(compute_risk_score, axis=1)

# Confidence score (signal quality proxy)
data["confidence"] = 1.0 - (data["motion"].clip(0, 3) / 3.0)
data["confidence"] = data["confidence"].clip(0.1, 1.0)

# Alert logic
data["alert"] = (
    (data["risk_score"] >= 60) &
    (data["confidence"] >= 0.4)
)

# ============================================================
# PART 3A — ALERT METRICS
# ============================================================
true_alerts = data["target"] == 1
pred_alerts = data["alert"] == True

false_alert_rate = (
    ((pred_alerts) & (~true_alerts)).sum() / max(pred_alerts.sum(), 1)
)

alert_latency = data.loc[true_alerts & pred_alerts].index.min()

print("\n================ ALERT METRICS ================\n")
print(f"False Alert Rate: {false_alert_rate:.3f}")
print(f"Alert Latency (index-based): {alert_latency}")

# ============================================================
# TRAIN INTERPRETABLE ML MODEL (LOGISTIC REGRESSION)
# ============================================================
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

y_pred = model.predict(X)

print("\n================ MODEL PERFORMANCE ================\n")
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# ============================================================
# MODEL INTERPRETABILITY
# ============================================================
coefficients = pd.Series(model.coef_[0], index=features)
print("\nModel Coefficients (Interpretability):")
print(coefficients.sort_values(ascending=False))

# ============================================================
# PART 3B — FAILURE ANALYSIS
# ============================================================
results = X.copy()
results["true_label"] = y.values
results["pred_label"] = y_pred

false_positives = results[
    (results["pred_label"] == 1) & (results["true_label"] == 0)
]

false_negatives = results[
    (results["pred_label"] == 0) & (results["true_label"] == 1)
]

print("\n================ FAILURE ANALYSIS ================\n")
print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

print("\nAverage False Positive Signals:")
print(false_positives.mean())

print("\nAverage False Negative Signals:")
print(false_negatives.mean())

# ============================================================
# SAVE OUTPUTS (FOR API / INSPECTION)
# ============================================================
output_path = DATA_DIR / "combined_results.csv"
data.to_csv(output_path, index=False)

print(f"\nSaved combined results to: {output_path}")
