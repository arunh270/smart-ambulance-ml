from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "generated"

# -----------------------
# Load all labeled patient files
# -----------------------
dfs = []

for file_path in data_dir.glob("patient_*_labeled.csv"):
    df = pd.read_csv(file_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# -----------------------
# Create binary target
# -----------------------
data["target"] = (data["event"] == "distress").astype(int)

# -----------------------
# Select features
# -----------------------
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

# -----------------------
# Train simple model
# -----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------
# Evaluate on same data
# (this is intentional)
# -----------------------
y_pred = model.predict(X)

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# -----------------------
# Model interpretability
# -----------------------
coefficients = pd.Series(model.coef_[0], index=features)
print("\nModel Coefficients:")
print(coefficients.sort_values(ascending=False))

# -----------------------
# Failure analysis
# -----------------------
results = X.copy()
results["true_label"] = y.values
results["pred_label"] = y_pred

# False positives: predicted distress but not true distress
false_positives = results[(results["pred_label"] == 1) & (results["true_label"] == 0)]

# False negatives: missed distress
false_negatives = results[(results["pred_label"] == 0) & (results["true_label"] == 1)]

print("\nFalse Positives count:", len(false_positives))
print("False Negatives count:", len(false_negatives))

print("\nAverage values for False Positives:")
print(false_positives.mean())

print("\nAverage values for False Negatives:")
print(false_negatives.mean())



