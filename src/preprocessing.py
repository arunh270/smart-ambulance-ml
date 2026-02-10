from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "generated"

# -----------------------
# Loop over all patient files
# -----------------------
for file_path in data_dir.glob("patient_*.csv"):

    # Skip already cleaned files
    if "cleaned" in file_path.name or "labeled" in file_path.name:
        continue

    print(f"Preprocessing {file_path.name}")

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(file_path)

    # -----------------------
    # Handle impossible values
    # -----------------------
    df["heart_rate"] = df["heart_rate"].replace(0, np.nan)
    df["spo2"] = df["spo2"].clip(70, 100)

    df.loc[df["systolic_bp"] < 80, "systolic_bp"] = np.nan
    df.loc[df["diastolic_bp"] < 40, "diastolic_bp"] = np.nan

    # -----------------------
    # Handle missing values
    # -----------------------
    df["heart_rate"] = df["heart_rate"].interpolate(limit=10)
    df["spo2"] = df["spo2"].interpolate(limit=10)

    df["systolic_bp"] = df["systolic_bp"].ffill()
    df["diastolic_bp"] = df["diastolic_bp"].ffill()

    # -----------------------
    # Smooth noisy signals
    # -----------------------
    df["heart_rate"] = df["heart_rate"].rolling(window=5, center=True).mean()
    df["spo2"] = df["spo2"].rolling(window=5, center=True).mean()

    # -----------------------
    # Save cleaned data
    # -----------------------
    output_path = data_dir / file_path.name.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned file: {output_path.name}")

print("All patient files preprocessed.")
