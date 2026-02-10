from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "generated"

# -----------------------
# Loop over all cleaned patient files
# -----------------------
for file_path in data_dir.glob("patient_*_cleaned.csv"):

    print(f"Labeling events for {file_path.name}")

    # -----------------------
    # Load cleaned data
    # -----------------------
    df = pd.read_csv(file_path)

    # -----------------------
    # Initialize event column
    # -----------------------
    df["event"] = "normal"

    # -----------------------
    # Sensor failure
    # -----------------------
    sensor_failure = df["heart_rate"].isna() | df["spo2"].isna()
    df.loc[sensor_failure, "event"] = "sensor_failure"

    # -----------------------
    # Motion artifact
    # -----------------------
    artifact = df["motion"] > 1.0
    df.loc[artifact, "event"] = "artifact"

    # -----------------------
    # Patient distress
    # -----------------------
    distress = (df["heart_rate"] > 120) & (df["spo2"] < 92)
    df.loc[distress, "event"] = "distress"

    # -----------------------
    # Save labeled data
    # -----------------------
    output_path = data_dir / file_path.name.replace("_cleaned.csv", "_labeled.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved labeled file: {output_path.name}")

print("All patient files labeled.")
