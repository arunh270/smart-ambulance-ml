from pathlib import Path
import numpy as np
import pandas as pd


# get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# create data/generated folder if missing
output_dir = BASE_DIR / "data" / "generated"
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# Simulation parameters
# -----------------------
DURATION_MIN = 30
FS = 1  # 1 sample per second
N = DURATION_MIN * 60 * FS

np.random.seed(42)

time = np.arange(N)

# -----------------------
# Baseline physiology
# -----------------------
heart_rate = 75 + np.random.normal(0, 2, N)
spo2 = 98 + np.random.normal(0, 0.3, N)
systolic_bp = 120 + np.random.normal(0, 3, N)
diastolic_bp = 80 + np.random.normal(0, 2, N)

# -----------------------
# Motion / vibration
# -----------------------
motion = np.random.normal(0, 0.2, N)

# Ambulance bumps
for t in range(300, 1200, 200):
    motion[t:t+20] += np.random.normal(2, 0.5, 20)

# -----------------------
# Patient distress
# -----------------------
distress_start = 15 * 60
distress_end = 20 * 60

heart_rate[distress_start:distress_end] += np.linspace(0, 40, distress_end - distress_start)
spo2[distress_start:distress_end] -= np.linspace(0, 6, distress_end - distress_start)
motion[distress_start:distress_end] += 1.5

# -----------------------
# Sensor artifacts
# -----------------------
artifact_idx = motion > 1.0
heart_rate[artifact_idx] += np.random.normal(5, 2, artifact_idx.sum())
spo2[artifact_idx] -= np.random.normal(3, 1, artifact_idx.sum())

# -----------------------
# Sensor failures
# -----------------------
spo2[1000:1050] = np.nan
heart_rate[1400:1420] = 0

# -----------------------
# Save data
# -----------------------
df = pd.DataFrame({
    "time_sec": time,
    "heart_rate": heart_rate,
    "spo2": spo2,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "motion": motion
})


def generate_patient(patient_id, hr_base=75, spo2_base=98):
    import numpy as np
    import pandas as pd

    DURATION_MIN = 30
    N = DURATION_MIN * 60
    time = np.arange(N)

    heart_rate = hr_base + np.random.normal(0, 2, N)
    spo2 = spo2_base + np.random.normal(0, 0.3, N)
    systolic_bp = 120 + np.random.normal(0, 3, N)
    diastolic_bp = 80 + np.random.normal(0, 2, N)

    motion = np.random.normal(0, 0.2, N)

    for t in range(300, 1200, 200):
        motion[t:t+20] += np.random.normal(2, 0.5, 20)

    distress_start = 15 * 60
    distress_end = 20 * 60

    heart_rate[distress_start:distress_end] += np.linspace(0, 40, distress_end - distress_start)
    spo2[distress_start:distress_end] -= np.linspace(0, 6, distress_end - distress_start)
    motion[distress_start:distress_end] += 1.5

    artifact_idx = motion > 1.0
    heart_rate[artifact_idx] += np.random.normal(5, 2, artifact_idx.sum())
    spo2[artifact_idx] -= np.random.normal(3, 1, artifact_idx.sum())

    spo2[1000:1050] = np.nan
    heart_rate[1400:1420] = 0

    df = pd.DataFrame({
        "time_sec": time,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "motion": motion
    })

    return df



from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "generated"

patients = [
    {"id": "01", "hr": 72, "spo2": 98},
    {"id": "02", "hr": 85, "spo2": 97},
    {"id": "03", "hr": 65, "spo2": 99},
]

for p in patients:
    df = generate_patient(p["id"], p["hr"], p["spo2"])
    df.to_csv(data_dir / f"patient_{p['id']}.csv", index=False)

print("Multiple patient data generated.")



df.to_csv(output_dir / "patient_01.csv", index=False)


print("Data generation complete: patient_01.csv")
