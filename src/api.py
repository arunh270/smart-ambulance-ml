from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Smart Ambulance Risk API")

# -----------------------
# Input schema
# -----------------------
class VitalInput(BaseModel):
    heart_rate: float
    spo2: float
    systolic_bp: float
    diastolic_bp: float
    motion: float
    anomaly: bool

# -----------------------
# Risk logic (same as modeling.py)
# -----------------------
def compute_risk(v):
    score = 0

    if v.heart_rate > 120:
        score += 30
    if v.spo2 < 92:
        score += 40
    if v.systolic_bp < 90:
        score += 20
    if v.anomaly:
        score += 20
    if v.motion > 1.5:
        score -= 15

    score = max(0, min(100, score))

    confidence = 1.0 - min(v.motion, 3) / 3.0
    confidence = max(0.1, confidence)

    alert = score >= 60 and confidence >= 0.4

    return {
        "risk_score": score,
        "confidence": round(confidence, 2),
        "alert": alert
    }

# -----------------------
# API endpoint
# -----------------------
@app.post("/assess")
def assess_risk(vitals: VitalInput):
    return compute_risk(vitals)
