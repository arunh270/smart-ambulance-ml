## Synthetic Patient Vital Data

### Signals
- Heart Rate (BPM)
- SpO₂ (%)
- Systolic Blood Pressure (mmHg)
- Diastolic Blood Pressure (mmHg)
- Motion/Vibration (arbitrary units)

### Assumptions
- Adult patient
- Non-invasive sensors
- 1 Hz sampling
- Single patient per file

### Failure Modes
- Motion-induced artifacts
- Sensor dropouts (NaN)
- Flatline readings

### Limitations
- No arrhythmias
- Simplified motion signal
- BP is simulated, not continuous
