# Smart Ambulance – Time-Series ML Assignment

This project simulates patient vital signs streamed in real time during
ambulance transport. The goal is to demonstrate engineering judgment,
signal understanding, and safety-aware machine learning rather than
maximize predictive accuracy.

## Overview
Patient vitals are simulated at 1 Hz over 30-minute transports, including:
- Normal transport conditions
- Patient distress scenarios
- Ambulance motion artifacts
- Sensor failures and dropouts

A full pipeline is implemented:
data generation → preprocessing → event labeling → simple modeling →
failure and safety analysis.

## Repository Structure
- data/: synthetic patient data and documentation
- src/: data generation, preprocessing, labeling, and modeling scripts
- notebooks/: exploratory analysis and visualization
- report.md: detailed technical report

## How to Run
```bash
python src/data_generation.py
python src/preprocessing.py
python src/label_events.py
python src/modeling.py
