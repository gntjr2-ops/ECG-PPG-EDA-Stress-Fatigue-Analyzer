# Heuristic Fusion Algorithm for Automatic Fatigue & Stress Assessment Using ECG·PPG·EDA Signals

## Overview
This project provides an **explainable, real-time physiological state assessment system** that automatically classifies **stress**, **fatigue**, and **normal** conditions using wearable biosignals:

- **ECG (Electrocardiogram)**
- **PPG (Photoplethysmogram)**
- **EDA (Electrodermal Activity)**

Unlike machine-learning or deep-learning-based approaches, this system uses a **heuristic fusion algorithm**, where physiological indicators and rule-based logic directly determine the user's state.  
This enables:

- No need for large training datasets  
- Real-time, low-latency inference  
- Transparent and explainable decision-making  

---

## Key Features

### 1. Multimodal Physiological Indicators
The system extracts biologically meaningful metrics from each signal:

**ECG**
- HR (Heart Rate)
- HRV (SDNN, RMSSD)
- LF/HF ratio (autonomic balance)

**PPG**
- Pulse Transit Time (PTT)
- Waveform morphology: Reflection Index (RI), Augmentation Index (AIx)

**EDA**
- Skin Conductance Level (SCL)
- Skin Conductance Response frequency (SCR_freq)

Combining multimodal indicators improves reliability and reduces false positives.

---

### 2. Heuristic Fusion Classification Algorithm
State classification uses **compound physiological rules**, not single-signal thresholds.

Example (Stress):
