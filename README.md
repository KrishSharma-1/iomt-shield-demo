# 🛡️ IoMT Shield v2 — Hospital Cyber Defence

**Real-time medical device network intrusion detection system** built on the CICIoMT2024 dataset.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iomt-shield-demo.streamlit.app)

## 📋 Features
- **📡 Live Feed** — Real-time IoMT network flow classification (Benign vs Attack)
- **⚡ Attack Scenarios** — One-click DDoS, DoS, and mixed traffic simulation
- **🗺️ Device Map** — Visual hospital floor plan showing compromised devices in real time
- **⚠️ Alert Log** — Persistent SQLite-backed alert history with HIPAA-compliant CSV export
- **📊 Feature Drivers** — Random Forest feature importance analysis

## 🏥 Hospital IoMT Devices Monitored
- ICU Infusion Pumps & Ventilators
- MRI / CT Scanners
- ECG Monitors
- Smart Bed Sensors
- Insulin Dispensers
- MQTT Network Gateway

## 🧠 ML Model
- **Dataset:** CICIoMT2024 (University of New Brunswick) — real hospital IoMT attack traffic
- **Model:** Random Forest Classifier (80 trees, max_depth=16)
- **Features:** 45 network flow features (IAT, flag counts, rate statistics, protocol flags)
- **Attack types:** DoS UDP Flood, DDoS Distributed UDP, Benign baseline

## 🗂️ Project Structure
```
iomt-shield-demo/
├── capture/simulator.py      # Async attack/benign event simulator
├── backend/
│   ├── main.py               # FastAPI backend (WebSocket + REST)
│   ├── inference.py          # ML scoring engine
│   ├── alert_store.py        # SQLite alert persistence
│   └── device_registry.py   # Hospital device IP mapping
├── frontend/app.py           # Streamlit dashboard (5 tabs)
├── data/device_registry.json # 12 hospital IoMT devices
├── models/iomt_rf_model.joblib
├── simulate_attack.py        # CLI attack launcher
└── train.py                  # Model training script
```

## 🏃 Run Locally
```bash
git clone https://github.com/dhirenpreddy/iomt-shield-demo.git
cd iomt-shield-demo
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamlit run frontend/app.py
```

## 🎓 B.Tech Project
Built as part of a 3rd semester B.Tech project on **IoMT Cybersecurity** at hospital networks.
Dataset: [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) — University of New Brunswick
