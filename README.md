# 🚦 Traffic Pattern Clustering AI Agent

A machine-learning web application that uses **K-Means clustering** to classify traffic patterns.

## Features

- K-Means traffic clustering
- Four traffic states:
  - Free Flow
  - Normal Traffic
  - Heavy Congestion
  - Severe Congestion
- Flask REST API
- Web dashboard
- Traffic visualization
- Explainable agent recommendations
- Saved ML model with Joblib

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train_model.py
```

This creates:

```text
kmeans_model.joblib
scaler.joblib
metadata.joblib
traffic_data.csv
```

## 3. Start the application

```bash
python app.py
```

Open:

http://localhost:5000

## 4. API

POST:

```text
/api/analyze
```

Example JSON:

```json
{
  "vehicle_count": 180,
  "avg_speed_kmh": 15,
  "occupancy": 85,
  "avg_wait_time_sec": 70
}
```

## Architecture

```text
Traffic Input
     ↓
StandardScaler
     ↓
K-Means
     ↓
Cluster
     ↓
Traffic Pattern
     ↓
Agent Reasoning
     ↓
Recommendation
```

## Next upgrade

Replace the generated demo data with real traffic data and connect a live traffic/camera/IoT source.
