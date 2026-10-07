import joblib
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

model = joblib.load(BASE / "kmeans_model.joblib")
scaler = joblib.load(BASE / "scaler.joblib")
metadata = joblib.load(BASE / "metadata.joblib")

FEATURES = metadata["features"]
CLUSTER_NAMES = metadata["cluster_names"]

def analyze_traffic(vehicle_count, avg_speed_kmh, occupancy, avg_wait_time_sec):
    values = np.array([[
        float(vehicle_count),
        float(avg_speed_kmh),
        float(occupancy),
        float(avg_wait_time_sec)
    ]])

    scaled = scaler.transform(values)
    cluster = int(model.predict(scaled)[0])
    pattern = CLUSTER_NAMES[cluster]

    # A simple interpretable agent layer.
    reasons = []

    if vehicle_count >= 150:
        reasons.append("vehicle volume is high")
    elif vehicle_count >= 70:
        reasons.append("vehicle volume is moderate")

    if avg_speed_kmh <= 20:
        reasons.append("average speed is very low")
    elif avg_speed_kmh <= 35:
        reasons.append("average speed is reduced")

    if occupancy >= 0.75:
        reasons.append("road occupancy is very high")
    elif occupancy >= 0.50:
        reasons.append("road occupancy is elevated")

    if avg_wait_time_sec >= 60:
        reasons.append("average waiting time is very high")
    elif avg_wait_time_sec >= 30:
        reasons.append("average waiting time is increasing")

    if pattern == "Free Flow":
        recommendation = (
            "Traffic is moving freely. Keep the current signal plan "
            "and continue monitoring."
        )
    elif pattern == "Normal Traffic":
        recommendation = (
            "Traffic is normal. Continue monitoring the junction "
            "for changes in volume and speed."
        )
    elif pattern == "Heavy Congestion":
        recommendation = (
            "Consider extending the green phase for the congested "
            "direction and monitor alternate routes."
        )
    else:
        recommendation = (
            "Severe congestion detected. Consider adaptive signal "
            "timing, diversion routes, and immediate junction monitoring."
        )

    return {
        "cluster": cluster,
        "pattern": pattern,
        "reasons": reasons,
        "recommendation": recommendation,
        "input": {
            "vehicle_count": vehicle_count,
            "avg_speed_kmh": avg_speed_kmh,
            "occupancy": occupancy,
            "avg_wait_time_sec": avg_wait_time_sec
        }
    }
