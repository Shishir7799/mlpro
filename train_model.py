import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent

# ---------------------------------------------------------
# Create a realistic demo traffic dataset.
# Replace this later with your real traffic CSV.
# ---------------------------------------------------------
rng = np.random.default_rng(42)
patterns = {
    "Free Flow":        (30, 55, 0.20, 8),
    "Normal Traffic":   (80, 42, 0.40, 20),
    "Heavy Congestion": (145, 25, 0.68, 48),
    "Severe Congestion":(210, 12, 0.88, 82),
}

rows = []
for pattern, (vehicles, speed, occupancy, wait) in patterns.items():
    for _ in range(300):
        rows.append({
            "vehicle_count": max(1, rng.normal(vehicles, vehicles * .12)),
            "avg_speed_kmh": max(3, rng.normal(speed, speed * .10)),
            "occupancy": np.clip(rng.normal(occupancy, .045), .01, .99),
            "avg_wait_time_sec": max(1, rng.normal(wait, max(2, wait * .12))),
            "expected_pattern": pattern
        })

df = pd.DataFrame(rows)

FEATURES = [
    "vehicle_count",
    "avg_speed_kmh",
    "occupancy",
    "avg_wait_time_sec"
]

X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Four traffic states are required for this application.
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

df["cluster"] = model.fit_predict(X_scaled)

# Automatically map arbitrary K-Means cluster numbers
# to human-readable traffic states.
centers = pd.DataFrame(
    scaler.inverse_transform(model.cluster_centers_),
    columns=FEATURES
)

# Higher score = more congested.
centers["congestion_score"] = (
    centers["vehicle_count"].rank(pct=True)
    + (-centers["avg_speed_kmh"]).rank(pct=True)
    + centers["occupancy"].rank(pct=True)
    + centers["avg_wait_time_sec"].rank(pct=True)
)

ordered_clusters = centers["congestion_score"].sort_values().index.tolist()

names = [
    "Free Flow",
    "Normal Traffic",
    "Heavy Congestion",
    "Severe Congestion"
]

cluster_names = {
    int(cluster): names[i]
    for i, cluster in enumerate(ordered_clusters)
}

df["traffic_pattern"] = df["cluster"].map(cluster_names)

joblib.dump(model, OUT / "kmeans_model.joblib")
joblib.dump(scaler, OUT / "scaler.joblib")
joblib.dump({
    "features": FEATURES,
    "cluster_names": cluster_names
}, OUT / "metadata.joblib")

df.drop(columns=["expected_pattern"]).to_csv(
    OUT / "traffic_data.csv",
    index=False
)

print("Model trained successfully.")
print("Cluster mapping:", cluster_names)
print("Saved:")
print("  kmeans_model.joblib")
print("  scaler.joblib")
print("  metadata.joblib")
print("  traffic_data.csv")
