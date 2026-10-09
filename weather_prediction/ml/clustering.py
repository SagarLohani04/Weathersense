"""
ml/clustering.py
WeatherSense AI - Weather-condition clustering (K-Means)

Determines a reasonable K using the elbow method + silhouette score,
fits K-Means, names clusters by examining their actual statistics
(rather than blindly assigning names), and produces a PCA 2D
projection for visualization.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "rainfall"]


def choose_k(X_scaled, k_range=range(3, 8)):
    # Note: K is restricted to >=3 by design so that distinct weather regimes
    # remain interpretable/actionable; K=2 is evaluated too and reported for
    # transparency but not auto-selected, since a 2-way split is too coarse
    # to be useful for a weather-risk dashboard.
    inertias = {}
    sil_scores = {}
    for k in range(2, 8):  # compute full range for the elbow chart / transparency
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = km.inertia_
        sil_scores[k] = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=42)
    candidates = {k: v for k, v in sil_scores.items() if k in k_range}
    best_k = max(candidates, key=candidates.get)
    return best_k, inertias, sil_scores


def name_cluster(stats_row):
    """Assign a human-meaningful name based on the cluster's actual mean values."""
    t, h, p, w, c, r = (stats_row[f] for f in CLUSTER_FEATURES)
    if r > 25 or (h > 75 and c > 65):
        return "Rainy"
    if t > 33 and h < 45:
        return "Dry & Hot"
    if w > 30 or (p < 1005 and w > 20):
        return "Extreme Weather"
    if h > 70 and t < 26:
        return "Cool & Humid"
    if 20 <= t <= 30 and h <= 70 and r < 5:
        return "Mild"
    return "Transitional"


def run_clustering(df):
    data = df.dropna(subset=CLUSTER_FEATURES).copy()
    X = data[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, inertias, sil_scores = choose_k(X_scaled)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    data["cluster"] = labels

    cluster_stats = data.groupby("cluster")[CLUSTER_FEATURES].mean()
    cluster_names = {c: name_cluster(cluster_stats.loc[c]) for c in cluster_stats.index}
    data["cluster_name"] = data["cluster"].map(cluster_names)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    data["pca_x"] = coords[:, 0]
    data["pca_y"] = coords[:, 1]

    result = {
        "best_k": int(best_k),
        "inertias": {int(k): float(v) for k, v in inertias.items()},
        "silhouette_scores": {int(k): float(v) for k, v in sil_scores.items()},
        "cluster_names": {int(k): v for k, v in cluster_names.items()},
        "cluster_stats": cluster_stats.round(2).to_dict(orient="index"),
        "cluster_sizes": data["cluster"].value_counts().to_dict(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return data, km, scaler, result


if __name__ == "__main__":
    import joblib
    df = pd.read_csv("data/weather_features.csv")
    data, km, scaler, result = run_clustering(df)
    print(f"Best K (by silhouette): {result['best_k']}")
    print("Cluster names:", result["cluster_names"])
    print("Cluster sizes:", result["cluster_sizes"])
    joblib.dump({"model": km, "scaler": scaler, "features": CLUSTER_FEATURES}, "models/clustering_model.pkl")
    print("Saved models/clustering_model.pkl")
