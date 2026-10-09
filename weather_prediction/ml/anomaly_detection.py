"""
ml/anomaly_detection.py
WeatherSense AI - Anomaly Detection

Combines Isolation Forest (multivariate), Z-score and IQR (univariate)
to flag unusual weather records, and assigns a severity level.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "cloud_cover"]

EXPECTED_RANGES = {
    "temperature": (5, 45),
    "pressure": (995, 1030),
    "wind_speed": (0, 45),
    "rainfall": (0, 120),
    "humidity": (10, 95),
}


def severity_from_score(score):
    """Isolation Forest score: more negative = more anomalous."""
    if score < -0.15:
        return "Critical"
    if score < -0.08:
        return "High"
    if score < -0.02:
        return "Medium"
    return "Low"


def zscore_iqr_flags(df):
    flags = pd.DataFrame(index=df.index)
    for col in ANOMALY_FEATURES:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        z = (df[col] - mu) / sigma if sigma > 0 else 0
        flags[f"{col}_zscore_flag"] = (z.abs() > 3)

        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags[f"{col}_iqr_flag"] = (df[col] < lower) | (df[col] > upper)
    return flags


def run_anomaly_detection(df, contamination=0.02):
    data = df.dropna(subset=ANOMALY_FEATURES).copy()
    X = data[ANOMALY_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)
    preds = iso.predict(X_scaled)  # -1 = anomaly, 1 = normal

    data["anomaly_score"] = scores
    data["is_anomaly"] = preds == -1

    stat_flags = zscore_iqr_flags(data)
    data = pd.concat([data, stat_flags], axis=1)

    flag_cols = [c for c in stat_flags.columns]
    data["stat_flag_count"] = stat_flags[flag_cols].sum(axis=1)
    data["is_anomaly"] = data["is_anomaly"] | (data["stat_flag_count"] >= 2)

    anomalies = data[data["is_anomaly"]].copy()
    anomalies["severity"] = anomalies["anomaly_score"].apply(severity_from_score)

    def dominant_parameter(row):
        best_col, best_dev = None, 0
        for col in ANOMALY_FEATURES:
            if col in EXPECTED_RANGES:
                lo, hi = EXPECTED_RANGES[col]
                if row[col] < lo:
                    dev = (lo - row[col]) / max(lo, 1)
                elif row[col] > hi:
                    dev = (row[col] - hi) / max(hi, 1)
                else:
                    dev = 0
                if dev > best_dev:
                    best_dev, best_col = dev, col
        return best_col or "temperature"

    anomalies["parameter"] = anomalies.apply(dominant_parameter, axis=1)
    anomalies["value"] = anomalies.apply(lambda r: r[r["parameter"]], axis=1)
    anomalies["expected_range"] = anomalies["parameter"].apply(
        lambda p: f"{EXPECTED_RANGES.get(p, (0, 0))[0]} - {EXPECTED_RANGES.get(p, (0, 0))[1]}"
    )

    result_cols = ["date", "city", "parameter", "value", "expected_range",
                   "anomaly_score", "severity"]
    anomalies_out = anomalies[result_cols].sort_values("anomaly_score").reset_index(drop=True)

    summary = {
        "total_records_checked": len(data),
        "total_anomalies": len(anomalies_out),
        "severity_counts": anomalies_out["severity"].value_counts().to_dict(),
        "anomaly_rate_pct": round(100 * len(anomalies_out) / len(data), 2) if len(data) else 0,
    }

    return anomalies_out, summary, iso, scaler


if __name__ == "__main__":
    import joblib
    df = pd.read_csv("data/weather_features.csv")
    anomalies, summary, iso, scaler = run_anomaly_detection(df)
    print("Anomaly detection summary:", summary)
    print(anomalies.head(10).to_string(index=False))
    anomalies.to_csv("data/anomalies_detected.csv", index=False)
    joblib.dump({"model": iso, "scaler": scaler, "features": ANOMALY_FEATURES},
                "models/anomaly_model.pkl")
    print("Saved data/anomalies_detected.csv and models/anomaly_model.pkl")
