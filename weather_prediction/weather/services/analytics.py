"""
weather/services/analytics.py
WeatherSense AI - Dashboard & analytics aggregation helpers
"""

import numpy as np
import pandas as pd

from . import data_loader


def apply_filters(df, city=None, year=None, month=None, season=None, condition=None):
    if city:
        df = df[df["city"] == city]
    if year:
        df = df[df["date"].dt.year == int(year)]
    if month:
        df = df[df["month"] == int(month)]
    if season:
        df = df[df["season"] == season]
    if condition:
        df = df[df["weather_condition"] == condition]
    return df


def dashboard_kpis(df):
    anomalies = data_loader.get_results().get("anomaly_detection", {}).get("summary", {})
    return {
        "total_records": int(len(df)),
        "avg_temperature": round(float(df["temperature"].mean()), 1) if len(df) else 0,
        "max_temperature": round(float(df["temperature"].max()), 1) if len(df) else 0,
        "min_temperature": round(float(df["temperature"].min()), 1) if len(df) else 0,
        "avg_humidity": round(float(df["humidity"].mean()), 1) if len(df) else 0,
        "total_rainfall": round(float(df["rainfall"].sum()), 1) if len(df) else 0,
        "extreme_events": int(df["heatwave_indicator"].sum() + df["storm_indicator"].sum()) if len(df) else 0,
        "num_anomalies": anomalies.get("total_anomalies", 0),
    }


def trend_series(df, value_col, freq="ME"):
    s = df.set_index("date")[value_col].resample(freq).mean().dropna()
    return {"labels": [d.strftime("%Y-%m") if freq in ("M", "ME") else d.strftime("%Y-%m-%d") for d in s.index],
            "values": [round(float(v), 2) for v in s.values]}


def weather_condition_distribution(df):
    counts = df["weather_condition"].value_counts()
    return {"labels": counts.index.tolist(), "values": counts.values.tolist()}


def city_comparison_data(df, cities):
    sub = df[df["city"].isin(cities)]
    agg = sub.groupby("city").agg(
        avg_temperature=("temperature", "mean"),
        avg_rainfall=("rainfall", "mean"),
        total_rainfall=("rainfall", "sum"),
        avg_humidity=("humidity", "mean"),
        avg_wind_speed=("wind_speed", "mean"),
        extreme_events=("heatwave_indicator", "sum"),
    ).round(2)
    return agg.to_dict(orient="index")


def seasonal_analysis(df):
    agg = df.groupby("season").agg(
        avg_temperature=("temperature", "mean"),
        avg_rainfall=("rainfall", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_wind_speed=("wind_speed", "mean"),
        extreme_events=("heatwave_indicator", "sum"),
    ).round(2)
    order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    agg = agg.reindex([o for o in order if o in agg.index])
    return agg.to_dict(orient="index")


def mann_kendall_test(series):
    """Simple Mann-Kendall trend test implementation (no extra dependency)."""
    x = np.asarray(series)
    n = len(x)
    if n < 4:
        return {"trend": "Insufficient data", "s": 0, "p_approx": None}

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))

    unique, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0
    else:
        z = 0

    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))

    if p < 0.05:
        trend = "Increasing" if s > 0 else "Decreasing"
    else:
        trend = "No statistically significant trend"

    return {"trend": trend, "s": int(s), "z": round(float(z), 3), "p_value": round(float(p), 4)}


def sens_slope(series):
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    return round(float(np.median(slopes)), 5)


def trend_analysis(df, city, variable="temperature", freq="ME"):
    sub = df[df["city"] == city].sort_values("date")
    monthly = sub.set_index("date")[variable].resample(freq).mean().dropna()
    if len(monthly) > 200:
        monthly = monthly.iloc[::max(1, len(monthly) // 200)]

    mk = mann_kendall_test(monthly.values)
    slope = sens_slope(monthly.values)

    # simple linear trend for the moving-average chart
    ma = monthly.rolling(6, min_periods=1).mean()

    return {
        "labels": [d.strftime("%Y-%m") for d in monthly.index],
        "values": [round(float(v), 2) for v in monthly.values],
        "moving_average": [round(float(v), 2) for v in ma.values],
        "mann_kendall": mk,
        "sens_slope_per_period": slope,
    }
