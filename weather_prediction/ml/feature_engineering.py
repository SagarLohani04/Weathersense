"""
ml/feature_engineering.py
WeatherSense AI - Feature Engineering

Creates derived features used by the classification/regression/rainfall
models: lag features, rolling statistics, cyclical date encodings and
binary flags. Operates per-city so lag/rolling windows never leak across
city boundaries, and always sorts chronologically first (no shuffling)
to avoid look-ahead leakage.
"""

import numpy as np
import pandas as pd


def engineer_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    out_frames = []
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        g["temperature_range"] = g["max_temperature"] - g["min_temperature"]
        g["humidity_temperature_ratio"] = g["humidity"] / g["temperature"].replace(0, np.nan)
        g["pressure_change"] = g["pressure"].diff()
        g["temperature_change"] = g["temperature"].diff()

        for lag in (1, 3, 7):
            g[f"rainfall_lag_{lag}"] = g["rainfall"].shift(lag)
        for lag in (1, 3):
            g[f"temperature_lag_{lag}"] = g["temperature"].shift(lag)
        g["humidity_lag_1"] = g["humidity"].shift(1)

        g["rolling_temperature_3"] = g["temperature"].rolling(3, min_periods=1).mean()
        g["rolling_temperature_7"] = g["temperature"].rolling(7, min_periods=1).mean()
        g["rolling_rainfall_7"] = g["rainfall"].rolling(7, min_periods=1).mean()
        g["rolling_humidity_7"] = g["humidity"].rolling(7, min_periods=1).mean()
        g["rainfall_7_day_total"] = g["rainfall"].rolling(7, min_periods=1).sum()
        g["temperature_7_day_average"] = g["rolling_temperature_7"]

        g["extreme_temperature_flag"] = (
            g["temperature"] > g["temperature"].quantile(0.95)
        ).astype(int)
        g["high_humidity_flag"] = (g["humidity"] > 85).astype(int)
        g["heavy_rain_flag"] = (g["rainfall"] > 40).astype(int)

        # Targets for supervised models (next-day values, still sorted chronologically)
        g["rainfall_next_day"] = (g["rainfall"].shift(-1) > 2.0).astype("Int64")
        g["max_temperature_next_day"] = g["max_temperature"].shift(-1)

        out_frames.append(g)

    df = pd.concat(out_frames, ignore_index=True)

    # Cyclical encodings
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Fill lag/rolling NaNs at series start with column median (only a handful of rows per city)
    lag_cols = [c for c in df.columns if "lag" in c or c.startswith("rolling_") or c == "pressure_change"
                or c == "temperature_change" or c == "humidity_temperature_ratio"]
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/weather_cleaned.csv")
    feat = engineer_features(df)
    feat.to_csv("data/weather_features.csv", index=False)
    print(f"Feature-engineered dataset shape: {feat.shape}")
    print(f"New columns added: {feat.shape[1] - df.shape[1]}")
