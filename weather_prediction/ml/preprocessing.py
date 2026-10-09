"""
ml/preprocessing.py
WeatherSense AI - Data Preprocessing Pipeline

Handles: loading, type conversion, duplicate removal, missing-value
treatment, outlier detection/treatment, invalid-value correction and
categorical encoding. Returns the cleaned dataframe plus a stats dict
describing what was found/fixed (used both by train_models.py and by
the Django "Upload Dataset" feature).
"""

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "latitude", "longitude", "month", "day_of_year", "temperature",
    "feels_like_temperature", "min_temperature", "max_temperature",
    "humidity", "dew_point", "pressure", "wind_speed", "wind_direction",
    "cloud_cover", "visibility", "precipitation", "precipitation_probability",
    "solar_radiation", "uv_index", "rainfall", "heat_index",
    "air_quality_index", "storm_indicator", "fog_indicator",
    "heatwave_indicator", "flood_risk", "drought_risk",
]

REQUIRED_COLUMNS = [
    "date", "city", "temperature", "humidity", "pressure",
    "wind_speed", "cloud_cover", "rainfall",
]

# Physically valid ranges used for invalid-value / outlier correction
VALID_RANGES = {
    "temperature": (-10, 55),
    "humidity": (0, 100),
    "pressure": (930, 1050),
    "wind_speed": (0, 150),
    "cloud_cover": (0, 100),
    "rainfall": (0, 300),
    "visibility": (0, 10),
    "uv_index": (0, 14),
    "air_quality_index": (0, 500),
}


def validate_columns(df):
    """Return list of missing required columns (empty list if all present)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def run_preprocessing(df, verbose=True):
    """
    Full preprocessing pipeline. Returns (cleaned_df, stats).
    Never silently drops rows without recording it in `stats`.
    """
    stats = {}
    df = df.copy()

    stats["records_before"] = len(df)

    # 1. Date parsing / type conversion
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        stats["invalid_dates"] = int(df["date"].isna().sum())
        df = df.dropna(subset=["date"])
    else:
        stats["invalid_dates"] = 0

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Duplicate detection (exact duplicate rows, and duplicate date+city)
    exact_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()
    key_dupes = 0
    if {"date", "city"}.issubset(df.columns):
        key_dupes = int(df.duplicated(subset=["date", "city"]).sum())
        df = df.drop_duplicates(subset=["date", "city"], keep="first")
    stats["duplicates_removed"] = exact_dupes + key_dupes

    # 3. Missing-value analysis (before treatment)
    missing_before = df.isna().sum()
    stats["missing_values_by_column"] = {
        k: int(v) for k, v in missing_before[missing_before > 0].items()
    }
    stats["missing_values_total"] = int(missing_before.sum())

    # 4. Invalid-value correction (clip physically impossible values to NaN,
    #    they will then be imputed like any other missing value)
    invalid_count = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            invalid_count += int(mask.sum())
            df.loc[mask, col] = np.nan
    stats["invalid_values_corrected"] = invalid_count

    # 5. Outlier detection using IQR, computed PER CITY+SEASON so that normal
    #    seasonal extremes (e.g. summer heat) are not mistaken for outliers.
    #    Flag, do not delete -- cap to the group's bound instead.
    outliers_capped = 0
    outlier_detail_cols = ["temperature", "wind_speed", "pressure", "rainfall", "humidity"]
    group_cols = [c for c in ["city", "season"] if c in df.columns]
    for col in outlier_detail_cols:
        if col not in df.columns or df[col].notna().sum() <= 10:
            continue
        if group_cols:
            def _cap(s):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
                return s.clip(lower, upper)

            before = df[col].copy()
            df[col] = df.groupby(group_cols)[col].transform(_cap)
            changed = (before != df[col]) & before.notna() & df[col].notna()
            outliers_capped += int(changed.sum())
        else:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            outliers_capped += int(mask.sum())
            df[col] = df[col].clip(lower, upper)
    stats["outliers_capped"] = outliers_capped

    # 6. Missing-value treatment
    #    Numeric: group-wise median (by city + month) falls back to global median
    if {"city", "month"}.issubset(df.columns):
        for col in NUMERIC_COLS:
            if col in df.columns and df[col].isna().any():
                df[col] = df.groupby(["city", "month"])[col].transform(
                    lambda s: s.fillna(s.median())
                )
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical: fill with mode
    for col in ["weather_condition", "season"]:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(fill_val)

    stats["missing_values_after"] = int(df.isna().sum().sum())
    stats["records_after"] = len(df)

    if verbose:
        print(f"Records before preprocessing : {stats['records_before']}")
        print(f"Duplicates removed           : {stats['duplicates_removed']}")
        print(f"Invalid values corrected     : {stats['invalid_values_corrected']}")
        print(f"Outliers capped (IQR)        : {stats['outliers_capped']}")
        print(f"Missing values (before)      : {stats['missing_values_total']}")
        print(f"Missing values (after fill)  : {stats['missing_values_after']}")
        print(f"Records after preprocessing  : {stats['records_after']}")

    return df, stats


if __name__ == "__main__":
    df = load_dataset("data/weather_data.csv")
    cleaned, stats = run_preprocessing(df)
    cleaned.to_csv("data/weather_cleaned.csv", index=False)
    print("\nSaved cleaned dataset to data/weather_cleaned.csv")
