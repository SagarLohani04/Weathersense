"""
generate_dataset.py
WeatherSense AI - Synthetic Weather Dataset Generator

Generates a realistic daily weather dataset for 10 Indian cities across
multiple years, preserving realistic physical relationships between
variables (temperature/humidity/pressure/rainfall/season etc.), and
injects a controlled amount of missing values and outliers so that the
preprocessing and anomaly-detection modules have genuine work to do.

Run:
    python generate_dataset.py

Output:
    data/weather_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------
# City metadata: (latitude, longitude, base_temp_summer, base_temp_winter,
# monsoon_intensity, coastal[bool])
# ----------------------------------------------------------------------
CITIES = {
    "Pune":       dict(lat=18.52, lon=73.86, summer=36, winter=13, monsoon=1.0, coastal=False),
    "Mumbai":     dict(lat=19.08, lon=72.88, summer=34, winter=19, monsoon=1.6, coastal=True),
    "Delhi":      dict(lat=28.70, lon=77.10, summer=41, winter=7,  monsoon=0.9, coastal=False),
    "Bengaluru":  dict(lat=12.97, lon=77.59, summer=32, winter=15, monsoon=0.8, coastal=False),
    "Hyderabad":  dict(lat=17.39, lon=78.49, summer=38, winter=15, monsoon=0.9, coastal=False),
    "Chennai":    dict(lat=13.08, lon=80.27, summer=37, winter=21, monsoon=1.1, coastal=True),
    "Kolkata":    dict(lat=22.57, lon=88.36, summer=36, winter=13, monsoon=1.3, coastal=True),
    "Jaipur":     dict(lat=26.91, lon=75.79, summer=40, winter=9,  monsoon=0.7, coastal=False),
    "Ahmedabad":  dict(lat=23.02, lon=72.57, summer=41, winter=13, monsoon=0.8, coastal=False),
    "Nagpur":     dict(lat=21.15, lon=79.09, summer=42, winter=12, monsoon=0.9, coastal=False),
}

START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 12, 31)
DATES = pd.date_range(START_DATE, END_DATE, freq="D")

WEATHER_CONDITIONS = ["Sunny", "Clear", "Cloudy", "Overcast", "Rainy", "Stormy", "Foggy"]


def season_of(month, day_of_year):
    """India-centric season classification."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def seasonal_temp_curve(day_of_year, summer_peak, winter_low):
    """Smooth annual temperature curve using a cosine wave, peak ~ mid-May (day 135)."""
    amplitude = (summer_peak - winter_low) / 2
    mean_temp = (summer_peak + winter_low) / 2
    # Shift so the peak occurs around day 135 (mid-May) and trough around day 15 (mid-Jan)
    phase = 2 * np.pi * (day_of_year - 135) / 365.25
    return mean_temp + amplitude * np.cos(phase)


def monsoon_factor(month, monsoon_intensity):
    """Rainfall likelihood multiplier by month, scaled by city's monsoon intensity."""
    base = {1: 0.02, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.10,
            6: 0.55, 7: 0.85, 8: 0.80, 9: 0.55,
            10: 0.20, 11: 0.05, 12: 0.02}[month]
    return base * monsoon_intensity


def generate_records():
    rows = []
    rng = np.random.default_rng(SEED)

    for city, meta in CITIES.items():
        # City-level persistent state for autocorrelation
        prev_temp = seasonal_temp_curve(1, meta["summer"], meta["winter"])
        prev_pressure = 1013.0
        prev_rainfall = 0.0
        consecutive_hot_days = 0
        consecutive_dry_days = 0

        for date in DATES:
            month = date.month
            day_of_year = date.timetuple().tm_yday
            season = season_of(month, day_of_year)

            # --- Temperature (autocorrelated around seasonal curve) ---
            seasonal_base = seasonal_temp_curve(day_of_year, meta["summer"], meta["winter"])
            temp_noise = rng.normal(0, 1.4)
            temperature = 0.65 * prev_temp + 0.35 * (seasonal_base) + temp_noise
            prev_temp = temperature

            # --- Pressure: inversely related to temperature/humidity, autocorrelated ---
            pressure_base = 1015 - (temperature - 25) * 0.35
            pressure = 0.7 * prev_pressure + 0.3 * pressure_base + rng.normal(0, 1.5)
            prev_pressure = pressure

            # --- Humidity: high in monsoon, correlated with rainfall & coastal proximity ---
            m_factor = monsoon_factor(month, meta["monsoon"])
            coastal_bonus = 12 if meta["coastal"] else 0
            humidity_base = 35 + 55 * m_factor + coastal_bonus - 0.3 * (temperature - 30)
            humidity = np.clip(humidity_base + rng.normal(0, 6), 8, 100)

            # --- Cloud cover: driven by humidity & monsoon ---
            cloud_cover = np.clip(0.55 * humidity + 25 * m_factor + rng.normal(0, 10), 0, 100)

            # --- Rainfall probability & amount: low pressure + high humidity + high cloud → rain ---
            pressure_drop_effect = np.clip((1013 - pressure) / 8, 0, 3)
            rain_probability = np.clip(
                0.75 * m_factor + 0.15 * (humidity / 100) + 0.10 * (cloud_cover / 100)
                + 0.05 * pressure_drop_effect, 0, 0.97
            )
            will_rain = rng.random() < rain_probability
            if will_rain:
                intensity_scale = 5 + 40 * m_factor + 10 * pressure_drop_effect
                precipitation = float(np.clip(rng.gamma(1.6, intensity_scale), 0, 250))
            else:
                precipitation = 0.0
            rainfall = precipitation

            # --- Wind speed: higher during storms / pressure drops ---
            storm_chance_base = 0.02 + 0.10 * m_factor + 0.05 * pressure_drop_effect
            wind_speed = np.clip(
                6 + 10 * pressure_drop_effect + rng.normal(0, 3) + (8 if rain_probability > 0.6 else 0) * rng.random(),
                0, 90
            )

            wind_direction = float(rng.uniform(0, 360))

            # --- Dew point derived from temperature & humidity (Magnus approximation) ---
            a, b = 17.27, 237.7
            alpha = (a * temperature) / (b + temperature) + np.log(max(humidity, 1) / 100.0)
            dew_point = (b * alpha) / (a - alpha)

            # --- Visibility: reduced by fog conditions (high humidity, low wind, cool temps) ---
            fog_likelihood = (
                1 if (humidity > 65 and wind_speed < 11 and 6 < temperature < 27 and month in (11, 12, 1, 2))
                else 0
            )
            fog_indicator = 1 if (fog_likelihood and rng.random() < 0.35) else 0
            visibility = np.clip(10 - (cloud_cover / 100) * 4 - (fog_indicator * 6) - (rainfall / 100) * 2
                                  + rng.normal(0, 0.5), 0.1, 10)

            # --- Solar radiation & UV: inversely related to cloud cover ---
            solar_radiation = np.clip(
                (250 + 5 * (temperature - 20)) * (1 - cloud_cover / 130) + rng.normal(0, 15), 0, 400
            )
            uv_index = np.clip(solar_radiation / 32 + rng.normal(0, 0.4), 0, 12)

            # --- Air Quality Index: worse in winter (inversion), better after rain ---
            aqi_base = 90 + (60 if season == "Winter" else 0) - (rainfall * 0.4) + rng.normal(0, 20)
            air_quality_index = np.clip(aqi_base, 15, 480)

            # --- Storm indicator ---
            storm_indicator = 1 if (wind_speed > 40 and rainfall > 30 and rng.random() < 0.8) else (
                1 if rng.random() < storm_chance_base * 0.15 else 0
            )

            # --- Heat index (feels-like) ---
            heat_index = temperature + 0.05 * humidity if temperature > 27 else temperature - 0.02 * (100 - humidity)
            feels_like_temperature = heat_index + rng.normal(0, 0.5)

            min_temperature = temperature - abs(rng.normal(4, 1.2))
            max_temperature = temperature + abs(rng.normal(4, 1.2))

            # --- Heatwave tracking (consecutive hot days) ---
            is_extreme_hot = temperature > (meta["summer"] - 1.5)
            consecutive_hot_days = consecutive_hot_days + 1 if is_extreme_hot else 0
            heatwave_indicator = 1 if consecutive_hot_days >= 3 else 0

            # --- Drought tracking (consecutive dry days) ---
            is_dry = rainfall < 1.0
            consecutive_dry_days = consecutive_dry_days + 1 if is_dry else 0
            drought_risk_flag = 1 if consecutive_dry_days >= 20 and season != "Winter" else 0

            # --- Flood risk proxy: heavy rainfall + already wet conditions ---
            flood_risk_flag = 1 if (rainfall > 90 and prev_rainfall > 30) else 0
            prev_rainfall = rainfall

            # --- Weather condition label (rule-based, derived from the physical vars) ---
            if storm_indicator == 1:
                condition = "Stormy"
            elif fog_indicator == 1:
                condition = "Foggy"
            elif rainfall > 40:
                condition = "Rainy"
            elif rainfall > 2:
                condition = "Rainy" if rng.random() < 0.8 else "Overcast"
            elif cloud_cover > 70:
                condition = "Overcast"
            elif cloud_cover > 40:
                condition = "Cloudy"
            elif cloud_cover > 15:
                condition = "Clear"
            else:
                condition = "Sunny"

            precipitation_probability = round(rain_probability * 100, 1)

            rows.append(dict(
                date=date.strftime("%Y-%m-%d"),
                city=city,
                latitude=meta["lat"],
                longitude=meta["lon"],
                season=season,
                month=month,
                day_of_year=day_of_year,
                temperature=round(temperature, 2),
                feels_like_temperature=round(feels_like_temperature, 2),
                min_temperature=round(min_temperature, 2),
                max_temperature=round(max_temperature, 2),
                humidity=round(humidity, 2),
                dew_point=round(dew_point, 2),
                pressure=round(pressure, 2),
                wind_speed=round(wind_speed, 2),
                wind_direction=round(wind_direction, 1),
                cloud_cover=round(cloud_cover, 2),
                visibility=round(visibility, 2),
                precipitation=round(precipitation, 2),
                precipitation_probability=precipitation_probability,
                solar_radiation=round(solar_radiation, 2),
                uv_index=round(uv_index, 2),
                weather_condition=condition,
                rainfall=round(rainfall, 2),
                heat_index=round(heat_index, 2),
                air_quality_index=round(air_quality_index, 1),
                storm_indicator=storm_indicator,
                fog_indicator=fog_indicator,
                heatwave_indicator=heatwave_indicator,
                flood_risk=flood_risk_flag,
                drought_risk=drought_risk_flag,
            ))

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "city"]).reset_index(drop=True)
    return df


def inject_missing_and_outliers(df, rng):
    """Introduce ~2-5% missing values in selected columns and a small number
    of realistic outliers, to give the preprocessing/anomaly modules real work."""
    df = df.copy()
    n = len(df)

    missing_cols = ["temperature", "humidity", "pressure", "wind_speed",
                     "visibility", "air_quality_index", "dew_point"]
    for col in missing_cols:
        frac = rng.uniform(0.02, 0.05)
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Outliers: implausible spikes in a handful of records, for anomaly detection to catch
    n_outliers = max(15, int(n * 0.0025))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    for i in outlier_idx:
        choice = rng.integers(0, 5)
        if choice == 0:
            df.loc[i, "temperature"] = df.loc[i, "temperature"] + rng.uniform(12, 18) * rng.choice([-1, 1])
        elif choice == 1:
            df.loc[i, "wind_speed"] = df.loc[i, "wind_speed"] + rng.uniform(60, 100)
        elif choice == 2:
            df.loc[i, "pressure"] = df.loc[i, "pressure"] - rng.uniform(25, 40)
        elif choice == 3:
            df.loc[i, "rainfall"] = df.loc[i, "rainfall"] + rng.uniform(250, 400)
        else:
            df.loc[i, "humidity"] = 100 + rng.uniform(5, 20)  # invalid > 100

    # A few duplicate rows (for duplicate-detection demo in preprocessing)
    dup_idx = rng.choice(n, size=max(10, int(n * 0.001)), replace=False)
    dup_rows = df.loc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main():
    print("=" * 60)
    print("WeatherSense AI - Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Cities: {len(CITIES)}  |  Date range: {START_DATE.date()} to {END_DATE.date()}")

    df = generate_records()
    print(f"Base records generated: {len(df)}")

    rng = np.random.default_rng(SEED + 1)
    df = inject_missing_and_outliers(df, rng)
    print(f"Records after injecting duplicates: {len(df)}")
    print(f"Missing values introduced: {df.isna().sum().sum()}")

    import os
    os.makedirs("data", exist_ok=True)
    out_path = "data/weather_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
    print(f"Final shape: {df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
