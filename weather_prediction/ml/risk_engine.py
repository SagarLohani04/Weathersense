"""
ml/risk_engine.py
WeatherSense AI - Extreme Weather Risk Engine

Rule-based (with model-derived probabilities feeding in where available)
risk scoring for Heatwave, Heavy Rainfall, Flood, Drought and Storm
conditions. Outputs a LOW/MODERATE/HIGH/CRITICAL level plus the
contributing factors. These are explicitly labeled as model-based
indicators for a student data-mining project, not official warnings.
"""

import numpy as np
import pandas as pd


def _level_from_score(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"


def heatwave_risk(temperature, feels_like, humidity, consecutive_hot_days=0):
    score = 0
    factors = []
    if temperature > 42:
        score += 40; factors.append(f"Very high temperature ({temperature}°C)")
    elif temperature > 38:
        score += 25; factors.append(f"High temperature ({temperature}°C)")
    elif temperature > 34:
        score += 10; factors.append(f"Elevated temperature ({temperature}°C)")

    if feels_like and feels_like > temperature + 3:
        score += 15; factors.append(f"Feels-like temperature much higher ({feels_like}°C)")

    if humidity is not None and humidity < 30 and temperature > 35:
        score += 10; factors.append("Low humidity intensifying heat stress")
    elif humidity is not None and humidity > 60 and temperature > 35:
        score += 15; factors.append("High humidity increasing apparent heat")

    if consecutive_hot_days >= 3:
        score += 20; factors.append(f"{consecutive_hot_days} consecutive hot days")

    return _level_from_score(min(score, 100)), min(score, 100), factors


def heavy_rainfall_risk(rainfall, humidity, cloud_cover, pressure, rainfall_lag_1=0):
    score = 0
    factors = []
    if rainfall > 100:
        score += 40; factors.append(f"Very heavy rainfall recorded ({rainfall} mm)")
    elif rainfall > 50:
        score += 25; factors.append(f"Heavy rainfall recorded ({rainfall} mm)")
    elif rainfall > 20:
        score += 12; factors.append(f"Moderate rainfall recorded ({rainfall} mm)")

    if humidity > 85:
        score += 15; factors.append(f"Very high humidity ({humidity}%)")
    if cloud_cover > 80:
        score += 15; factors.append(f"Near-total cloud cover ({cloud_cover}%)")
    if pressure < 1005:
        score += 15; factors.append(f"Low atmospheric pressure ({pressure} hPa)")
    if rainfall_lag_1 and rainfall_lag_1 > 30:
        score += 10; factors.append("Significant rainfall the previous day")

    return _level_from_score(min(score, 100)), min(score, 100), factors


def flood_risk(rainfall_7_day_total, rainfall, rainfall_lag_1=0):
    score = 0
    factors = []
    if rainfall_7_day_total > 300:
        score += 45; factors.append(f"Very high 7-day rainfall total ({rainfall_7_day_total} mm)")
    elif rainfall_7_day_total > 150:
        score += 25; factors.append(f"High 7-day rainfall total ({rainfall_7_day_total} mm)")
    elif rainfall_7_day_total > 80:
        score += 12; factors.append(f"Elevated 7-day rainfall total ({rainfall_7_day_total} mm)")

    if rainfall > 60:
        score += 25; factors.append(f"Intense rainfall today ({rainfall} mm)")
    if rainfall_lag_1 and rainfall_lag_1 > 40 and rainfall > 40:
        score += 20; factors.append("Consecutive days of intense rainfall (saturated ground)")

    return _level_from_score(min(score, 100)), min(score, 100), factors


def drought_risk(rolling_rainfall_7, temperature, consecutive_dry_days=0):
    score = 0
    factors = []
    if consecutive_dry_days >= 30:
        score += 40; factors.append(f"{consecutive_dry_days} consecutive dry days")
    elif consecutive_dry_days >= 14:
        score += 20; factors.append(f"{consecutive_dry_days} consecutive dry days")

    if rolling_rainfall_7 is not None and rolling_rainfall_7 < 1:
        score += 25; factors.append("Negligible rainfall over the last 7 days")
    if temperature > 36:
        score += 20; factors.append(f"High temperature accelerating soil moisture loss ({temperature}°C)")

    return _level_from_score(min(score, 100)), min(score, 100), factors


def storm_risk(wind_speed, pressure, precipitation, cloud_cover):
    score = 0
    factors = []
    if wind_speed > 50:
        score += 40; factors.append(f"Very high wind speed ({wind_speed} km/h)")
    elif wind_speed > 35:
        score += 22; factors.append(f"High wind speed ({wind_speed} km/h)")
    elif wind_speed > 22:
        score += 10; factors.append(f"Elevated wind speed ({wind_speed} km/h)")

    if pressure < 1000:
        score += 20; factors.append(f"Low pressure system ({pressure} hPa)")
    if precipitation > 40:
        score += 15; factors.append(f"Heavy precipitation ({precipitation} mm)")
    if cloud_cover > 85:
        score += 10; factors.append(f"Dense cloud cover ({cloud_cover}%)")

    return _level_from_score(min(score, 100)), min(score, 100), factors


def assess_all_risks(row):
    """`row` is a dict-like with the relevant weather fields; missing fields
    default to safe/neutral values so the engine never crashes on partial input."""
    g = lambda k, d=0: row.get(k, d) if row.get(k, d) is not None else d

    hw_level, hw_score, hw_factors = heatwave_risk(
        g("temperature"), g("feels_like_temperature", g("temperature")),
        g("humidity"), g("consecutive_hot_days", 0)
    )
    hr_level, hr_score, hr_factors = heavy_rainfall_risk(
        g("rainfall"), g("humidity"), g("cloud_cover"), g("pressure", 1013), g("rainfall_lag_1", 0)
    )
    fl_level, fl_score, fl_factors = flood_risk(
        g("rainfall_7_day_total", g("rainfall") * 3), g("rainfall"), g("rainfall_lag_1", 0)
    )
    dr_level, dr_score, dr_factors = drought_risk(
        g("rolling_rainfall_7", g("rainfall")), g("temperature"), g("consecutive_dry_days", 0)
    )
    st_level, st_score, st_factors = storm_risk(
        g("wind_speed"), g("pressure", 1013), g("rainfall"), g("cloud_cover")
    )

    return {
        "heatwave": {"level": hw_level, "score": hw_score, "factors": hw_factors},
        "heavy_rainfall": {"level": hr_level, "score": hr_score, "factors": hr_factors},
        "flood": {"level": fl_level, "score": fl_score, "factors": fl_factors},
        "drought": {"level": dr_level, "score": dr_score, "factors": dr_factors},
        "storm": {"level": st_level, "score": st_score, "factors": st_factors},
        "disclaimer": ("These are model-based risk indicators generated for a data-mining "
                        "academic project and are NOT official emergency warnings."),
    }


if __name__ == "__main__":
    sample = {
        "temperature": 39.5, "feels_like_temperature": 42.0, "humidity": 28,
        "pressure": 1002, "wind_speed": 18, "cloud_cover": 20, "rainfall": 0,
        "rainfall_7_day_total": 0, "rolling_rainfall_7": 0, "consecutive_hot_days": 4,
        "consecutive_dry_days": 25,
    }
    risks = assess_all_risks(sample)
    for k, v in risks.items():
        if k == "disclaimer":
            continue
        print(f"{k:15s} -> {v['level']:9s} (score={v['score']})  factors={v['factors']}")
