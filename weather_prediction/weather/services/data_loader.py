"""
weather/services/data_loader.py
WeatherSense AI - Cached data & model access layer

Loads the feature-engineered dataset and the consolidated
model_results.json ONCE per process (module-level cache) so that every
dashboard/page view is fast, rather than re-reading a 20,000+ row CSV
or retraining models on every request.
"""

import json
import threading

import joblib
import pandas as pd
from django.conf import settings

_lock = threading.Lock()
_cache = {}


def get_dataframe():
    if "df" not in _cache:
        with _lock:
            if "df" not in _cache:
                df = pd.read_csv(settings.WEATHER_FEATURES_CSV)
                df["date"] = pd.to_datetime(df["date"])
                _cache["df"] = df
    return _cache["df"]


def get_results():
    if "results" not in _cache:
        with _lock:
            if "results" not in _cache:
                if settings.MODEL_RESULTS_PATH.exists():
                    with open(settings.MODEL_RESULTS_PATH) as f:
                        _cache["results"] = json.load(f)
                else:
                    _cache["results"] = {}
    return _cache["results"]


def get_model(name):
    """name: one of rainfall_model, temperature_model, weather_classifier,
    clustering_model, anomaly_model -- loaded once and cached."""
    key = f"model:{name}"
    if key not in _cache:
        with _lock:
            if key not in _cache:
                path = settings.MODELS_DIR / f"{name}.pkl"
                _cache[key] = joblib.load(path) if path.exists() else None
    return _cache[key]


def get_cities():
    df = get_dataframe()
    return sorted(df["city"].unique().tolist())


def get_years():
    df = get_dataframe()
    return sorted(df["date"].dt.year.unique().tolist())


def clear_cache():
    """Used by the upload feature to force a refresh if needed."""
    _cache.clear()
