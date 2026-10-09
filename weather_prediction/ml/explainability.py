"""
ml/explainability.py
WeatherSense AI - Explainable AI (SHAP)

Generates SHAP-based feature importance and per-prediction explanations
for the strongest tree-based model (rainfall prediction, Random
Forest / XGBoost). Falls back gracefully to the model's native
feature_importances_ if SHAP is unavailable in the runtime, so the
core application never depends on it.

The generated natural-language explanations (e.g. "High humidity
increased the predicted rainfall probability") are derived directly
from the sign and magnitude of each feature's SHAP value -- never
hard-coded.
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

FRIENDLY_NAMES = {
    "temperature": "temperature",
    "humidity": "humidity",
    "pressure": "atmospheric pressure",
    "wind_speed": "wind speed",
    "cloud_cover": "cloud cover",
    "dew_point": "dew point",
    "rainfall": "today's rainfall",
    "rolling_rainfall_7": "recent 7-day rainfall trend",
    "rainfall_lag_1": "yesterday's rainfall",
    "rainfall_lag_3": "rainfall 3 days ago",
    "rainfall_7_day_total": "total rainfall over the last week",
    "month": "time of year",
    "sin_month": "seasonal cycle",
    "cos_month": "seasonal cycle",
}


def _select_positive_class(shap_values):
    """Normalize SHAP output across shap versions/model types into a single
    (n_samples, n_features) array representing the positive/target class."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> take the last class (positive class)
        return arr[:, :, -1]
    return arr


def global_feature_importance(model_bundle, X_sample):
    """Return {feature: mean_abs_shap} sorted descending. Falls back to
    the model's native importances if SHAP cannot run."""
    model = model_bundle["model"]
    features = model_bundle["features"]

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_values = _select_positive_class(shap_values)
            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = dict(zip(features, mean_abs))
            return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)), True
        except Exception:
            pass

    if hasattr(model, "feature_importances_"):
        importance = dict(zip(features, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)), False

    return {f: 0.0 for f in features}, False


def explain_single_prediction(model_bundle, input_row_scaled, input_row_raw):
    """Explain one prediction. `input_row_scaled` is a 2D array (1, n_features)
    already through the model's scaler; `input_row_raw` is the raw feature dict
    for readability in the generated text."""
    model = model_bundle["model"]
    features = model_bundle["features"]

    explanations = []
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_row_scaled)
            shap_values = _select_positive_class(shap_values)
            values = shap_values[0]
            pairs = list(zip(features, values))
            pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
            for feat, val in pairs[:5]:
                name = FRIENDLY_NAMES.get(feat, feat.replace("_", " "))
                direction = "increased" if val > 0 else "decreased"
                raw_val = input_row_raw.get(feat)
                explanations.append({
                    "feature": name,
                    "value": raw_val,
                    "impact": round(float(val), 4),
                    "direction": direction,
                    "text": f"{name.capitalize()} ({raw_val}) {direction} the predicted probability.",
                })
            return explanations, True
        except Exception:
            pass

    # Fallback: use feature importances (no directionality available)
    if hasattr(model, "feature_importances_"):
        pairs = list(zip(features, model.feature_importances_))
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        for feat, val in pairs[:5]:
            name = FRIENDLY_NAMES.get(feat, feat.replace("_", " "))
            explanations.append({
                "feature": name,
                "value": input_row_raw.get(feat),
                "impact": round(float(val), 4),
                "direction": "influential",
                "text": f"{name.capitalize()} is one of the most influential factors for this model.",
            })
    return explanations, False


if __name__ == "__main__":
    import joblib
    bundle = joblib.load("models/rainfall_model.pkl")
    df = pd.read_csv("data/weather_features.csv").dropna(subset=bundle["features"])
    sample = df[bundle["features"]].sample(min(300, len(df)), random_state=42)
    X_scaled = bundle["scaler"].transform(sample)

    importance, used_shap = global_feature_importance(bundle, X_scaled)
    print(f"SHAP available: {used_shap}")
    print("Top features driving rainfall prediction:")
    for feat, score in list(importance.items())[:8]:
        print(f"  {feat:25s} {score:.4f}")

    row = df.iloc[[0]]
    row_scaled = bundle["scaler"].transform(row[bundle["features"]])
    explanation, _ = explain_single_prediction(bundle, row_scaled, row.iloc[0].to_dict())
    print("\nSample explanation for one record:")
    for e in explanation:
        print("  -", e["text"])
