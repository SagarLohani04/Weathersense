"""
train_models.py
WeatherSense AI - Master Training Orchestrator

Runs the complete pipeline end-to-end:
  raw data -> preprocessing -> feature engineering -> EDA stats ->
  pattern mining -> clustering -> anomaly detection -> classification ->
  regression -> rainfall prediction -> explainability -> risk-engine demo

Saves all trained models to models/*.pkl (joblib) and a consolidated
metrics/results JSON to data/model_results.json, which the Django app
loads at request time instead of retraining on every page view.

Run:
    python train_models.py
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ml import preprocessing
from ml import feature_engineering
from ml import clustering
from ml import anomaly_detection
from ml import association_rules
from ml import classification
from ml import regression
from ml import rainfall_prediction
from ml import explainability

RAW_PATH = "data/weather_data.csv"
CLEANED_PATH = "data/weather_cleaned.csv"
FEATURES_PATH = "data/weather_features.csv"
MODELS_DIR = "models"
RESULTS_PATH = "data/model_results.json"


def explain_best_model(results_dict, best_name, metric_key, higher_is_better=True, metric_label=None):
    """Builds a natural-language explanation of why the winning model was
    chosen, computed from the ACTUAL metric values -- never hard-coded."""
    metric_label = metric_label or metric_key.upper()
    sorted_items = sorted(
        results_dict.items(), key=lambda kv: kv[1][metric_key],
        reverse=higher_is_better,
    )
    best_val = results_dict[best_name][metric_key]
    runner_up_name, runner_up_vals = sorted_items[1] if len(sorted_items) > 1 else (None, None)
    worst_name, worst_vals = sorted_items[-1]
    best_time = results_dict[best_name].get("training_time_sec")

    parts = [f"{best_name} achieved the best {metric_label} ({best_val})."]
    if runner_up_name:
        gap = abs(best_val - runner_up_vals[metric_key])
        parts.append(
            f"It edged out the runner-up, {runner_up_name} ({runner_up_vals[metric_key]}), "
            f"by {gap:.4f}, and outperformed the weakest model, {worst_name} "
            f"({worst_vals[metric_key]}), by a clear margin."
        )
    if best_time is not None:
        parts.append(f"It trained in {best_time}s on this dataset.")
    return " ".join(parts)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def to_native(obj):
    """Recursively convert numpy/pandas types to plain Python for JSON dumping."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    return obj


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {"generated_at": pd.Timestamp.now().isoformat()}

    # ---------------------------------------------------------------- #
    section("PHASE 1-2: Loading raw data & running preprocessing pipeline")
    raw_df = preprocessing.load_dataset(RAW_PATH)
    cleaned_df, prep_stats = preprocessing.run_preprocessing(raw_df)
    cleaned_df.to_csv(CLEANED_PATH, index=False)
    results["preprocessing"] = prep_stats

    # ---------------------------------------------------------------- #
    section("PHASE 2b: Feature engineering")
    feat_df = feature_engineering.engineer_features(cleaned_df)
    feat_df.to_csv(FEATURES_PATH, index=False)
    results["feature_engineering"] = {
        "columns_before": int(cleaned_df.shape[1]),
        "columns_after": int(feat_df.shape[1]),
        "records": int(feat_df.shape[0]),
    }

    # ---------------------------------------------------------------- #
    section("PHASE 3: Exploratory Data Analysis (summary statistics)")
    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed",
                     "rainfall", "cloud_cover", "visibility", "air_quality_index"]
    eda = {
        "summary_stats": feat_df[numeric_cols].describe().round(2).to_dict(),
        "correlation_matrix": feat_df[numeric_cols].corr().round(3).to_dict(),
        "monthly_avg_temperature": feat_df.groupby("month")["temperature"].mean().round(2).to_dict(),
        "monthly_avg_rainfall": feat_df.groupby("month")["rainfall"].mean().round(2).to_dict(),
        "seasonal_avg_temperature": feat_df.groupby("season")["temperature"].mean().round(2).to_dict(),
        "seasonal_avg_rainfall": feat_df.groupby("season")["rainfall"].mean().round(2).to_dict(),
        "city_avg_stats": feat_df.groupby("city")[numeric_cols].mean().round(2).to_dict(orient="index"),
        "condition_distribution": feat_df["weather_condition"].value_counts().to_dict(),
    }
    corr = feat_df[numeric_cols].corr()
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            pairs.append((c1, c2, corr.loc[c1, c2]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    eda["strongest_correlations"] = [
        {"var1": p[0], "var2": p[1], "correlation": round(float(p[2]), 3)} for p in pairs[:10]
    ]
    results["eda"] = eda
    print(f"Computed EDA summary over {len(feat_df)} records.")

    # ---------------------------------------------------------------- #
    section("PHASE 5a: Weather pattern mining (association rules)")
    rules = association_rules.run_association_mining(feat_df)
    rules.to_csv("data/association_rules.csv", index=False)
    results["association_rules"] = rules.head(25).to_dict(orient="records")
    print(f"Discovered {len(rules)} rules (showing top 25 in dashboard).")

    # ---------------------------------------------------------------- #
    section("PHASE 5b: Clustering (K-Means + PCA)")
    cluster_data, cluster_model, cluster_scaler, cluster_result = clustering.run_clustering(feat_df)
    joblib.dump({"model": cluster_model, "scaler": cluster_scaler, "features": clustering.CLUSTER_FEATURES},
                f"{MODELS_DIR}/clustering_model.pkl")
    # keep a light sample of points for the PCA scatter chart (avoid huge JSON)
    sample_pts = cluster_data.sample(min(1500, len(cluster_data)), random_state=42)
    results["clustering"] = {
        "best_k": cluster_result["best_k"],
        "silhouette_scores": cluster_result["silhouette_scores"],
        "cluster_names": cluster_result["cluster_names"],
        "cluster_stats": cluster_result["cluster_stats"],
        "cluster_sizes": cluster_result["cluster_sizes"],
        "pca_sample": {
            "x": sample_pts["pca_x"].round(3).tolist(),
            "y": sample_pts["pca_y"].round(3).tolist(),
            "cluster": sample_pts["cluster_name"].tolist(),
        },
    }
    print(f"Best K = {cluster_result['best_k']}  Names = {cluster_result['cluster_names']}")

    # ---------------------------------------------------------------- #
    section("PHASE 5c: Anomaly detection")
    anomalies_df, anomaly_summary, anomaly_model, anomaly_scaler = anomaly_detection.run_anomaly_detection(feat_df)
    anomalies_df.to_csv("data/anomalies_detected.csv", index=False)
    joblib.dump({"model": anomaly_model, "scaler": anomaly_scaler, "features": anomaly_detection.ANOMALY_FEATURES},
                f"{MODELS_DIR}/anomaly_model.pkl")
    results["anomaly_detection"] = {
        "summary": anomaly_summary,
        "sample_anomalies": anomalies_df.head(30).to_dict(orient="records"),
    }
    print(f"Anomalies found: {anomaly_summary['total_anomalies']} ({anomaly_summary['anomaly_rate_pct']}%)")

    # ---------------------------------------------------------------- #
    section("PHASE 4a: Weather condition classification")
    clf_models, clf_scaler, clf_le, clf_output = classification.train_and_compare(feat_df)
    best_clf_name = clf_output["best_model_name"]
    joblib.dump(
        {"model": clf_models[best_clf_name], "scaler": clf_scaler, "label_encoder": clf_le,
         "features": classification.FEATURES, "model_name": best_clf_name},
        f"{MODELS_DIR}/weather_classifier.pkl",
    )
    # Save EVERY trained classifier (not just the winner) so the Algorithm
    # Showdown page can run every algorithm on the same input and compare.
    joblib.dump(
        {"models": clf_models, "scaler": clf_scaler, "label_encoder": clf_le,
         "features": classification.FEATURES, "best_model_name": best_clf_name},
        f"{MODELS_DIR}/all_classifiers.pkl",
    )
    results["classification"] = clf_output
    print(f"Best classifier: {best_clf_name}  (F1={clf_output['results'][best_clf_name]['f1_score']})")

    # ---------------------------------------------------------------- #
    section("PHASE 4b: Next-day maximum temperature regression")
    reg_models, reg_scaler, reg_output = regression.train_and_compare(feat_df)
    best_reg_name = reg_output["best_model_name"]
    joblib.dump(
        {"model": reg_models[best_reg_name], "scaler": reg_scaler,
         "features": regression.FEATURES, "model_name": best_reg_name},
        f"{MODELS_DIR}/temperature_model.pkl",
    )
    joblib.dump(
        {"models": reg_models, "scaler": reg_scaler,
         "features": regression.FEATURES, "best_model_name": best_reg_name},
        f"{MODELS_DIR}/all_regressors.pkl",
    )
    results["regression"] = reg_output
    print(f"Best regressor: {best_reg_name}  (RMSE={reg_output['results'][best_reg_name]['rmse']})")

    # ---------------------------------------------------------------- #
    section("PHASE 4c: Rainfall next-day prediction")
    rain_models, rain_scaler, rain_output = rainfall_prediction.train_and_compare(feat_df)
    best_rain_name = rain_output["best_model_name"]
    joblib.dump(
        {"model": rain_models[best_rain_name], "scaler": rain_scaler,
         "features": rainfall_prediction.FEATURES, "model_name": best_rain_name},
        f"{MODELS_DIR}/rainfall_model.pkl",
    )
    joblib.dump(
        {"models": rain_models, "scaler": rain_scaler,
         "features": rainfall_prediction.FEATURES, "best_model_name": best_rain_name},
        f"{MODELS_DIR}/all_rainfall_models.pkl",
    )
    results["rainfall_prediction"] = rain_output
    print(f"Best rainfall model: {best_rain_name}  (F1={rain_output['results'][best_rain_name]['f1_score']})")

    # ---------------------------------------------------------------- #
    section("PHASE 7: Explainable AI (SHAP on rainfall model)")
    rain_bundle = joblib.load(f"{MODELS_DIR}/rainfall_model.pkl")
    sample_df = feat_df.dropna(subset=rain_bundle["features"]).sample(
        min(400, len(feat_df)), random_state=42
    )
    X_sample_scaled = rain_bundle["scaler"].transform(sample_df[rain_bundle["features"]])
    importance, used_shap = explainability.global_feature_importance(rain_bundle, X_sample_scaled)
    results["explainability"] = {
        "used_shap": used_shap,
        "global_feature_importance": importance,
        "model_explained": best_rain_name,
    }
    print(f"SHAP available and used: {used_shap}")

    # ---------------------------------------------------------------- #
    section("PHASE 9: Model comparison summary (all tasks)")
    results["model_comparison_summary"] = {
        "classification": clf_output["results"],
        "regression": reg_output["results"],
        "rainfall_prediction": rain_output["results"],
    }

    # Natural-language "why this model wins" explanations, generated from
    # the actual metric values above -- surfaced on the Algorithm Showdown page.
    results["algorithm_showdown"] = {
        "classification": {
            "best_model_name": best_clf_name,
            "explanation": explain_best_model(
                clf_output["results"], best_clf_name, "f1_score",
                higher_is_better=True, metric_label="weighted F1 score"
            ),
        },
        "regression": {
            "best_model_name": best_reg_name,
            "explanation": explain_best_model(
                reg_output["results"], best_reg_name, "rmse",
                higher_is_better=False, metric_label="RMSE"
            ),
        },
        "rainfall_prediction": {
            "best_model_name": best_rain_name,
            "explanation": explain_best_model(
                rain_output["results"], best_rain_name, "f1_score",
                higher_is_better=True, metric_label="weighted F1 score"
            ),
        },
    }

    # ---------------------------------------------------------------- #
    section("Saving consolidated results JSON")
    with open(RESULTS_PATH, "w") as f:
        json.dump(to_native(results), f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")

    # scaler used generically for quick numeric scaling on the prediction form
    joblib.dump({"note": "individual scalers are embedded per-model bundle"}, f"{MODELS_DIR}/scaler.pkl")

    section("TRAINING COMPLETE")
    print("All models saved to models/*.pkl")
    print("All metrics saved to data/model_results.json")
    print("The Django app will load these directly (no retraining at request time).")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal pipeline time: {round(time.time() - t0, 1)}s")
