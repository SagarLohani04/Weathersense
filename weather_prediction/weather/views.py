"""
weather/views.py
WeatherSense AI - Views for all dashboard pages
"""

import io
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render

from .forms import WeatherPredictionForm, DatasetUploadForm, ForecastForm
from .models import Prediction
from .services import data_loader, analytics

PROJECT_ROOT = str(settings.BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml import forecasting as ml_forecasting          # noqa: E402
from ml import risk_engine                             # noqa: E402
from ml import explainability                          # noqa: E402
from ml import preprocessing as ml_preprocessing        # noqa: E402


# --------------------------------------------------------------------- #
# 1. DASHBOARD
# --------------------------------------------------------------------- #
def dashboard_view(request):
    df = data_loader.get_dataframe()
    results = data_loader.get_results()

    city = request.GET.get("city") or None
    year = request.GET.get("year") or None
    month = request.GET.get("month") or None
    season = request.GET.get("season") or None
    condition = request.GET.get("condition") or None

    filtered = analytics.apply_filters(df, city, year, month, season, condition)
    if filtered.empty:
        filtered = df

    kpis = analytics.dashboard_kpis(filtered)
    temp_trend = analytics.trend_series(filtered, "temperature")
    rainfall_trend = analytics.trend_series(filtered, "rainfall")
    humidity_trend = analytics.trend_series(filtered, "humidity")
    wind_trend = analytics.trend_series(filtered, "wind_speed")
    condition_dist = analytics.weather_condition_distribution(filtered)

    city_compare = analytics.city_comparison_data(df, data_loader.get_cities()[:6])

    recent_anomalies = results.get("anomaly_detection", {}).get("sample_anomalies", [])[:8]

    context = {
        "kpis": kpis,
        "temp_trend": json.dumps(temp_trend),
        "rainfall_trend": json.dumps(rainfall_trend),
        "humidity_trend": json.dumps(humidity_trend),
        "wind_trend": json.dumps(wind_trend),
        "condition_dist": json.dumps(condition_dist),
        "city_compare": json.dumps(city_compare),
        "recent_anomalies": recent_anomalies,
        "cities": data_loader.get_cities(),
        "years": data_loader.get_years(),
        "seasons": ["Winter", "Summer", "Monsoon", "Post-Monsoon"],
        "conditions": sorted(df["weather_condition"].unique().tolist()),
        "selected": {"city": city, "year": year, "month": month, "season": season, "condition": condition},
        "active_page": "dashboard",
    }
    return render(request, "dashboard.html", context)


# --------------------------------------------------------------------- #
# 2. WEATHER PREDICTION
# --------------------------------------------------------------------- #
def _build_full_feature_row(df, city, date_obj, form_data):
    """Build a full feature row for the ML models: uses the user's form inputs
    for the fields they specified, and fills every other required feature with
    the historical climatological average for that city + month (so the models,
    which expect many engineered features, always receive a complete vector)."""
    month = date_obj.month
    clim = df[(df["city"] == city) & (df["month"] == month)]
    if clim.empty:
        clim = df[df["city"] == city]
    if clim.empty:
        clim = df

    row = clim.mean(numeric_only=True).to_dict()
    row.update({
        "temperature": form_data["temperature"],
        "humidity": form_data["humidity"],
        "pressure": form_data["pressure"],
        "wind_speed": form_data["wind_speed"],
        "cloud_cover": form_data["cloud_cover"],
        "rainfall": form_data["rainfall"],
        "dew_point": form_data["dew_point"],
        "month": month,
        "day_of_year": date_obj.timetuple().tm_yday,
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "sin_day_of_year": np.sin(2 * np.pi * date_obj.timetuple().tm_yday / 365.25),
        "cos_day_of_year": np.cos(2 * np.pi * date_obj.timetuple().tm_yday / 365.25),
    })
    row["feels_like_temperature"] = row["temperature"] + 0.05 * row["humidity"] if row["temperature"] > 27 \
        else row["temperature"] - 0.02 * (100 - row["humidity"])
    return row


def prediction_view(request):
    cities = data_loader.get_cities()
    df = data_loader.get_dataframe()

    result = None
    if request.method == "POST":
        form = WeatherPredictionForm(request.POST, cities=cities)
        if form.is_valid():
            fd = form.cleaned_data
            row = _build_full_feature_row(df, fd["city"], fd["date"], fd)

            # --- Weather condition classification ---
            clf_bundle = data_loader.get_model("weather_classifier")
            clf_features = clf_bundle["features"]
            X_clf = pd.DataFrame([{f: row.get(f, 0) for f in clf_features}])
            X_clf_scaled = clf_bundle["scaler"].transform(X_clf)
            condition_pred = clf_bundle["label_encoder"].inverse_transform(
                clf_bundle["model"].predict(X_clf_scaled)
            )[0]
            condition_proba = float(np.max(clf_bundle["model"].predict_proba(X_clf_scaled)))

            # --- Rainfall next-day probability ---
            rain_bundle = data_loader.get_model("rainfall_model")
            rain_features = rain_bundle["features"]
            X_rain = pd.DataFrame([{f: row.get(f, 0) for f in rain_features}])
            X_rain_scaled = rain_bundle["scaler"].transform(X_rain)
            rain_proba = float(rain_bundle["model"].predict_proba(X_rain_scaled)[0][1])
            rain_pred = "RAIN" if rain_proba >= 0.5 else "NO RAIN"

            explanation, used_shap = explainability.explain_single_prediction(
                rain_bundle, X_rain_scaled, row
            )

            # --- Next-day max temperature regression ---
            temp_bundle = data_loader.get_model("temperature_model")
            temp_features = temp_bundle["features"]
            X_temp = pd.DataFrame([{f: row.get(f, 0) for f in temp_features}])
            X_temp_scaled = temp_bundle["scaler"].transform(X_temp)
            temp_pred = float(temp_bundle["model"].predict(X_temp_scaled)[0])

            # --- Risk engine ---
            risks = risk_engine.assess_all_risks(row)

            result = {
                "city": fd["city"],
                "date": fd["date"],
                "condition": condition_pred,
                "condition_confidence": round(condition_proba * 100, 1),
                "rain_probability": round(rain_proba * 100, 1),
                "rain_prediction": rain_pred,
                "temperature_prediction": round(temp_pred, 1),
                "risks": risks,
                "explanation": explanation,
                "used_shap": used_shap,
            }

            Prediction.objects.create(
                city=fd["city"],
                prediction_type="weather_condition",
                input_features={k: fd[k] for k in fd if k not in ("city", "date")},
                predicted_value=f"{condition_pred} | {rain_pred} ({round(rain_proba*100,1)}%) | {round(temp_pred,1)}C",
                model_used=f"{clf_bundle['model_name']} / {rain_bundle['model_name']} / {temp_bundle['model_name']}",
                confidence=round(condition_proba, 3),
            )
    else:
        form = WeatherPredictionForm(cities=cities, initial={"date": datetime.now().date()})

    history = Prediction.objects.filter(prediction_type="weather_condition").order_by("-timestamp")[:10]

    return render(request, "prediction.html", {
        "form": form, "result": result, "history": history, "active_page": "prediction",
    })


# --------------------------------------------------------------------- #
# 2b. ALGORITHM SHOWDOWN -- compare every trained algorithm on one input
# --------------------------------------------------------------------- #
def algorithm_showdown_view(request):
    cities = data_loader.get_cities()
    df = data_loader.get_dataframe()
    results = data_loader.get_results()
    showdown_meta = results.get("algorithm_showdown", {})
    metrics_summary = results.get("model_comparison_summary", {})

    showdown = None
    if request.method == "POST":
        form = WeatherPredictionForm(request.POST, cities=cities)
        if form.is_valid():
            fd = form.cleaned_data
            row = _build_full_feature_row(df, fd["city"], fd["date"], fd)

            # ---- Classification: run EVERY trained algorithm ----
            clf_bundle = data_loader.get_model("all_classifiers")
            clf_rows = []
            if clf_bundle:
                features = clf_bundle["features"]
                X = pd.DataFrame([{f: row.get(f, 0) for f in features}])
                X_scaled = clf_bundle["scaler"].transform(X)
                for name, model in clf_bundle["models"].items():
                    pred_idx = model.predict(X_scaled)[0]
                    pred_label = clf_bundle["label_encoder"].inverse_transform([pred_idx])[0]
                    proba = float(np.max(model.predict_proba(X_scaled)))
                    task_metrics = metrics_summary.get("classification", {}).get(name, {})
                    clf_rows.append({
                        "name": name, "prediction": pred_label,
                        "confidence": round(proba * 100, 1),
                        "f1_score": task_metrics.get("f1_score"),
                        "is_best": name == clf_bundle["best_model_name"],
                    })
            clf_rows.sort(key=lambda r: r["confidence"], reverse=True)

            # ---- Regression (temperature): run EVERY trained algorithm ----
            reg_bundle = data_loader.get_model("all_regressors")
            reg_rows = []
            if reg_bundle:
                features = reg_bundle["features"]
                X = pd.DataFrame([{f: row.get(f, 0) for f in features}])
                X_scaled = reg_bundle["scaler"].transform(X)
                for name, model in reg_bundle["models"].items():
                    pred_val = float(model.predict(X_scaled)[0])
                    task_metrics = metrics_summary.get("regression", {}).get(name, {})
                    reg_rows.append({
                        "name": name, "prediction": round(pred_val, 2),
                        "rmse": task_metrics.get("rmse"),
                        "is_best": name == reg_bundle["best_model_name"],
                    })
            reg_rows.sort(key=lambda r: (r["rmse"] if r["rmse"] is not None else 999))
            reg_values = [r["prediction"] for r in reg_rows]
            reg_spread = round(max(reg_values) - min(reg_values), 2) if reg_values else 0

            # ---- Rainfall prediction: run EVERY trained algorithm ----
            rain_bundle = data_loader.get_model("all_rainfall_models")
            rain_rows = []
            if rain_bundle:
                features = rain_bundle["features"]
                X = pd.DataFrame([{f: row.get(f, 0) for f in features}])
                X_scaled = rain_bundle["scaler"].transform(X)
                for name, model in rain_bundle["models"].items():
                    proba = float(model.predict_proba(X_scaled)[0][1])
                    task_metrics = metrics_summary.get("rainfall_prediction", {}).get(name, {})
                    rain_rows.append({
                        "name": name, "probability": round(proba * 100, 1),
                        "prediction": "RAIN" if proba >= 0.5 else "NO RAIN",
                        "f1_score": task_metrics.get("f1_score"),
                        "is_best": name == rain_bundle["best_model_name"],
                    })
            rain_rows.sort(key=lambda r: r["probability"], reverse=True)
            rain_values = [r["probability"] for r in rain_rows]
            rain_spread = round(max(rain_values) - min(rain_values), 1) if rain_values else 0

            # ---- Agreement stats ----
            distinct_conditions = len(set(r["prediction"] for r in clf_rows))
            agreement_pct = round(
                100 * max([sum(1 for r in clf_rows if r["prediction"] == c)
                           for c in set(r["prediction"] for r in clf_rows)]) / len(clf_rows), 1
            ) if clf_rows else 0

            showdown = {
                "city": fd["city"], "date": str(fd["date"]),
                "classification": clf_rows,
                "regression": reg_rows,
                "rainfall": rain_rows,
                "reg_spread": reg_spread,
                "rain_spread": rain_spread,
                "distinct_conditions": distinct_conditions,
                "agreement_pct": agreement_pct,
            }
    else:
        form = WeatherPredictionForm(cities=cities, initial={"date": datetime.now().date()})

    return render(request, "algorithm_showdown.html", {
        "form": form, "showdown": showdown,
        "showdown_meta": showdown_meta,
        "showdown_json": json.dumps(showdown) if showdown else None,
        "active_page": "algorithm_showdown",
    })


# --------------------------------------------------------------------- #
# 3. FORECAST
# --------------------------------------------------------------------- #
def forecast_view(request):
    cities = data_loader.get_cities()
    df = data_loader.get_dataframe()

    city = request.GET.get("city", cities[0] if cities else None)
    variable = request.GET.get("variable", "temperature")
    horizon = int(request.GET.get("horizon", 14))

    form = ForecastForm(cities=cities, initial={"city": city, "variable": variable, "horizon": horizon})

    forecast_result = None
    error = None
    if city:
        try:
            forecast_result = ml_forecasting.forecast_city(df, city, variable, horizon=horizon)
            risk_rows = []
            for i, val in enumerate(forecast_result["forecast"]):
                sample = {"temperature": val if variable == "temperature" else 28,
                          "rainfall": val if variable == "rainfall" else 5,
                          "humidity": val if variable == "humidity" else 55,
                          "wind_speed": 12, "pressure": 1010, "cloud_cover": 40,
                          "rainfall_7_day_total": val * 3 if variable == "rainfall" else 15,
                          "rolling_rainfall_7": val if variable == "rainfall" else 5}
                risks = risk_engine.assess_all_risks(sample)
                overall = max(
                    [risks["heavy_rainfall"]["score"], risks["flood"]["score"],
                     risks["heatwave"]["score"], risks["storm"]["score"]]
                )
                level = "LOW"
                if overall >= 75:
                    level = "CRITICAL"
                elif overall >= 50:
                    level = "HIGH"
                elif overall >= 25:
                    level = "MODERATE"
                risk_rows.append({"date": forecast_result["dates"][i], "value": val, "risk_level": level})
            forecast_result["risk_rows"] = risk_rows
        except Exception as e:
            error = f"Forecast could not be generated for this selection: {e}"

    return render(request, "forecast.html", {
        "form": form, "forecast_result": json.dumps(forecast_result) if forecast_result else None,
        "forecast_table": forecast_result, "error": error,
        "city": city, "variable": variable, "horizon": horizon,
        "active_page": "forecast",
    })


# --------------------------------------------------------------------- #
# 4. DATA MINING / EDA + SEASONAL + TREND ANALYSIS
# --------------------------------------------------------------------- #
def analysis_view(request):
    df = data_loader.get_dataframe()
    results = data_loader.get_results()
    eda = results.get("eda", {})

    cities = data_loader.get_cities()
    trend_city = request.GET.get("trend_city", cities[0] if cities else None)
    trend_var = request.GET.get("trend_var", "temperature")

    trend_data = analytics.trend_analysis(df, trend_city, trend_var) if trend_city else None
    seasonal = analytics.seasonal_analysis(df)

    return render(request, "analysis.html", {
        "eda": eda,
        "seasonal": seasonal,
        "trend_data": json.dumps(trend_data) if trend_data else None,
        "trend_summary": trend_data,
        "cities": cities,
        "trend_city": trend_city,
        "trend_var": trend_var,
        "active_page": "analysis",
    })


# --------------------------------------------------------------------- #
# 5. WEATHER PATTERN MINING (Association Rules)
# --------------------------------------------------------------------- #
def patterns_view(request):
    results = data_loader.get_results()
    rules = results.get("association_rules", [])
    return render(request, "patterns.html", {"rules": rules, "active_page": "patterns"})


# --------------------------------------------------------------------- #
# 6. CLUSTERING
# --------------------------------------------------------------------- #
def clustering_view(request):
    results = data_loader.get_results()
    clustering_data = results.get("clustering", {})

    cluster_rows = []
    names = clustering_data.get("cluster_names", {})
    sizes = clustering_data.get("cluster_sizes", {})
    stats = clustering_data.get("cluster_stats", {})
    for cid, name in names.items():
        size = sizes.get(str(cid), sizes.get(cid, sizes.get(int(cid) if str(cid).isdigit() else cid, "-")))
        s = stats.get(str(cid), stats.get(cid, {}))
        cluster_rows.append({
            "id": cid, "name": name, "size": size,
            "temperature": s.get("temperature", "-"),
            "humidity": s.get("humidity", "-"),
            "pressure": s.get("pressure", "-"),
            "wind_speed": s.get("wind_speed", "-"),
            "cloud_cover": s.get("cloud_cover", "-"),
            "rainfall": s.get("rainfall", "-"),
        })

    return render(request, "clustering.html", {
        "clustering": clustering_data,
        "cluster_rows": cluster_rows,
        "clustering_json": json.dumps(clustering_data),
        "active_page": "clustering",
    })


# --------------------------------------------------------------------- #
# 7. ANOMALIES
# --------------------------------------------------------------------- #
def anomalies_view(request):
    results = data_loader.get_results()
    anomaly_data = results.get("anomaly_detection", {})
    severity_filter = request.GET.get("severity")

    rows = anomaly_data.get("sample_anomalies", [])
    if severity_filter:
        rows = [r for r in rows if r.get("severity") == severity_filter]

    return render(request, "anomalies.html", {
        "summary": anomaly_data.get("summary", {}),
        "rows": rows,
        "severity_filter": severity_filter,
        "active_page": "anomalies",
    })


# --------------------------------------------------------------------- #
# 8. MODEL COMPARISON
# --------------------------------------------------------------------- #
def model_comparison_view(request):
    results = data_loader.get_results()
    classification = results.get("classification", {})
    regression = results.get("regression", {})
    rainfall = results.get("rainfall_prediction", {})
    showdown_meta = results.get("algorithm_showdown", {})

    chart_data = {
        "classification": {
            "labels": list(classification.get("results", {}).keys()),
            "values": [m["f1_score"] for m in classification.get("results", {}).values()],
            "best": classification.get("best_model_name"),
        },
        "regression": {
            "labels": list(regression.get("results", {}).keys()),
            "values": [m["rmse"] for m in regression.get("results", {}).values()],
            "best": regression.get("best_model_name"),
        },
        "rainfall": {
            "labels": list(rainfall.get("results", {}).keys()),
            "values": [m["f1_score"] for m in rainfall.get("results", {}).values()],
            "best": rainfall.get("best_model_name"),
        },
    }

    clf_res = classification.get("results", {})
    chart_data["classification_metrics"] = {
        "labels": list(clf_res.keys()),
        "accuracy": [m["accuracy"] for m in clf_res.values()],
        "precision": [m["precision"] for m in clf_res.values()],
        "recall": [m["recall"] for m in clf_res.values()],
        "f1": [m["f1_score"] for m in clf_res.values()],
        "time": [m["training_time_sec"] for m in clf_res.values()],
        "best": classification.get("best_model_name"),
    }

    reg_res = regression.get("results", {})
    chart_data["regression_metrics"] = {
        "labels": list(reg_res.keys()),
        "mae": [m["mae"] for m in reg_res.values()],
        "mse": [m["mse"] for m in reg_res.values()],
        "rmse": [m["rmse"] for m in reg_res.values()],
        "r2": [m["r2"] for m in reg_res.values()],
        "time": [m["training_time_sec"] for m in reg_res.values()],
        "best": regression.get("best_model_name"),
    }
    rain_res = rainfall.get("results", {})
    chart_data["rainfall_metrics"] = {
        "labels": list(rain_res.keys()),
        "accuracy": [m["accuracy"] for m in rain_res.values()],
        "precision": [m["precision"] for m in rain_res.values()],
        "recall": [m["recall"] for m in rain_res.values()],
        "f1": [m["f1_score"] for m in rain_res.values()],
        "time": [m["training_time_sec"] for m in rain_res.values()],
        "best": rainfall.get("best_model_name"),
    }

    return render(request, "model_comparison.html", {
        "classification": classification,
        "regression": regression,
        "rainfall": rainfall,
        "showdown_meta": showdown_meta,
        "chart_data_json": json.dumps(chart_data),
        "active_page": "model_comparison",
    })


# --------------------------------------------------------------------- #
# 9. EXPLAINABLE AI
# --------------------------------------------------------------------- #
def explainable_ai_view(request):
    results = data_loader.get_results()
    explain = results.get("explainability", {})
    df = data_loader.get_dataframe()

    rain_bundle = data_loader.get_model("rainfall_model")
    sample_explanations = []
    if rain_bundle:
        sample_rows = df.dropna(subset=rain_bundle["features"]).sample(3, random_state=7)
        for _, r in sample_rows.iterrows():
            X = rain_bundle["scaler"].transform(pd.DataFrame([r[rain_bundle["features"]]]))
            expl, _ = explainability.explain_single_prediction(rain_bundle, X, r.to_dict())
            proba = float(rain_bundle["model"].predict_proba(X)[0][1])
            sample_explanations.append({
                "city": r["city"],
                "date": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
                "rain_probability": round(proba * 100, 1),
                "explanations": expl,
            })

    top_features = list(explain.get("global_feature_importance", {}).items())[:10]

    return render(request, "explainable_ai.html", {
        "used_shap": explain.get("used_shap", False),
        "top_features": top_features,
        "top_features_json": json.dumps({
            "labels": [f[0] for f in top_features], "values": [f[1] for f in top_features]
        }),
        "sample_explanations": sample_explanations,
        "active_page": "explainable_ai",
    })


# --------------------------------------------------------------------- #
# 10. CITY COMPARISON
# --------------------------------------------------------------------- #
def city_comparison_view(request):
    df = data_loader.get_dataframe()
    all_cities = data_loader.get_cities()
    selected = request.GET.getlist("cities") or all_cities[:3]

    comparison = analytics.city_comparison_data(df, selected)

    trend_data = {}
    for city in selected:
        t = analytics.trend_series(df[df["city"] == city], "temperature")
        trend_data[city] = t

    return render(request, "city_comparison.html", {
        "all_cities": all_cities,
        "selected": selected,
        "comparison": comparison,
        "comparison_json": json.dumps(comparison),
        "trend_json": json.dumps(trend_data),
        "active_page": "city_comparison",
    })


# --------------------------------------------------------------------- #
# 11. UPLOAD DATASET
# --------------------------------------------------------------------- #
def upload_view(request):
    result = None
    preview = None
    error = None

    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_df = pd.read_csv(request.FILES["csv_file"])
                missing_cols = ml_preprocessing.validate_columns(uploaded_df)
                if missing_cols:
                    error = f"Missing required columns: {', '.join(missing_cols)}"
                else:
                    preview = uploaded_df.head(10).to_html(
                        classes="table table-sm table-striped", index=False, border=0
                    )
                    cleaned, stats = ml_preprocessing.run_preprocessing(uploaded_df, verbose=False)
                    result = {
                        "stats": stats,
                        "columns": uploaded_df.columns.tolist(),
                        "n_rows": len(uploaded_df),
                    }
            except Exception as e:
                error = f"Could not process this file: {e}"
        else:
            error = "; ".join([f"{k}: {', '.join(v)}" for k, v in form.errors.items()])
    else:
        form = DatasetUploadForm()

    return render(request, "upload.html", {
        "form": form, "result": result, "preview": preview, "error": error, "active_page": "upload",
    })


# --------------------------------------------------------------------- #
# 12. REPORTS
# --------------------------------------------------------------------- #
def reports_view(request):
    results = data_loader.get_results()
    return render(request, "reports.html", {"results_available": bool(results), "active_page": "reports"})


def download_report_view(request):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    results = data_loader.get_results()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20)
    h2 = styles["Heading2"]
    normal = styles["BodyText"]

    story = []
    story.append(Paragraph("WeatherSense AI", title_style))
    story.append(Paragraph("Intelligent Weather Data Mining, Analysis &amp; Prediction System", normal))
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("1. Dataset Summary", h2))
    prep = results.get("preprocessing", {})
    data = [
        ["Metric", "Value"],
        ["Total records (raw)", prep.get("records_before", "-")],
        ["Duplicates removed", prep.get("duplicates_removed", "-")],
        ["Invalid values corrected", prep.get("invalid_values_corrected", "-")],
        ["Outliers capped", prep.get("outliers_capped", "-")],
        ["Missing values found", prep.get("missing_values_total", "-")],
        ["Records after cleaning", prep.get("records_after", "-")],
    ]
    t = Table(data, colWidths=[8 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("2. Correlation Highlights", h2))
    for c in results.get("eda", {}).get("strongest_correlations", [])[:6]:
        story.append(Paragraph(f"- {c['var1']} vs {c['var2']}: r = {c['correlation']}", normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("3. Clustering Results", h2))
    cl = results.get("clustering", {})
    story.append(Paragraph(f"Optimal clusters (K): {cl.get('best_k', '-')}", normal))
    for cid, name in cl.get("cluster_names", {}).items():
        size = cl.get("cluster_sizes", {}).get(str(cid), cl.get("cluster_sizes", {}).get(cid, "-"))
        story.append(Paragraph(f"- Cluster {cid} -- {name} ({size} records)", normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("4. Anomaly Detection", h2))
    an = results.get("anomaly_detection", {}).get("summary", {})
    story.append(Paragraph(
        f"Anomalies detected: {an.get('total_anomalies', '-')} "
        f"({an.get('anomaly_rate_pct', '-')}% of records checked)", normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(PageBreak())
    story.append(Paragraph("5. Model Comparison", h2))
    story.append(Paragraph("Weather Condition Classification", styles["Heading3"]))
    clf = results.get("classification", {}).get("results", {})
    clf_rows = [["Model", "Accuracy", "Precision", "Recall", "F1"]]
    for name, m in clf.items():
        clf_rows.append([name, m["accuracy"], m["precision"], m["recall"], m["f1_score"]])
    t2 = Table(clf_rows, colWidths=[5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Next-Day Max Temperature Regression", styles["Heading3"]))
    reg = results.get("regression", {}).get("results", {})
    reg_rows = [["Model", "MAE", "RMSE", "R2"]]
    for name, m in reg.items():
        reg_rows.append([name, m["mae"], m["rmse"], m["r2"]])
    t3 = Table(reg_rows, colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("6. Conclusions", h2))
    story.append(Paragraph(
        "The dataset shows strong, physically consistent relationships between humidity, "
        "cloud cover and rainfall, and between pressure and temperature. Tree-based models "
        "(Random Forest / XGBoost) generally performed best for classification and rainfall "
        "prediction tasks, while a simple linear model was sufficient for next-day maximum "
        "temperature given strong autocorrelation. All risk indicators in this system are "
        "model-based outputs for an academic data-mining project and are not official weather warnings.",
        normal
    ))

    doc.build(story)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename="WeatherSense_AI_Report.pdf")


# --------------------------------------------------------------------- #
# 13. ABOUT
# --------------------------------------------------------------------- #
def about_view(request):
    return render(request, "about.html", {"active_page": "about"})
