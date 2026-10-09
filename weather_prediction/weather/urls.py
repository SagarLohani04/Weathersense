from django.urls import path
from . import views

app_name = "weather"

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("prediction/", views.prediction_view, name="prediction"),
    path("algorithm-showdown/", views.algorithm_showdown_view, name="algorithm_showdown"),
    path("forecast/", views.forecast_view, name="forecast"),
    path("analysis/", views.analysis_view, name="analysis"),
    path("patterns/", views.patterns_view, name="patterns"),
    path("clustering/", views.clustering_view, name="clustering"),
    path("anomalies/", views.anomalies_view, name="anomalies"),
    path("model-comparison/", views.model_comparison_view, name="model_comparison"),
    path("explainable-ai/", views.explainable_ai_view, name="explainable_ai"),
    path("city-comparison/", views.city_comparison_view, name="city_comparison"),
    path("upload/", views.upload_view, name="upload"),
    path("reports/", views.reports_view, name="reports"),
    path("reports/download/", views.download_report_view, name="download_report"),
    path("about/", views.about_view, name="about"),
]
