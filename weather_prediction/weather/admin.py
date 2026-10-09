from django.contrib import admin
from .models import WeatherRecord, Prediction, Anomaly, ModelPerformance


@admin.register(WeatherRecord)
class WeatherRecordAdmin(admin.ModelAdmin):
    list_display = ('date', 'city', 'temperature', 'humidity', 'rainfall', 'weather_condition')
    list_filter = ('city', 'weather_condition')
    search_fields = ('city',)
    date_hierarchy = 'date'


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'city', 'prediction_type', 'predicted_value', 'model_used', 'confidence')
    list_filter = ('prediction_type', 'city', 'model_used')
    search_fields = ('city', 'model_used')


@admin.register(Anomaly)
class AnomalyAdmin(admin.ModelAdmin):
    list_display = ('date', 'city', 'anomaly_type', 'value', 'severity', 'score')
    list_filter = ('severity', 'city', 'anomaly_type')
    search_fields = ('city',)
    date_hierarchy = 'date'


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'task', 'metric_name', 'metric_value', 'timestamp')
    list_filter = ('task', 'model_name')
