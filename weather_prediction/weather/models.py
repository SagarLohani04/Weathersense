"""
weather/models.py
WeatherSense AI - Database Models

Note: the bulk of historical weather data is analyzed directly from
data/weather_features.csv via pandas for speed (21,000+ rows); these
Django models store what genuinely belongs in a relational database
for this app -- user-triggered predictions, detected anomalies (a
persisted sample), and model performance snapshots -- plus a
WeatherRecord model so the admin panel and ORM layer are demonstrated
as required by the DWM database-design brief.
"""

from django.db import models


class WeatherRecord(models.Model):
    date = models.DateField(db_index=True)
    city = models.CharField(max_length=50, db_index=True)
    temperature = models.FloatField()
    humidity = models.FloatField()
    pressure = models.FloatField()
    rainfall = models.FloatField()
    wind_speed = models.FloatField()
    cloud_cover = models.FloatField()
    weather_condition = models.CharField(max_length=30, blank=True)

    class Meta:
        ordering = ['-date']
        indexes = [models.Index(fields=['city', 'date'])]

    def __str__(self):
        return f"{self.city} - {self.date}"


class Prediction(models.Model):
    PREDICTION_TYPES = [
        ('weather_condition', 'Weather Condition'),
        ('rainfall', 'Rainfall'),
        ('temperature', 'Temperature'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    city = models.CharField(max_length=50)
    prediction_type = models.CharField(max_length=30, choices=PREDICTION_TYPES)
    input_features = models.JSONField()
    predicted_value = models.CharField(max_length=100)
    model_used = models.CharField(max_length=100)
    confidence = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.prediction_type} for {self.city} @ {self.timestamp:%Y-%m-%d %H:%M}"


class Anomaly(models.Model):
    SEVERITY_CHOICES = [
        ('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High'), ('Critical', 'Critical'),
    ]

    date = models.DateField()
    city = models.CharField(max_length=50)
    anomaly_type = models.CharField(max_length=50)
    value = models.FloatField()
    expected_range = models.CharField(max_length=50, blank=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    score = models.FloatField()

    class Meta:
        ordering = ['-date']
        verbose_name_plural = 'Anomalies'

    def __str__(self):
        return f"{self.city} {self.anomaly_type} anomaly ({self.severity}) on {self.date}"


class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=100)
    task = models.CharField(max_length=50)  # classification / regression / rainfall
    metric_name = models.CharField(max_length=30)
    metric_value = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.model_name} [{self.task}] {self.metric_name}={self.metric_value}"
