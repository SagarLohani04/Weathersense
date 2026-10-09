"""
weather/forms.py
WeatherSense AI - Forms (prediction input, CSV upload)
"""

from django import forms


class WeatherPredictionForm(forms.Form):
    city = forms.ChoiceField(label="City")
    date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    temperature = forms.FloatField(min_value=-10, max_value=55, label="Temperature (°C)")
    humidity = forms.FloatField(min_value=0, max_value=100, label="Humidity (%)")
    pressure = forms.FloatField(min_value=930, max_value=1050, label="Pressure (hPa)")
    wind_speed = forms.FloatField(min_value=0, max_value=150, label="Wind Speed (km/h)")
    cloud_cover = forms.FloatField(min_value=0, max_value=100, label="Cloud Cover (%)")
    rainfall = forms.FloatField(min_value=0, max_value=300, label="Rainfall Today (mm)")
    dew_point = forms.FloatField(min_value=-20, max_value=40, label="Dew Point (°C)")

    def __init__(self, *args, cities=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cities:
            self.fields["city"].choices = [(c, c) for c in cities]


class DatasetUploadForm(forms.Form):
    csv_file = forms.FileField(label="Upload a weather CSV file")

    def clean_csv_file(self):
        f = self.cleaned_data["csv_file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Only .csv files are supported.")
        max_bytes = 15 * 1024 * 1024
        if f.size > max_bytes:
            raise forms.ValidationError("File is too large (max 15 MB).")
        return f


class ForecastForm(forms.Form):
    HORIZON_CHOICES = [(7, "7 Days"), (14, "14 Days"), (30, "30 Days")]
    VARIABLE_CHOICES = [
        ("temperature", "Temperature"),
        ("rainfall", "Rainfall"),
        ("humidity", "Humidity"),
    ]

    city = forms.ChoiceField(label="City")
    variable = forms.ChoiceField(choices=VARIABLE_CHOICES, label="Variable")
    horizon = forms.ChoiceField(choices=HORIZON_CHOICES, label="Forecast Horizon")

    def __init__(self, *args, cities=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cities:
            self.fields["city"].choices = [(c, c) for c in cities]
