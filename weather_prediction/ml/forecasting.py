"""
ml/forecasting.py
WeatherSense AI - Time-Series Forecasting (ARIMA/SARIMA)

Forecasts temperature, rainfall and humidity for a chosen city over a
7/14/30-day horizon. Uses a chronological (non-shuffled) train/test
split for backtesting, and reports real MAE/RMSE on the held-out tail
of the series before producing the future forecast.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

FORECAST_VARS = ["temperature", "rainfall", "humidity"]


def _fit_sarimax(series, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def backtest(series, horizon=14):
    """Chronological backtest: fit on all but the last `horizon` points, forecast, compare."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    fit = _fit_sarimax(train)
    pred = fit.forecast(steps=horizon)
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    return {"mae": round(float(mae), 3), "rmse": round(float(rmse), 3)}


def forecast_city(df, city, variable, horizon=14):
    city_df = df[df["city"] == city].copy()
    city_df["date"] = pd.to_datetime(city_df["date"])
    city_df = city_df.sort_values("date")
    series = city_df.set_index("date")[variable].asfreq("D")
    series = series.interpolate()

    backtest_metrics = backtest(series, horizon=min(horizon, 14))

    fit = _fit_sarimax(series)
    forecast_res = fit.get_forecast(steps=horizon)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.2)

    last_date = series.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    if variable == "rainfall":
        mean_forecast = mean_forecast.clip(lower=0)

    result = {
        "city": city,
        "variable": variable,
        "horizon": horizon,
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast": [round(float(v), 2) for v in mean_forecast.values],
        "lower_bound": [round(float(max(v, 0) if variable == "rainfall" else v), 2)
                         for v in conf_int.iloc[:, 0].values],
        "upper_bound": [round(float(v), 2) for v in conf_int.iloc[:, 1].values],
        "backtest_mae": backtest_metrics["mae"],
        "backtest_rmse": backtest_metrics["rmse"],
    }
    return result


if __name__ == "__main__":
    import json
    df = pd.read_csv("data/weather_features.csv")
    out = forecast_city(df, "Pune", "temperature", horizon=14)
    print(f"City: {out['city']}  Variable: {out['variable']}")
    print(f"Backtest MAE={out['backtest_mae']}  RMSE={out['backtest_rmse']}")
    print("Next 5 forecast values:", out["forecast"][:5])
