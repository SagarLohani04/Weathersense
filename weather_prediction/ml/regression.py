"""
ml/regression.py
WeatherSense AI - Next-Day Maximum Temperature Regression

Trains and compares Linear Regression, Decision Tree Regressor,
Random Forest Regressor, Gradient Boosting and XGBoost to predict
`max_temperature_next_day`, using a chronological (time-respecting)
train/test split to avoid leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import time

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "dew_point", "solar_radiation", "min_temperature", "max_temperature",
    "rolling_temperature_3", "rolling_temperature_7", "temperature_lag_1",
    "temperature_lag_3", "sin_day_of_year", "cos_day_of_year",
]
TARGET = "max_temperature_next_day"


def chronological_split(df, test_frac=0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_data(df):
    data = df.dropna(subset=FEATURES + [TARGET]).copy()
    data["date"] = pd.to_datetime(data["date"])
    train_df, test_df = chronological_split(data)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit ONLY on train
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train.values, y_test.values, scaler


def train_and_compare(df):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=9, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 4),
            "training_time_sec": round(train_time, 3),
        }
        trained_models[name] = model

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = trained_models[best_name]
    best_preds = best_model.predict(X_test)

    output = {
        "results": results,
        "best_model_name": best_name,
        "sample_actual_vs_predicted": {
            "actual": [round(float(v), 2) for v in y_test[:60]],
            "predicted": [round(float(v), 2) for v in best_preds[:60]],
        },
    }
    return trained_models, scaler, output


if __name__ == "__main__":
    import joblib
    df = pd.read_csv("data/weather_features.csv")
    trained_models, scaler, output = train_and_compare(df)

    print("Model comparison (next-day max temperature regression):")
    for name, m in output["results"].items():
        print(f"  {name:20s} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}  time={m['training_time_sec']}s")
    print(f"\nBest model: {output['best_model_name']}")

    best_model = trained_models[output["best_model_name"]]
    joblib.dump(
        {"model": best_model, "scaler": scaler, "features": FEATURES, "model_name": output["best_model_name"]},
        "models/temperature_model.pkl",
    )
    print("Saved models/temperature_model.pkl")
