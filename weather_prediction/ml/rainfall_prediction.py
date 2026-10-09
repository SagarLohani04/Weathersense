"""
ml/rainfall_prediction.py
WeatherSense AI - Rainfall Next-Day Prediction

Predicts whether it will rain tomorrow (`rainfall_next_day`), using
Logistic Regression, Random Forest and XGBoost with a chronological
train/test split. Outputs both the class prediction and the
probability of rain.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import time

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "dew_point", "rainfall", "rolling_rainfall_7", "rainfall_lag_1",
    "rainfall_lag_3", "rainfall_7_day_total", "month", "sin_month", "cos_month",
]
TARGET = "rainfall_next_day"


def chronological_split(df, test_frac=0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_data(df):
    data = df.dropna(subset=FEATURES + [TARGET]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data[TARGET] = data[TARGET].astype(int)
    train_df, test_df = chronological_split(data)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train.values, y_test.values, scaler


def train_and_compare(df):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08,
                                  eval_metric="logloss", random_state=42, n_jobs=-1),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        preds = model.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
            "training_time_sec": round(train_time, 3),
        }
        trained_models[name] = model

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = trained_models[best_name]
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

    output = {
        "results": results,
        "best_model_name": best_name,
        "confusion_matrix": cm.tolist(),
    }
    return trained_models, scaler, output


if __name__ == "__main__":
    import joblib
    df = pd.read_csv("data/weather_features.csv")
    trained_models, scaler, output = train_and_compare(df)

    print("Model comparison (rainfall next-day prediction):")
    for name, m in output["results"].items():
        print(f"  {name:20s} acc={m['accuracy']:.3f}  f1={m['f1_score']:.3f}  time={m['training_time_sec']}s")
    print(f"\nBest model: {output['best_model_name']}")

    best_model = trained_models[output["best_model_name"]]
    joblib.dump(
        {"model": best_model, "scaler": scaler, "features": FEATURES, "model_name": output["best_model_name"]},
        "models/rainfall_model.pkl",
    )
    print("Saved models/rainfall_model.pkl")
