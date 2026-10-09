"""
ml/classification.py
WeatherSense AI - Weather Condition Classification

Trains and compares Logistic Regression, Decision Tree, Random Forest,
KNN, SVM and XGBoost to predict `weather_condition`, using a held-out
test set (no leakage) and reports Accuracy/Precision/Recall/F1 plus a
confusion matrix for the best model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from xgboost import XGBClassifier
import time

FEATURES = [
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "visibility", "dew_point", "solar_radiation", "uv_index",
    "precipitation_probability", "sin_month", "cos_month",
    "rolling_humidity_7", "rolling_temperature_7",
]
TARGET = "weather_condition"


def prepare_data(df, test_size=0.2, seed=42):
    data = df.dropna(subset=FEATURES + [TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, le


def train_and_compare(df):
    X_train, X_test, y_train, y_test, scaler, le = prepare_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        ),
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
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
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
        "class_labels": le.classes_.tolist(),
    }
    return trained_models, scaler, le, output


if __name__ == "__main__":
    import joblib
    df = pd.read_csv("data/weather_features.csv")
    trained_models, scaler, le, output = train_and_compare(df)

    print("Model comparison (weather condition classification):")
    for name, m in output["results"].items():
        print(f"  {name:22s} acc={m['accuracy']:.3f}  f1={m['f1_score']:.3f}  time={m['training_time_sec']}s")
    print(f"\nBest model: {output['best_model_name']}")

    best_model = trained_models[output["best_model_name"]]
    joblib.dump(
        {"model": best_model, "scaler": scaler, "label_encoder": le, "features": FEATURES,
         "model_name": output["best_model_name"]},
        "models/weather_classifier.pkl",
    )
    print("Saved models/weather_classifier.pkl")
