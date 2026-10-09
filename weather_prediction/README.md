# WeatherSense AI

**Intelligent Weather Data Mining, Analysis & Prediction System**
*A Data Mining and Machine Learning Framework for Weather Forecasting, Pattern Discovery, Anomaly Detection and Risk Analysis*

A DWM (Data Warehousing and Mining) mini-project built as a full, working AI-powered
weather intelligence platform — not a collection of disconnected scripts. Every chart,
metric and prediction in the running application is generated from a real trained model
or a real computation over the dataset; nothing is hard-coded.

---

## 1. Project Overview

WeatherSense AI ingests ~22,000 synthetic daily weather records across 10 Indian cities
(2019–2024), cleans and engineers features from them, mines patterns and clusters,
detects anomalies, trains and compares multiple classification/regression/time-series
models, explains its predictions with SHAP, and serves everything through a Django
dashboard with 13 pages.

## 2. Features

- Synthetic dataset generator preserving realistic physical relationships (temperature ↔
  humidity ↔ pressure ↔ rainfall ↔ season ↔ monsoon)
- Full preprocessing pipeline: duplicate removal, missing-value treatment, invalid-value
  correction, city/season-aware outlier capping, with real before/after statistics
- Feature engineering: lags, rolling windows, cyclical date encodings
- Exploratory Data Analysis: correlation matrix, seasonal/monthly aggregates
- Weather pattern mining via Apriori association rules (support/confidence/lift)
- K-Means clustering with elbow/silhouette-based K selection, PCA visualization, and
  clusters named from their actual statistics ("Rainy", "Dry & Hot", etc.)
- Anomaly detection combining Isolation Forest, Z-score and IQR, with severity levels
- Weather-condition classification comparing 6 algorithms (Logistic Regression, Decision
  Tree, Random Forest, KNN, SVM, XGBoost)
- Next-day maximum temperature regression comparing 5 algorithms, using a chronological
  (non-shuffled) train/test split
- Rainfall next-day prediction (Logistic Regression, Random Forest, XGBoost)
- SARIMA time-series forecasting (7/14/30-day) with chronological backtesting
- SHAP-based explainable AI with natural-language, model-derived explanations
- Rule + ML-based extreme-weather risk engine (heatwave, heavy rainfall, flood, drought,
  storm) — clearly labeled as academic model-based indicators, not official warnings
- Interactive Django dashboard (13 pages), CSV upload with validation, PDF report
  generation, Django admin, and full model persistence via Joblib

## 3. Architecture

```
weather_prediction/
├── manage.py
├── weather_prediction/        # Django settings package
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py / asgi.py
├── weather/                    # Main Django app
│   ├── models.py                (WeatherRecord, Prediction, Anomaly, ModelPerformance)
│   ├── views.py                 (13 page views)
│   ├── urls.py
│   ├── forms.py
│   ├── admin.py
│   └── services/
│       ├── data_loader.py       (cached CSV + model loading)
│       └── analytics.py         (KPIs, trends, seasonal/trend analysis)
├── ml/                          # Data-mining / ML pipeline (framework-agnostic)
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── association_rules.py
│   ├── classification.py
│   ├── regression.py
│   ├── rainfall_prediction.py
│   ├── forecasting.py
│   ├── explainability.py
│   └── risk_engine.py
├── data/
│   ├── weather_data.csv         (raw synthetic dataset)
│   ├── weather_cleaned.csv      (after preprocessing)
│   ├── weather_features.csv     (after feature engineering)
│   ├── association_rules.csv
│   ├── anomalies_detected.csv
│   └── model_results.json       (consolidated metrics used by the dashboard)
├── models/                      # Persisted trained models (Joblib)
│   ├── weather_classifier.pkl
│   ├── rainfall_model.pkl
│   ├── temperature_model.pkl
│   ├── clustering_model.pkl
│   └── anomaly_model.pkl
├── templates/                   # 13 pages + base.html
├── static/css/style.css
├── generate_dataset.py
├── train_models.py
└── requirements.txt
```

## 4. Technology Stack

- **Backend**: Python, Django
- **Data processing**: Pandas, NumPy
- **Visualization**: Chart.js (dashboard), Matplotlib/Seaborn concepts in EDA stats
- **Machine learning**: Scikit-learn, XGBoost
- **Time-series**: Statsmodels (SARIMAX)
- **Pattern mining**: Mlxtend (Apriori)
- **Explainability**: SHAP
- **PDF reports**: ReportLab
- **Database**: SQLite via Django ORM
- **Model persistence**: Joblib

No paid APIs or services are required. Runs fully locally.

## 5. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python manage.py migrate
```

The repository already includes a generated dataset and pre-trained models under
`data/` and `models/`, so you can skip straight to step 7 (Running Django). To
regenerate everything from scratch instead:

## 6. Dataset Generation & Model Training (optional — already done)

```bash
python generate_dataset.py     # creates data/weather_data.csv (~22,000 records, ~10s)
python train_models.py         # runs the full pipeline end-to-end (~90-120s)
```

`train_models.py` runs preprocessing → feature engineering → EDA → pattern mining →
clustering → anomaly detection → classification → regression → rainfall prediction →
explainability, and saves every model to `models/*.pkl` and every metric to
`data/model_results.json`. The Django app reads these files at request time; it never
retrains a model on page load.

## 7. Running Django

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/ in your browser.

To use the Django admin panel:

```bash
python manage.py createsuperuser
```

then visit http://127.0.0.1:8000/admin/

## 8. Pages

1. **Dashboard** — KPIs, trend charts, city comparison, recent anomalies, filters
2. **Weather Prediction** — form-driven prediction (condition, rain probability, next-day
   temperature, risk levels, SHAP explanation)
3. **Algorithm Showdown** — run one weather input through every trained algorithm at once
   (all 6 classifiers, all 5 regressors, all 3 rainfall models); see each one's prediction
   side by side as bar charts and ranked lists, how much they disagree (prediction "spread"
   and an agreement meter), and a plain-language explanation of which model wins and why,
   generated from its actual held-out metrics
4. **Forecast** — 7/14/30-day SARIMA forecast per city/variable with backtested accuracy
5. **Data Mining & EDA** — correlation table, seasonal breakdown, Mann-Kendall trend test
6. **Weather Patterns** — Apriori association rules (support/confidence/lift)
7. **Clustering** — K-Means cluster profiles, silhouette-by-K chart, PCA scatter
8. **Anomalies** — Isolation Forest + statistical anomaly table with severity filter
9. **Model Comparison** — full metrics table and comparison charts for every trained model,
   with the same "why this model wins" explanation shown on the Algorithm Showdown page
10. **Explainable AI** — SHAP global feature importance + sample per-record explanations
11. **City Comparison** — multi-city selection and comparison charts
12. **Upload Dataset** — CSV upload, column validation, preview, preprocessing summary
13. **Reports** — downloadable consolidated PDF report
14. **About Project** — DWM concepts demonstrated, applications, limitations, future scope

## 9. ML Models & Data Mining Techniques

| Task | Algorithms compared | Evaluation |
|---|---|---|
| Weather condition classification | Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Next-day max temperature regression | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost | MAE, MSE, RMSE, R² |
| Rainfall next-day prediction | Logistic Regression, Random Forest, XGBoost | Accuracy, Precision, Recall, F1 |
| Clustering | K-Means (K chosen via silhouette score) | Silhouette score, PCA visualization |
| Anomaly detection | Isolation Forest, Z-score, IQR | Severity levels (Low/Medium/High/Critical) |
| Pattern mining | Apriori association rules | Support, Confidence, Lift |
| Forecasting | SARIMA | Chronological backtest MAE/RMSE |

All metrics displayed in the application are generated at training time from real,
held-out evaluation — none are hard-coded.

## 10. Applications

Agriculture (crop planning), water resource management (drought/rainfall analysis),
disaster risk analysis, transportation planning, energy demand analysis, urban planning.
This is an academic project — it does not provide official weather warnings.

## 11. Limitations

- All data is synthetically generated for academic purposes; it does not represent
  observed real-world weather
- Risk indicators are model-based outputs for a data-mining course project, not official
  emergency warnings
- Trend analysis reflects patterns within the synthetic dataset only; no real-world
  climate causation is claimed

## 12. Future Scope

- Integrate a live weather API for real-time ingestion
- Add an LSTM sequence model for multi-day forecasting where GPU resources allow
- District-level granularity and satellite imagery for the risk engine

## 13. Troubleshooting

- **"No module named ml"**: run commands from the project root (where `manage.py` lives)
- **Dataset/model files missing**: run `python generate_dataset.py` then
  `python train_models.py`
- **Port already in use**: `python manage.py runserver 8080` and open that port instead
- **Charts appear blank / empty boxes where graphs should be**: Chart.js is bundled locally at
  `static/js/chart.umd.js` specifically so charts never depend on an internet connection or an
  external CDN. If charts are still blank, run `python manage.py collectstatic` (not usually
  needed in development) or hard-refresh the browser (Ctrl+Shift+R) to clear a cached, older
  version of a template. The Google Fonts stylesheet referenced in `static/css/style.css` does
  require internet access, but if it's blocked the page simply falls back to the next font in
  the stack (Segoe UI / system sans-serif) rather than showing blank space.
