# Sales Forecast Dashboard

An interactive Streamlit dashboard for retail sales forecasting using XGBoost with lag features and rolling statistics, built on the Walmart 45-store dataset. Includes 7-day forward forecasting and live metrics display.

---

## Overview

Retail demand forecasting requires capturing temporal patterns — weekly seasonality, recent trends, and holiday effects — in a form a gradient-boosted model can use. This project engineers time-series features from raw weekly sales data, trains an XGBoost regressor, evaluates it on a held-out test set, and wraps everything in a Streamlit dashboard with live forecast generation.

---

## Features

- **Feature engineering** — lag-1 sales, 7-day rolling mean, holiday flag derived from day-of-week
- **XGBoost regression** — trained on an 80/20 chronological split (no shuffle, preserving time order)
- **Evaluation metrics** — MAE and RMSE displayed in the dashboard
- **Interactive chart** — actual vs. predicted sales plotted over time
- **7-day forward forecast** — iterative prediction loop using the model's own outputs as future lag features

---

## Model & Features

| Feature | Description |
|---|---|
| `lag_1` | Previous week's sales |
| `rolling_mean_7` | 7-period rolling mean of sales |
| `Holiday_Flag` | 1 if date falls on weekend (Sat/Sun), else 0 |

**Model:** `XGBRegressor` (default hyperparameters)  
**Split:** Chronological 80/20 — training on earlier dates, testing on most recent

**Metrics reported:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## Dashboard

The Streamlit app (`st.line_chart`) displays:
1. Model performance metrics (MAE, RMSE)
2. Actual vs. predicted sales chart on the test period
3. A 7-day rolling forecast table generated autoregressively

---

## Tech Stack

- **Modeling:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Visualization:** Built-in Streamlit charts

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit
```

### 2. Dataset
Uses the [Walmart Store Sales Forecasting dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) (45 stores). Download from Kaggle and update the path:
```python
data = pd.read_csv('walmart-sales-dataset-of-45stores.csv', parse_dates=['Date'])
```

### 3. Run the dashboard
```bash
streamlit run AadityaMKulkarni_sales_dashboard.py
```

The app opens at `http://localhost:8501`.

---

## Key Design Decisions

**Why chronological split?** Random splitting of time series leaks future data into training. An 80/20 split on sorted dates ensures the model is evaluated on genuinely unseen future periods.

**Why lag + rolling features?** XGBoost has no built-in notion of time. Lag-1 encodes last week's value directly; the 7-day rolling mean smooths short-term noise to capture trend. Together they give the model enough temporal signal to outperform a naive baseline.

**Autoregressive 7-day forecast:** Each forecasted value is fed back as `lag_1` for the next step. This is a simple but realistic forecasting loop — the same approach used in production time-series pipelines where future actuals aren't available.

---

## Author
Aaditya Kulkarni — [GitHub](https://github.com/AadityaK16) · [LinkedIn](https://www.linkedin.com/in/aaditya-kulkarni-06932b32a/)
