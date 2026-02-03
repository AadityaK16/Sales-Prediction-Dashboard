#Aaditya Makarand Kulkarni
# Sales Forecast Dashboard Project

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from xgboost import XGBRegressor

# Load data
data = pd.read_csv('walmart-sales-dataset-of-45stores.csv', parse_dates=['Date']) # Update with your file path
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
# Drop rows where Date could not be parsed
data = data.dropna(subset=['Date'])

# Data Preprocessing
data.fillna(0, inplace=True)  # Handle missing sales
data['Holiday_Flag'] = data['Date'].dt.dayofweek.isin([5, 6]).astype(int)  # Example holiday feature

# Feature Engineering
data['lag_1'] = data['Weekly_Sales'].shift(1)
data['rolling_mean_7'] = data['Weekly_Sales'].rolling(window=7).mean()
data.fillna(0, inplace=True)

# Modeling
features = ['lag_1', 'rolling_mean_7', 'Holiday_Flag']
X = data[features]
y = data['Weekly_Sales']

split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Deployment: Streamlit Dashboard
st.title("Sales Forecast Dashboard")
st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=data['Date'][split:]))

st.write("Forecast for next 7 days:")
future = data.tail(7).copy()
for i in range(7):
    last_row = future.iloc[-1]
    next_sales = model.predict([[last_row['lag_1'], last_row['rolling_mean_7'], last_row['Holiday_Flag']]])[0]
    next_date = last_row['Date'] + pd.Timedelta(days=1)
    new_row = {
        'Date': next_date,
        'Weekly_Sales': next_sales,
        'lag_1': next_sales,
        'rolling_mean_7': future['Weekly_Sales'].rolling(window=7).mean().iloc[-1],
        'Holiday_Flag': int(next_date.dayofweek in [5, 6])
    }
    future = pd.concat([future, pd.DataFrame([new_row])], ignore_index=True)
st.dataframe(future[['Date', 'Weekly_Sales']].tail(7))
