# Task 3: Forecast Future Market Trends

# Overview
This task involves forecasting Tesla's (TSLA) future stock prices using a pre-trained XGBoost model from Task 2. The goal is to predict prices over a 12-month horizon (252 trading days), analyze the forecast with historical data, and provide insights into trends, volatility, and market opportunities/risks. The forecast is visualized with confidence intervals, and results are interpreted for investment insights.

## Objectives
- **Use the Trained Model for Forecasting**: Generate a 12-month forecast of Tesla’s stock prices using the XGBoost model.
- **Forecast Analysis**: Visualize the forecast alongside historical data with confidence intervals.
- **Interpret the Results**:
  - **Trend Analysis**: Identify long-term trends (upward, downward, or stable) and patterns or anomalies.
  - **Volatility and Risk**: Assess uncertainty via confidence intervals and highlight potential volatility periods.
  - **Market Opportunities and Risks**: Outline opportunities (e.g., price increases) and risks (e.g., high volatility or declines).

# Prerequisites
- **Python Environment**: Python 3.9+ with libraries:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `matplotlib`
- **Data**: `merged_data.csv` (in `week-11/notebooks/data/`), with historical prices for TSLA, BND, and SPY (e.g., `'nan_x.3'` for TSLA Close).
- **Model**: Pre-trained XGBoost model (`xgboost_trained_model.json`) in `week-11/`.

# Implementation
The implementation leverages an XGBoost model trained on TSLA, BND, and SPY price features. Here’s the process:

## 1. Forecasting with the Trained Model
- **Input**: Historical data (`merged_data.csv`) and XGBoost model expecting features: `['TSLA_Open', 'TSLA_High', 'TSLA_Low', 'BND_Close', 'BND_High', 'BND_Low', 'SPY_Close', 'SPY_High', 'SPY_Low']`.
- **Mapping**: TSLA Close is `'nan_x.3'`, with other features from `'nan_x'`, `'nan_y'`, `'Unnamed: 11'`, etc.
- **Forecast Horizon**: 252 trading days starting March 4, 2025.
- **Method**: Iterative prediction adjusting TSLA Open/High/Low with predicted Close, using historical averages for BND/SPY.

## 2. Forecast Analysis
- **Visualization**: Plot of historical TSLA prices (`'nan_x.3'`), forecast, and 95% confidence intervals (`±1.96 * historical_std`).
- **Output**: `matplotlib` plot with historical data (blue), forecast (orange), and shaded intervals.

## 3. Result Interpretation
- **Trend Analysis**: Checks forecast slope (e.g., upward from $310 to $335) and anomalies (e.g., flatness from static inputs).
- **Volatility and Risk**: Compares forecast volatility (`std` of predictions) to historical and assesses interval width.
- **Opportunities and Risks**: Identifies buy/sell points (e.g., 8% gain) and risks (e.g., wide intervals late 2025).
