# Financial Time Series Forecasting and Portfolio Optimization Challenge

# Overview
This challenge equips trainees with skills to preprocess financial data, develop time series forecasting models, analyze market trends, and optimize investment portfolios. Participants gain hands-on experience in leveraging data-driven insights to enhance portfolio performance, minimize risks, and capitalize on market opportunities using Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) data.

## Objectives
- **Preprocessing**: Clean and explore financial data for modeling.
- **Forecasting**: Build and evaluate time series models to predict TSLA prices.
- **Trend Analysis**: Forecast future market trends and assess risks/opportunities.
- **Portfolio Optimization**: Optimize a portfolio to maximize returns and minimize risks.

# Prerequisites
- **Python Environment**: Python 3.9+ with libraries:
  - `pandas`, `numpy`, `xgboost`, `matplotlib`, `scipy`, `yfinance`, `statsmodels`, `pmdarima` (optional for ARIMA/SARIMA)
- **Data**: `merged_data.csv` (e.g., `/Users/zelalemtegene/Documents/week-11/notebooks/data/merged_data.csv`) or fetched via `yfinance`.
- **Model**: Pre-trained XGBoost model (`xgboost_trained_model.json`, e.g., `/Users/zelalemtegene/Documents/week-11/`).

---

# Task 1: Preprocess and Explore the Data

## Objective
Load, clean, and analyze historical financial data for TSLA, BND, and SPY to prepare it for modeling.

## Implementation

### Data Loading
- **Source**: Use `yfinance` to fetch data or load `merged_data.csv`.
- **Assets**:
  - TSLA: High-return, high-volatility growth stock.
  - BND: Stable, low-risk bond ETF.
  - SPY: Diversified, moderate-risk index fund.

### Data Cleaning
- **Statistics**: Check `describe()` for distribution.
- **Data Types**: Ensure numeric columns and datetime index.
- **Missing Values**: Interpolate or fill (e.g., `fillna(method='ffill')`).
- **Scaling**: Normalize if needed (not applied for XGBoost here).

### Exploratory Data Analysis (EDA)
- **Visualization**: Plot closing prices (`'nan_x.3'`, `'nan_y'`, `'Unnamed: 11'`) over time.
- **Daily Returns**: Compute and plot `pct_change()` for volatility.
- **Rolling Stats**: Calculate 30-day rolling mean/std for trends.
- **Outliers**: Detect extreme returns (e.g., >3 std devs).
- **Seasonality**: Decompose TSLA prices using `statsmodels.tsa.seasonal_decompose`.

### Volatility Analysis
- **Rolling Means/Std**: Assess short-term fluctuations.
- **VaR**: Calculate historical 95% VaR for TSLA.
- **Sharpe Ratio**: Compute risk-adjusted return (assumes risk-free rate).

## Code Example
```python
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data via yfinance or CSV
data = pd.read_csv('/Users/zelalemtegene/Documents/week-11/notebooks/data/merged_data.csv', parse_dates=['Date'], index_col='Date')
# Alternative: data = yf.download(['TSLA', 'BND', 'SPY'], start='2015-01-01', end='2025-03-04')['Adj Close']

# Clean data
data = data.fillna(method='ffill')
tsla = data['nan_x.3']

# EDA
plt.plot(tsla, label='TSLA Close')
plt.title('TSLA Closing Price')
plt.show()

daily_returns = tsla.pct_change().dropna()
plt.plot(daily_returns, label='Daily Returns')
plt.title('TSLA Daily Returns')
plt.show()

rolling_mean = tsla.rolling(window=30).mean()
rolling_std = tsla.rolling(window=30).std()
plt.plot(rolling_mean, label='30-Day Mean')
plt.plot(rolling_std, label='30-Day Std')
plt.legend()
plt.show()

# Decomposition
decomp = seasonal_decompose(tsla, period=252)
decomp.plot()
plt.show()

# VaR
var_95 = daily_returns.quantile(0.05)
print(f"TSLA 95% VaR: {var_95:.4f}")