# Task 4: Optimize Portfolio Based on Forecast

# Overview
This task uses the forecasted data from Task 3 to optimize a sample investment portfolio containing Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY). The objective is to adjust portfolio allocations to maximize returns while minimizing risks, leveraging predicted market trends. The process involves forecasting BND and SPY prices, combining them with TSLA’s forecast, computing key metrics (returns, volatility, Sharpe Ratio, VaR), optimizing weights, and visualizing performance.

## Objectives
- **Portfolio Setup**: Use a portfolio with TSLA (growth/high risk), BND (stability), and SPY (diversification).
- **Forecasting**: Use TSLA’s forecast from Task 3 and generate forecasts for BND and SPY.
- **Data Combination**: Create a DataFrame `df` with daily closing prices for TSLA, BND, and SPY.
- **Returns and Covariance**: Compute annual returns and a covariance matrix.
- **Portfolio Metrics**: Define weights, calculate weighted return/risk, and optimize for Sharpe Ratio.
- **Risk and Return Analysis**: Assess average returns, volatility, TSLA’s Value at Risk (VaR), and Sharpe Ratio.
- **Optimization**: Adjust allocations to maximize returns or minimize risks, favoring BND if TSLA volatility is high.
- **Visualization**: Plot cumulative returns and risk-return analysis.
- **Summary**: Report expected return, volatility, Sharpe Ratio, allocation adjustments, and include charts.

# Prerequisites
- **Python Environment**: Python 3.9+ with libraries:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `matplotlib`
  - `scipy`
- **Data**: `merged_data.csv` (e.g., `/Users/zelalemtegene/Documents/week-11/notebooks/data/merged_data.csv`) with historical prices.
- **Model**: Pre-trained XGBoost model (`xgboost_trained_model.json`) from Task 2 (e.g., `/Users/zelalemtegene/Documents/week-11/`).
- **Task 3 Output**: TSLA forecast (regenerated if not in session).

# Implementation
The implementation builds on Task 3’s TSLA forecast, generates BND/SPY forecasts, and optimizes the portfolio.

## Step 1: Portfolio Data Setup
- **TSLA Forecast**: Regenerated using the XGBoost model from Task 3.
- **BND/SPY Forecasts**: Simple exponential growth based on historical mean daily returns.
- **DataFrame**: Combined into `df` with columns `TSLA`, `BND`, `SPY`.

## Step 2: Compute Returns and Covariance
- **Annual Returns**: Compounded average daily returns over 252 trading days.
- **Covariance Matrix**: Annualized to show asset return correlations.

## Step 3: Define Portfolio Metrics
- **Initial Weights**: Equal allocation (1/3 each).
- **Weighted Return/Volatility**: Calculated using weights and covariance.

## Step 4: Optimize Portfolio
- **Sharpe Ratio**: Maximized via `scipy.optimize.minimize` with a 2% risk-free rate.
- **Constraints**: Weights sum to 1, bounded 0–1.

## Step 5: Analyze Risk and Return
- **Average Returns**: Annualized portfolio return.
- **Volatility**: Standard deviation of portfolio returns.
- **VaR**: 95% daily Value at Risk for TSLA.
- **Sharpe Ratio**: Risk-adjusted return metric.

## Step 6: Visualization
- **Cumulative Returns**: Plot of individual assets and portfolio.
- **Risk-Return Scatter**: Compares assets and portfolio.

## Step 7: Summary
- Reports optimized weights, return, volatility, Sharpe Ratio, adjustments, and includes charts.

# Code
```python
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
data = pd.read_csv('/Users/zelalemtegene/Documents/week-11/notebooks/data/merged_data.csv', parse_dates=['Date'], index_col='Date')

# Map historical data
tsla_historical = data['nan_x.3']
bnd_historical = data['nan_y']
spy_historical = data['Unnamed: 11']

# Regenerate TSLA forecast
model = xgb.Booster()
model.load_model('/Users/zelalemtegene/Documents/week-11/xgboost_trained_model.json')
forecast_horizon = 252
future_dates = pd.date_range(start=tsla_historical.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
last_data = data.tail(1)[['nan_x', 'nan_x.1', 'nan_x.2', 'nan_y', 'nan_y.1', 'nan_y.2', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13']]
last_data.columns = ['TSLA_Open', 'TSLA_High', 'TSLA_Low', 'BND_Close', 'BND_High', 'BND_Low', 'SPY_Close', 'SPY_High', 'SPY_Low']
d_last = xgb.DMatrix(last_data)
bnd_close_avg = data['nan_y'][-252:].mean()
bnd_high_avg = data['nan_y.1'][-252:].mean()
bnd_low_avg = data['nan_y.2'][-252:].mean()
spy_close_avg = data['Unnamed: 11'][-252:].mean()
spy_high_avg = data['Unnamed: 12'][-252:].mean()
spy_low_avg = data['Unnamed: 13'][-252:].mean()
forecast = []
for i in range(forecast_horizon):
    pred = model.predict(d_last)[0]
    forecast.append(pred)
    last_data['TSLA_Open'] = last_data['TSLA_Open'].values[0] * (pred / tsla_historical[-1])
    last_data['TSLA_High'] = last_data['TSLA_High'].values[0] * (pred / tsla_historical[-1])
    last_data['TSLA_Low'] = last_data['TSLA_Low'].values[0] * (pred / tsla_historical[-1])
    last_data['BND_Close'] = bnd_close_avg
    last_data['BND_High'] = bnd_high_avg
    last_data['BND_Low'] = bnd_low_avg
    last_data['SPY_Close'] = spy_close_avg
    last_data['SPY_High'] = spy_high_avg
    last_data['SPY_Low'] = spy_low_avg
    d_last = xgb.DMatrix(last_data)
tsla_forecast = pd.Series(forecast, index=future_dates)

# BND and SPY forecasts
bnd_mean_daily_return = bnd_historical.pct_change().mean()
spy_mean_daily_return = spy_historical.pct_change().mean()
bnd_forecast = bnd_historical[-1] * (1 + bnd_mean_daily_return) ** np.arange(1, forecast_horizon + 1)
spy_forecast = spy_historical[-1] * (1 + spy_mean_daily_return) ** np.arange(1, forecast_horizon + 1)

# Combine into DataFrame
df = pd.DataFrame({'TSLA': tsla_forecast, 'BND': bnd_forecast, 'SPY': spy_forecast}, index=future_dates)

# Returns and covariance
daily_returns = df.pct_change().dropna()
annual_returns = (1 + daily_returns.mean()) ** 252 - 1
annual_cov_matrix = daily_returns.cov() * 252

# Initial portfolio
weights = np.array([1/3, 1/3, 1/3])
portfolio_return = np.dot(weights, annual_returns)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

# Optimize Sharpe Ratio
rf = 0.02
def neg_sharpe_ratio(weights):
    p_return = np.dot(weights, annual_returns)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))
    return -(p_return - rf) / p_vol
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1)] * 3
result = minimize(neg_sharpe_ratio, weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x
opt_return = np.dot(optimal_weights, annual_returns)
opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(annual_cov_matrix, optimal_weights)))
opt_sharpe = (opt_return - rf) / opt_vol

# Portfolio analysis
portfolio_returns = daily_returns.dot(optimal_weights)
avg_portfolio_return = (1 + portfolio_returns.mean()) ** 252 - 1
portfolio_std = portfolio_returns.std() * np.sqrt(252)
var_95 = np.percentile(daily_returns['TSLA'], 5) * df['TSLA'].iloc[-1]

# Visualization
cumulative_returns = (1 + daily_returns).cumprod()
portfolio_cumulative = (1 + portfolio_returns).cumprod()
plt.figure(figsize=(12, 6))
for col in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
plt.plot(cumulative_returns.index, portfolio_cumulative, label='Portfolio', linewidth=2, color='black')
plt.title('Cumulative Returns Forecast')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(portfolio_std, avg_portfolio_return, c='black', marker='*', label='Portfolio')
plt.scatter(annual_returns['TSLA'], annual_returns['TSLA'], c='blue', label='TSLA')
plt.scatter(annual_returns['BND'], annual_returns['BND'], c='green', label='BND')
plt.scatter(annual_returns['SPY'], annual_returns['SPY'], c='red', label='SPY')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Risk-Return Analysis')
plt.legend()
plt.show()

# Summary
print(f"Optimal Weights: TSLA {optimal_weights[0]:.2%}, BND {optimal_weights[1]:.2%}, SPY {optimal_weights[2]:.2%}")
print(f"Expected Return: {opt_return:.4f}")
print(f"Volatility: {opt_vol:.4f}")
print(f"Sharpe Ratio: {opt_sharpe:.4f}")
print(f"TSLA 95% VaR (Daily): ${-var_95:.2f}")