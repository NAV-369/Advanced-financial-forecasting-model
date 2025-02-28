# Task 1: Preprocess and Explore the Data

## Overview
This notebook focuses on **loading, cleaning, and exploring financial data** to prepare it for further modeling. We extract historical financial data from YFinance for three assets:

- **TSLA (Tesla, Inc.)** – High volatility, potential high returns.
- **BND (Vanguard Total Bond Market ETF)** – Low risk, provides portfolio stability.
- **SPY (S&P 500 ETF)** – Diversified, moderate-risk market exposure.

## Data Preprocessing
### 1. Load and Inspect the Data
- Extract historical stock data using **YFinance**.
- Ensure all columns have appropriate **data types**.
- Check for **missing values** and **basic statistics** to understand data distribution.

### 2. Handle Missing Values
- Identify missing values in **Close price**.
- Apply one of the following strategies:
  - **Drop** missing values (if minimal impact).
  - **Interpolate** missing values using linear methods.
  - **Forward fill/backward fill** if suitable.

### 3. Normalize or Scale Data
- Normalize data if needed, especially for **machine learning models**.

## Exploratory Data Analysis (EDA)
### 1. Visualizing Time-Series Data
- **Plot closing prices** over time to identify **trends and patterns**.
- **Calculate & plot daily percentage change** to analyze volatility.

### 2. Analyze Volatility
- Compute **rolling means and standard deviations** to detect **short-term trends**.
- Identify periods of **high and low fluctuations**.
- Perform **outlier detection** to highlight significant anomalies.

### 3. Seasonality and Trends
- Use **time-series decomposition (statsmodels)** to split data into:
  - **Trend**: Long-term movement.
  - **Seasonality**: Recurring patterns.
  - **Residual**: Unexplained variation.

### 4. Risk Assessment
- Analyze **Value at Risk (VaR)** to estimate potential losses.
- Compute **Sharpe Ratio** to evaluate **risk-adjusted returns**.
- Document key insights:
  - Tesla’s stock price trends.
  - Fluctuations in daily returns and their impact.
  - Portfolio risk assessment for BND and SPY.

## Expected Outcomes
- A clean dataset **ready for modeling**.
- Key insights into **volatility, trends, and seasonality**.
- A strong foundation for **predictive analysis and investment strategy** development.

