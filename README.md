# Advanced Financial Forecasting Model

This project implements an advanced financial analysis and forecasting system for stock market data, focusing on TESLA (TSLA), Bond ETF (BND), and S&P 500 ETF (SPY).

## Project Structure

```
project/
├── notebooks/
│   └── data/
│       ├── TSLA_data.csv
│       ├── BND_data.csv
│       └── SPY_data.csv
├── src/
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── visualization/
│   │   ├── plotting.py
│   │   └── __init__.py
│   ├── analysis/
│   │   ├── portfolio_analysis.py
│   │   └── __init__.py
│   └── main.py
└── output/
    ├── plots/
    └── analysis/
```

## Features

- Data Processing:
  - Automated data loading from CSV files
  - Daily returns calculation
  - Return statistics computation

- Visualization:
  - Daily returns comparison
  - Price trends with moving averages
  - Return distribution analysis
  - Risk-return scatter plots

- Analysis:
  - Portfolio metrics calculation
  - Risk metrics (Sharpe ratio, Sortino ratio, VaR, CVaR)
  - Correlation analysis
  - Rolling metrics computation

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/NAV-369/Advanced-financial-forecasting-model.git
   cd Advanced-financial-forecasting-model
   ```

2. Install required packages:
   ```bash
   pip install pandas numpy scipy matplotlib seaborn statsmodels yfinance
   ```

3. Run the analysis:
   ```bash
   python src/main.py
   ```

## Output

The analysis generates various visualizations and metrics saved in the `output` directory:

- Plots:
  - Daily returns
  - Price trends
  - Return distributions
  - Risk-return analysis

- Analysis:
  - Portfolio metrics
  - Correlation matrices
  - Rolling metrics
  - Portfolio betas

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request