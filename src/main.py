import os
import pandas as pd
import numpy as np
from data_processing.data_loader import (
    download_stock_data, clean_data, detect_outliers,
    perform_seasonal_decomposition, calculate_risk_metrics,
    get_data_summary
)
from visualization.plotting import (
    plot_stock_prices, plot_daily_returns, plot_correlation_matrix,
    plot_rolling_metrics, plot_risk_return_scatter
)
from datetime import datetime, timedelta
from src.data.data_loader import load_stock_data
from src.analysis.portfolio_analysis import (
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    calculate_correlation_matrix
)
from src.analysis.forecasting import forecast_portfolio

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    directories = ['output', 'output/plots', 'output/data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """
    Main function to run the financial analysis pipeline.
    """
    print("Starting financial analysis pipeline...")
    
    # Create output directories
    create_output_directories()
    
    # Define portfolio tickers and weights
    portfolio = {
        'TSLA': 0.4,
        'BND': 0.3,
        'SPY': 0.3
    }
    
    # Load data
    print("Loading stock data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    stock_data = load_stock_data(list(portfolio.keys()), start_date, end_date)
    
    # Calculate portfolio metrics
    print("\nCalculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(stock_data, portfolio)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv('output/data/portfolio_metrics.csv', index=False)
    print("\nPortfolio Metrics:")
    print(metrics_df)
    
    # Calculate and plot correlation matrix
    print("\nGenerating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(stock_data)
    plot_correlation_matrix(corr_matrix, 'output/plots/correlation_matrix.png')
    
    # Calculate and plot rolling metrics
    print("\nCalculating rolling metrics...")
    rolling_metrics = calculate_rolling_metrics(stock_data, portfolio)
    plot_rolling_metrics(rolling_metrics, 'output/plots/rolling_metrics.png')
    
    # Plot stock prices
    print("\nPlotting stock prices...")
    plot_stock_prices(stock_data, 'output/plots/stock_prices.png')
    
    # Generate and plot forecasts
    print("\nGenerating forecasts...")
    forecasts = forecast_portfolio(stock_data, forecast_days=30)
    
    # Save forecasts
    for ticker, models in forecasts.items():
        for model_name, predictions in models.items():
            predictions.to_csv(f'output/data/{ticker}_{model_name}_forecast.csv')
        
        # Plot forecasts
        plot_forecasts(
            stock_data[ticker]['Close'],
            models['lstm'],
            models['prophet'],
            f'output/plots/{ticker}_forecasts.png',
            ticker
        )
    
    print("\nAnalysis complete! Check the 'output' directory for results.")

if __name__ == "__main__":
    main() 