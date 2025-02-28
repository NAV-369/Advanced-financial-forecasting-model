from data_processing.data_loader import load_stock_data, calculate_returns
from visualization.plotting import (
    plot_daily_returns,
    plot_price_trends,
    plot_return_distribution,
    plot_risk_return_analysis
)
from analysis.portfolio_analysis import (
    calculate_portfolio_metrics,
    calculate_correlation_matrix,
    calculate_rolling_metrics,
    calculate_portfolio_betas
)
import os
import pandas as pd

def main():
    # Create output directories
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/analysis', exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    data = load_stock_data()
    data = calculate_returns(data)
    
    # Generate and save plots
    print("\nGenerating visualizations...")
    
    # Basic plots
    plot_daily_returns(data, save_path='output/plots/daily_returns.png')
    plot_price_trends(data, save_path='output/plots/price_trends.png')
    plot_return_distribution(data, save_path='output/plots/return_distribution.png')
    plot_risk_return_analysis(data, save_path='output/plots/risk_return.png')
    
    # Calculate and display metrics
    print("\nCalculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(data, risk_free_rate=0.02)
    
    # Create a DataFrame for better display and save to CSV
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    print("\nPortfolio Metrics:")
    print(metrics_df)
    metrics_df.to_csv('output/analysis/portfolio_metrics.csv')
    
    # Calculate and display correlation matrices
    print("\nCalculating correlation matrices...")
    # Pearson correlation
    pearson_corr = calculate_correlation_matrix(data, method='pearson')
    print("\nPearson Correlation Matrix:")
    print(pearson_corr)
    pearson_corr.to_csv('output/analysis/pearson_correlation.csv')
    
    # Spearman correlation
    spearman_corr = calculate_correlation_matrix(data, method='spearman')
    print("\nSpearman Correlation Matrix:")
    print(spearman_corr)
    spearman_corr.to_csv('output/analysis/spearman_correlation.csv')
    
    # Calculate rolling metrics
    print("\nCalculating rolling metrics...")
    rolling_metrics = calculate_rolling_metrics(data)
    
    # Save rolling metrics to CSV
    for ticker, metrics_df in rolling_metrics.items():
        print(f"\n{ticker} Rolling Metrics (last 5 days):")
        print(metrics_df.tail())
        metrics_df.to_csv(f'output/analysis/rolling_metrics_{ticker}.csv')
    
    # Calculate and display portfolio betas
    print("\nCalculating portfolio betas...")
    betas = calculate_portfolio_betas(data)
    betas_df = pd.DataFrame.from_dict(betas, orient='index')
    print("\nPortfolio Betas (relative to SPY):")
    print(betas_df)
    betas_df.to_csv('output/analysis/portfolio_betas.csv')
    
    print("\nAnalysis complete! Results have been saved to the 'output' directory.")
    print("\nOutput files:")
    print("  Plots:")
    print("    - output/plots/daily_returns.png")
    print("    - output/plots/price_trends.png")
    print("    - output/plots/return_distribution.png")
    print("    - output/plots/risk_return.png")
    print("  Analysis:")
    print("    - output/analysis/portfolio_metrics.csv")
    print("    - output/analysis/pearson_correlation.csv")
    print("    - output/analysis/spearman_correlation.csv")
    print("    - output/analysis/portfolio_betas.csv")
    print("    - output/analysis/rolling_metrics_*.csv")

if __name__ == "__main__":
    main() 