import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import statsmodels.api as sm

# Set the default style once for all plots
plt.style.use('fivethirtyeight')  # Using a built-in style that looks professional

def plot_daily_returns(data: Dict[str, pd.DataFrame], 
                      figsize: tuple = (12, 6),
                      save_path: str = None) -> None:
    """
    Plot daily returns for multiple stocks with enhanced visualization.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot daily returns
    for ticker, df in data.items():
        ax1.plot(df.index, df['Daily Return'], label=f'{ticker} Daily Return', alpha=0.7)
    
    ax1.set_title('Daily Returns Comparison', fontsize=12, pad=15)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Return')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot rolling volatility
    for ticker, df in data.items():
        rolling_vol = df['Daily Return'].rolling(window=30).std() * np.sqrt(252)
        ax2.plot(df.index, rolling_vol, label=f'{ticker} 30-Day Volatility', alpha=0.7)
    
    ax2.set_title('30-Day Rolling Volatility', fontsize=12, pad=15)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Annualized Volatility')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_price_trends(data: Dict[str, pd.DataFrame],
                     figsize: tuple = (12, 6),
                     save_path: str = None) -> None:
    """
    Plot closing price trends with additional technical indicators.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot normalized prices
    for ticker, df in data.items():
        normalized_price = df['Close'] / df['Close'].iloc[0] * 100
        ax1.plot(df.index, normalized_price, label=f'{ticker} (Normalized)', alpha=0.7)
        
        # Add 50-day moving average
        ma50 = df['Close'].rolling(window=50).mean() / df['Close'].iloc[0] * 100
        ax1.plot(df.index, ma50, '--', label=f'{ticker} 50-day MA', alpha=0.5)
    
    ax1.set_title('Normalized Price Performance (Base=100)', fontsize=12, pad=15)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot trading volume
    for ticker, df in data.items():
        ax2.bar(df.index, df['Volume'], label=f'{ticker} Volume', alpha=0.3)
    
    ax2.set_title('Trading Volume', fontsize=12, pad=15)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_return_distribution(data: Dict[str, pd.DataFrame],
                           figsize: tuple = (15, 10),
                           save_path: str = None) -> None:
    """
    Plot enhanced return distribution analysis.
    """
    # Create figure with custom background
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(2, 2)
    
    # Histogram of returns
    ax1 = fig.add_subplot(gs[0, 0])
    for ticker, df in data.items():
        df['Daily Return'].hist(bins=50, alpha=0.5, label=ticker, ax=ax1)
    ax1.set_title('Return Distribution', fontsize=12, pad=15)
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = fig.add_subplot(gs[0, 1])
    for ticker, df in data.items():
        sm.ProbPlot(df['Daily Return'].dropna()).qqplot(line='45', ax=ax2, label=ticker)
    ax2.set_title('Q-Q Plot', fontsize=12, pad=15)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    # Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    return_data = pd.DataFrame({ticker: df['Daily Return'] for ticker, df in data.items()})
    sns.boxplot(data=return_data, ax=ax3)
    ax3.set_title('Return Distribution Box Plot', fontsize=12, pad=15)
    ax3.set_ylabel('Daily Return')
    ax3.grid(True, alpha=0.3)
    
    # Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(return_data.corr(), annot=True, cmap='coolwarm', ax=ax4, 
                vmin=-1, vmax=1, center=0)
    ax4.set_title('Return Correlation Heatmap', fontsize=12, pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_risk_return_analysis(data: Dict[str, pd.DataFrame],
                            figsize: tuple = (10, 6),
                            save_path: str = None) -> None:
    """
    Plot risk-return analysis scatter plot.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    returns = []
    volatilities = []
    tickers = []
    
    for ticker, df in data.items():
        annual_return = df['Daily Return'].mean() * 252
        annual_vol = df['Daily Return'].std() * np.sqrt(252)
        returns.append(annual_return)
        volatilities.append(annual_vol)
        tickers.append(ticker)
    
    # Create scatter plot
    ax.scatter(volatilities, returns, s=100)
    
    # Add labels for each point
    for i, ticker in enumerate(tickers):
        ax.annotate(ticker, (volatilities[i], returns[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_title('Risk-Return Analysis', fontsize=12, pad=15)
    ax.set_xlabel('Annual Volatility (Risk)')
    ax.set_ylabel('Annual Return')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show() 