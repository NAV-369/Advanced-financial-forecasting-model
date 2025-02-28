import yfinance as yf
import pandas as pd
from typing import List, Dict
from datetime import datetime

def load_stock_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """
    Load stock data for given tickers using yfinance.
    
    Args:
        tickers (List[str]): List of stock tickers
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with stock data
    """
    data = {}
    for ticker in tickers:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=30).std() * (252 ** 0.5)
        
        data[ticker] = df
        
    return data

def calculate_portfolio_metrics(data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate portfolio metrics including returns, volatility, and risk-adjusted returns.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        weights (Dict[str, float]): Dictionary of portfolio weights
        
    Returns:
        pd.DataFrame: Portfolio metrics
    """
    # Create returns DataFrame
    returns_df = pd.DataFrame()
    for ticker in data:
        returns_df[ticker] = data[ticker]['Daily_Return']
    
    # Calculate portfolio returns
    portfolio_returns = returns_df.dot(pd.Series(weights))
    
    # Calculate metrics
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * (252 ** 0.5)
    sharpe_ratio = annual_return / annual_vol
    
    # Calculate maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    metrics = pd.DataFrame({
        'Annual_Return': [annual_return],
        'Annual_Volatility': [annual_vol],
        'Sharpe_Ratio': [sharpe_ratio],
        'Max_Drawdown': [max_drawdown]
    })
    
    return metrics 