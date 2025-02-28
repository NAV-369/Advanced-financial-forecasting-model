import pandas as pd
import os
import numpy as np
from typing import Dict, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

def load_stock_data(data_dir: str = 'notebooks/data') -> Dict[str, pd.DataFrame]:
    """
    Load stock data from CSV files.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each stock
    """
    try:
        # Load each stock data file
        data = {}
        for ticker in ['TSLA', 'BND', 'SPY']:
            file_path = os.path.join(data_dir, f'{ticker}_data.csv')
            df = pd.read_csv(file_path)
            
            # Convert index to datetime
            if 'Date' in df.columns:
                df.set_index(pd.to_datetime(df['Date']), inplace=True)
            elif 'Unnamed: 0' in df.columns:
                df.set_index(pd.to_datetime(df['Unnamed: 0']), inplace=True)
                df.index.name = 'Date'
                df.drop('Unnamed: 0', axis=1, inplace=True)
            
            # Convert 'Close' to numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            data[ticker] = df
            
        return data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def calculate_returns(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate daily returns for each stock.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with daily returns added
    """
    try:
        for ticker, df in data.items():
            df['Daily Return'] = df['Close'].pct_change()
        return data
    
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        raise

def get_return_statistics(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Calculate summary statistics for daily returns.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        
    Returns:
        Dict[str, pd.Series]: Dictionary of summary statistics for each stock
    """
    stats = {}
    for ticker, df in data.items():
        stats[ticker] = df['Daily Return'].describe()
    return stats

def download_stock_data(start_date: str = '2015-01-01', 
                       end_date: str = '2025-01-31') -> Dict[str, pd.DataFrame]:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        start_date (str): Start date for data download
        end_date (str): End date for data download
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each stock
    """
    tickers = ['TSLA', 'BND', 'SPY']
    data = {}
    
    for ticker in tickers:
        print(f"Downloading {ticker} data...")
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    
    return data

def clean_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and preprocess the data.
    
    Args:
        data (Dict[str, pd.DataFrame]): Raw data dictionary
        
    Returns:
        Dict[str, pd.DataFrame]: Cleaned data dictionary
    """
    cleaned_data = {}
    
    for ticker, df in data.items():
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        # Forward fill price data
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Interpolate volume data
        df['Volume'] = df['Volume'].interpolate(method='linear')
        
        # Calculate additional features
        df['Daily Return'] = df['Close'].pct_change()
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Daily Return'].rolling(window=30).std() * np.sqrt(252)
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        cleaned_data[ticker] = df
    
    return cleaned_data

def detect_outliers(data: Dict[str, pd.DataFrame], 
                   column: str = 'Daily Return', 
                   threshold: float = 3.0) -> Dict[str, pd.DataFrame]:
    """
    Detect outliers using z-score method.
    
    Args:
        data (Dict[str, pd.DataFrame]): Data dictionary
        column (str): Column to check for outliers
        threshold (float): Z-score threshold
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with outlier information
    """
    outliers = {}
    
    for ticker, df in data.items():
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers[ticker] = df[z_scores > threshold].copy()
        outliers[ticker]['Z_Score'] = z_scores[z_scores > threshold]
    
    return outliers

def perform_seasonal_decomposition(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Perform seasonal decomposition of time series data.
    
    Args:
        data (Dict[str, pd.DataFrame]): Data dictionary
        
    Returns:
        Dict[str, Dict]: Dictionary containing decomposition results
    """
    decompositions = {}
    
    for ticker, df in data.items():
        try:
            # Perform decomposition on log prices to ensure additivity
            log_prices = np.log(df['Close'])
            decomposition = seasonal_decompose(log_prices, period=252)  # 252 trading days
            
            decompositions[ticker] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            print(f"Could not decompose {ticker} data: {str(e)}")
            continue
    
    return decompositions

def calculate_risk_metrics(data: Dict[str, pd.DataFrame], 
                         risk_free_rate: float = 0.02) -> Dict[str, Dict[str, float]]:
    """
    Calculate various risk metrics for each stock.
    
    Args:
        data (Dict[str, pd.DataFrame]): Data dictionary
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing risk metrics
    """
    risk_metrics = {}
    
    for ticker, df in data.items():
        returns = df['Daily Return'].dropna()
        
        # Calculate metrics
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        risk_metrics[ticker] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'VaR 95%': var_95,
            'VaR 99%': var_99,
            'Max Drawdown': max_drawdown,
            'Positive Days %': (returns > 0).mean() * 100,
            'Negative Days %': (returns < 0).mean() * 100
        }
    
    return risk_metrics

def get_data_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Get comprehensive summary statistics for each stock.
    
    Args:
        data (Dict[str, pd.DataFrame]): Data dictionary
        
    Returns:
        Dict[str, Dict]: Dictionary containing summary statistics
    """
    summary = {}
    
    for ticker, df in data.items():
        # Basic statistics
        basic_stats = df['Close'].describe()
        
        # Additional statistics
        returns_stats = df['Daily Return'].describe()
        
        # Missing values information
        missing_values = df.isnull().sum()
        
        # Data types
        dtypes = df.dtypes
        
        summary[ticker] = {
            'basic_stats': basic_stats,
            'returns_stats': returns_stats,
            'missing_values': missing_values,
            'dtypes': dtypes,
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'trading_days': len(df)
            }
        }
    
    return summary 