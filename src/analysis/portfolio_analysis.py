import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List

def calculate_portfolio_metrics(data: Dict[str, pd.DataFrame], risk_free_rate: float = 0.02) -> Dict[str, dict]:
    """
    Calculate comprehensive portfolio metrics for each stock.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        risk_free_rate (float): Annual risk-free rate (default: 2%)
        
    Returns:
        Dict[str, dict]: Dictionary containing metrics for each stock
    """
    metrics = {}
    
    for ticker, df in data.items():
        returns = df['Daily Return'].dropna()
        
        # Basic metrics
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown = calculate_max_drawdown(df['Close'])
        var_95 = calculate_var(returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(returns, confidence_level=0.95)
        
        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        jarque_bera = stats.jarque_bera(returns.dropna())
        
        metrics[ticker] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Value at Risk (95%)': var_95,
            'Conditional VaR (95%)': cvar_95,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Jarque-Bera Statistic': jarque_bera[0],
            'JB p-value': jarque_bera[1],
            'Best Day Return': returns.max(),
            'Worst Day Return': returns.min(),
            'Positive Days %': (returns > 0).mean() * 100
        }
    
    return metrics

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """
    Calculate Sortino ratio (similar to Sharpe ratio but only considers downside volatility).
    """
    annual_return = returns.mean() * 252
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    
    return (annual_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk using historical simulation method.
    """
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate the maximum drawdown from peak to trough.
    """
    rolling_max = prices.expanding().max()
    drawdowns = prices / rolling_max - 1.0
    return drawdowns.min()

def calculate_rolling_metrics(data: Dict[str, pd.DataFrame], 
                            window: int = 252,
                            risk_free_rate: float = 0.02) -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive rolling metrics for each stock.
    """
    rolling_metrics = {}
    
    for ticker, df in data.items():
        returns = df['Daily Return'].dropna()
        
        # Calculate rolling metrics
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
        rolling_var = returns.rolling(window=window).quantile(0.05)
        
        # Calculate rolling downside deviation
        downside_returns = returns.copy()
        downside_returns[returns > 0] = 0
        rolling_downside_std = downside_returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sortino = (rolling_mean - risk_free_rate) / rolling_downside_std
        
        rolling_metrics[ticker] = pd.DataFrame({
            'Rolling Annual Return': rolling_mean,
            'Rolling Volatility': rolling_std,
            'Rolling Sharpe': rolling_sharpe,
            'Rolling Sortino': rolling_sortino,
            'Rolling VaR (95%)': rolling_var,
            'Rolling Skewness': returns.rolling(window=window).skew(),
            'Rolling Kurtosis': returns.rolling(window=window).kurtosis()
        })
    
    return rolling_metrics

def calculate_correlation_matrix(data: Dict[str, pd.DataFrame], 
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix between different stocks.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        method (str): Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    returns_df = pd.DataFrame()
    for ticker, df in data.items():
        returns_df[ticker] = df['Daily Return']
    
    return returns_df.corr(method=method)

def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate beta and alpha of a stock relative to the market.
    
    Returns:
        Tuple[float, float]: (beta, alpha)
    """
    # Prepare data for regression
    X = market_returns.values.reshape(-1, 1)
    y = stock_returns.values
    
    # Add constant for alpha calculation
    X = np.concatenate([np.ones_like(X), X], axis=1)
    
    # Calculate beta and alpha using linear regression
    beta_alpha = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    return beta_alpha[1], beta_alpha[0]  # beta, alpha

def calculate_portfolio_betas(data: Dict[str, pd.DataFrame], 
                            market_ticker: str = 'SPY') -> Dict[str, Tuple[float, float]]:
    """
    Calculate beta and alpha for each stock relative to the market.
    """
    market_returns = data[market_ticker]['Daily Return']
    betas = {}
    
    for ticker, df in data.items():
        if ticker != market_ticker:
            beta, alpha = calculate_beta(df['Daily Return'], market_returns)
            betas[ticker] = {'Beta': beta, 'Alpha': alpha} 