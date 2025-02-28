import pandas as pd
import os
from typing import Dict, Tuple

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