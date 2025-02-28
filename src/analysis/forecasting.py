import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class StockForecaster:
    def __init__(self, window_size: int = 60):
        """
        Initialize the forecaster with window size for sequence prediction.
        
        Args:
            window_size (int): Number of time steps to use for prediction
        """
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.prophet_model = None
        
    def prepare_lstm_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for training
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data
        """
        self.lstm_model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
    def train_lstm(self, data: pd.Series, epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train LSTM model on the data.
        
        Args:
            data (pd.Series): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        X, y = self.prepare_lstm_data(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if self.lstm_model is None:
            self.build_lstm_model((X.shape[1], 1))
            
        history = self.lstm_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        return history.history
    
    def predict_lstm(self, data: pd.Series, forecast_steps: int = 30) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            data (pd.Series): Input data
            forecast_steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predicted values
        """
        # Prepare the last window of data
        last_window = data[-self.window_size:].values.reshape(-1, 1)
        last_window = self.scaler.transform(last_window)
        
        predictions = []
        current_window = last_window.copy()
        
        for _ in range(forecast_steps):
            # Reshape for LSTM
            current_window_reshaped = current_window.reshape((1, self.window_size, 1))
            # Get prediction
            next_pred = self.lstm_model.predict(current_window_reshaped, verbose=0)
            predictions.append(next_pred[0, 0])
            # Update window
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_pred
            
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def train_prophet(self, data: pd.Series) -> None:
        """
        Train Prophet model on the data.
        
        Args:
            data (pd.Series): Training data
        """
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        self.prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        self.prophet_model.fit(df)
        
    def predict_prophet(self, forecast_steps: int = 30) -> pd.DataFrame:
        """
        Make predictions using the trained Prophet model.
        
        Args:
            forecast_steps (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: Forecast results
        """
        future_dates = self.prophet_model.make_future_dataframe(periods=forecast_steps)
        forecast = self.prophet_model.predict(future_dates)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def forecast_portfolio(data: Dict[str, pd.DataFrame], 
                      forecast_days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate forecasts for all stocks in the portfolio using both LSTM and Prophet.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames
        forecast_days (int): Number of days to forecast
        
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Forecasts for each stock
    """
    forecasts = {}
    
    for ticker, df in data.items():
        print(f"\nGenerating forecasts for {ticker}...")
        
        # Initialize forecaster
        forecaster = StockForecaster()
        
        # Train and predict with LSTM
        print("Training LSTM model...")
        lstm_history = forecaster.train_lstm(df['Close'])
        lstm_predictions = forecaster.predict_lstm(df['Close'], forecast_days)
        
        # Train and predict with Prophet
        print("Training Prophet model...")
        forecaster.train_prophet(df['Close'])
        prophet_predictions = forecaster.predict_prophet(forecast_days)
        
        # Store results
        forecasts[ticker] = {
            'lstm': pd.DataFrame({
                'Date': pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_days
                ),
                'Prediction': lstm_predictions
            }).set_index('Date'),
            'prophet': prophet_predictions.set_index('ds').rename(
                columns={'yhat': 'Prediction', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
            )
        }
        
    return forecasts 