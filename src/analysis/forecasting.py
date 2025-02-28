import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    def __init__(self, window_size: int = 60):
        """
        Initialize the forecaster with multiple models.
        
        Args:
            window_size (int): Number of time steps for sequence prediction
        """
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.prophet_model = None
        self.sarima_model = None
        self.sarima_order = None
        self.sarima_seasonal_order = None
        
    def prepare_lstm_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model with advanced feature engineering.
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            # Include additional features: returns and moving averages
            window = scaled_data[i-self.window_size:i, 0]
            returns = np.diff(window) / window[:-1]  # Returns
            ma5 = np.mean(window[-5:])  # 5-day MA
            ma20 = np.mean(window[-20:])  # 20-day MA
            
            features = np.concatenate([
                window,
                returns,
                [ma5, ma20]
            ])
            
            X.append(features)
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build enhanced LSTM model architecture with additional layers.
        """
        self.lstm_model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='huber_loss'  # More robust to outliers
        )
        
    def train_lstm(self, data: pd.Series, epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train LSTM model with validation and early stopping.
        """
        X, y = self.prepare_lstm_data(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        if self.lstm_model is None:
            self.build_lstm_model((X.shape[1], 1))
            
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.lstm_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return history.history
    
    def find_optimal_sarima_params(self, data: pd.Series) -> Tuple[tuple, tuple]:
        """
        Find optimal SARIMA parameters using grid search.
        """
        import itertools
        
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(
                        data,
                        order=param,
                        seasonal_order=param_seasonal
                    )
                    results = model.fit(disp=0)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = param
                        best_seasonal_order = param_seasonal
                except:
                    continue
                    
        return best_order, best_seasonal_order
    
    def train_sarima(self, data: pd.Series) -> None:
        """
        Train SARIMA model with optimal parameters.
        """
        if self.sarima_order is None:
            self.sarima_order, self.sarima_seasonal_order = self.find_optimal_sarima_params(data)
            
        self.sarima_model = SARIMAX(
            data,
            order=self.sarima_order,
            seasonal_order=self.sarima_seasonal_order
        ).fit(disp=0)
    
    def predict_sarima(self, forecast_steps: int = 30) -> pd.Series:
        """
        Generate SARIMA predictions with confidence intervals.
        """
        forecast = self.sarima_model.get_forecast(steps=forecast_steps)
        
        return pd.DataFrame({
            'Prediction': forecast.predicted_mean,
            'Lower': forecast.conf_int()[:, 0],
            'Upper': forecast.conf_int()[:, 1]
        })
    
    def train_prophet(self, data: pd.Series) -> None:
        """
        Train Prophet model with additional seasonality and holidays.
        """
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        self.prophet_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10
        )
        
        # Add US holidays
        from prophet.holidays import USFederalHolidays
        holidays = USFederalHolidays().holidays()
        self.prophet_model.add_country_holidays(country_name='US')
        
        self.prophet_model.fit(df)
        
    def predict_prophet(self, forecast_steps: int = 30) -> pd.DataFrame:
        """
        Generate Prophet predictions with uncertainty intervals.
        """
        future_dates = self.prophet_model.make_future_dataframe(periods=forecast_steps)
        forecast = self.prophet_model.predict(future_dates)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def forecast_portfolio(data: Dict[str, pd.DataFrame], 
                      forecast_days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate comprehensive forecasts using multiple models.
    """
    forecasts = {}
    
    for ticker, df in data.items():
        print(f"\nGenerating forecasts for {ticker}...")
        
        # Initialize forecaster
        forecaster = TimeSeriesForecaster()
        
        # Train and predict with LSTM
        print("Training LSTM model...")
        lstm_history = forecaster.train_lstm(df['Close'])
        lstm_predictions = forecaster.predict_lstm(df['Close'], forecast_days)
        
        # Train and predict with SARIMA
        print("Training SARIMA model...")
        forecaster.train_sarima(df['Close'])
        sarima_predictions = forecaster.predict_sarima(forecast_days)
        
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
            'sarima': sarima_predictions,
            'prophet': prophet_predictions.set_index('ds').rename(
                columns={'yhat': 'Prediction', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
            )
        }
        
    return forecasts 