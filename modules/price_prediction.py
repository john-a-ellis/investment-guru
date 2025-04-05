# modules/price_prediction.py
"""
Machine learning models for price prediction in the Investment Recommendation System.
Implements time series forecasting using Prophet, ARIMA, and LSTM neural networks.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pickle
import os
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePredictionModel:
    """
    Base class for price prediction models.
    """
    def __init__(self, model_name="base", prediction_days=30):
        self.model_name = model_name
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.model_dir = "models"
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def preprocess_data(self, historical_data):
        """
        Preprocess data for training or prediction.
        
        Args:
            historical_data (DataFrame): Historical price data with 'Close' column
        
        Returns:
            DataFrame: Preprocessed data
        """
        # Ensure data is sorted by date
        data = historical_data.sort_index()
        
        # Extract close prices
        if 'Close' in data.columns:
            close_prices = data['Close'].values.reshape(-1, 1)
        else:
            raise ValueError("Data must contain 'Close' column")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        return scaled_data
    
    def train(self, historical_data):
        """
        Train the model on historical data.
        
        Args:
            historical_data (DataFrame): Historical price data
        
        Returns:
            bool: Success status
        """
        # Implementation in child classes
        raise NotImplementedError("train() method must be implemented in child classes")
    
    def predict(self, historical_data, days=None):
        """
        Make predictions for future prices.
        
        Args:
            historical_data (DataFrame): Historical price data
            days (int): Number of days to predict (defaults to self.prediction_days)
        
        Returns:
            DataFrame: Predicted prices
        """
        # Implementation in child classes
        raise NotImplementedError("predict() method must be implemented in child classes")
    
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (DataFrame): Test data with actual prices
        
        Returns:
            dict: Evaluation metrics
        """
        # Implementation in child classes
        raise NotImplementedError("evaluate() method must be implemented in child classes")
    
    def save_model(self):
        """
        Save model to disk.
        
        Returns:
            bool: Success status
        """
        if not self.is_trained:
            logger.warning(f"Model {self.model_name} not trained, cannot save")
            return False
        
        try:
            model_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model {self.model_name} saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {e}")
            return False
    
    def load_model(self):
        """
        Load model from disk with improved error handling.
        
        Returns:
            bool: Success status
        """
        try:
            model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.symbol}.pkl")
            standard_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
            
            print(f"Attempting to load model from: {os.path.abspath(model_path)}")
            
            # First try the symbol-specific path
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"Successfully loaded model from {os.path.abspath(model_path)}")
                return True
                
            # Try the standard path as fallback
            elif os.path.exists(standard_path):
                print(f"Trying fallback path: {os.path.abspath(standard_path)}")
                with open(standard_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"Successfully loaded model from {os.path.abspath(standard_path)}")
                return True
            
            # List available model files
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            print(f"Available model files: {model_files}")
            
            # Model file not found
            print(f"Model file not found for {self.symbol}")
            return False
            
        except Exception as e:
            print(f"Error loading model for {self.symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False



class ARIMAModel(PricePredictionModel):
    """
    Price prediction model using ARIMA (AutoRegressive Integrated Moving Average).
    """
    def __init__(self, prediction_days=30, order=(5,1,0)):
        super().__init__(model_name="arima", prediction_days=prediction_days)
        self.order = order
    
    def train(self, historical_data):
        """
        Train ARIMA model on historical data.
        
        Args:
            historical_data (DataFrame): Historical price data with 'Close' column
        
        Returns:
            bool: Success status
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Preprocess data
            data = historical_data.sort_index()
            
            # Extract close prices
            if 'Close' in data.columns:
                close_prices = data['Close']
            else:
                raise ValueError("Data must contain 'Close' column")
            
            # Train ARIMA model
            self.model = ARIMA(close_prices, order=self.order)
            self.model = self.model.fit()
            
            self.is_trained = True
            logger.info(f"ARIMA model trained successfully with order {self.order}")
            return True
        
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return False
    
    def predict(self, historical_data, days=None):
        """
        Make predictions using ARIMA model.
        
        Args:
            historical_data (DataFrame): Historical price data
            days (int): Number of days to predict
        
        Returns:
            DataFrame: Predicted prices
        """
        if not self.is_trained and not self.load_model():
            logger.error("ARIMA model not trained and could not be loaded")
            return pd.DataFrame()
        
        try:
            # Use provided days or default
            pred_days = days if days is not None else self.prediction_days
            
            # Get the last date in the historical data
            last_date = historical_data.index[-1]
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
            
            # Make predictions
            forecast = self.model.forecast(steps=pred_days)
            
            # Create a DataFrame with predictions
            predictions = pd.DataFrame(forecast, index=future_dates, columns=['Close'])
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions with ARIMA model: {e}")
            return pd.DataFrame()
    
    def evaluate(self, test_data):
        """
        Evaluate ARIMA model performance.
        
        Args:
            test_data (DataFrame): Test data with actual prices
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained and not self.load_model():
            logger.error("ARIMA model not trained and could not be loaded")
            return {}
        
        try:
            # Make predictions for the test period
            predictions = self.model.predict(
                start=test_data.index[0],
                end=test_data.index[-1]
            )
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(test_data['Close'], predictions)
            mse = mean_squared_error(test_data['Close'], predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((test_data['Close'] - predictions) / test_data['Close'])) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
        
        except Exception as e:
            logger.error(f"Error evaluating ARIMA model: {e}")
            return {}


class ProphetModel(PricePredictionModel):
    """
    Price prediction model using Facebook Prophet.
    """
    def __init__(self, prediction_days=30):
        super().__init__(model_name="prophet", prediction_days=prediction_days)
        self.symbol = symbol
    
    def train(self, historical_data):
        """
        Train Prophet model on historical data.
        
        Args:
            historical_data (DataFrame): Historical price data with 'Close' column
        
        Returns:
            bool: Success status
        """
        try:
            from prophet import Prophet
            
            # Preprocess data
            data = historical_data.sort_index().copy()
            
            # Extract close prices
            if 'Close' in data.columns:
                # Prophet requires 'ds' (date) and 'y' (target) columns
                prophet_data = pd.DataFrame({
                    'ds': data.index,
                    'y': data['Close']
                })
            else:
                raise ValueError("Data must contain 'Close' column")
            
            # Train Prophet model
            self.model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.model.fit(prophet_data)
            
            self.is_trained = True
            logger.info("Prophet model trained successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return False
    
    def predict(self, historical_data, days=None):
        """
        Make predictions using Prophet model.
        
        Args:
            historical_data (DataFrame): Historical price data
            days (int): Number of days to predict
        
        Returns:
            DataFrame: Predicted prices
        """
        if not self.is_trained and not self.load_model():
            logger.error("Prophet model not trained and could not be loaded")
            return pd.DataFrame()
        
        try:
            # Use provided days or default
            pred_days = days if days is not None else self.prediction_days
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=pred_days)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Extract predictions for future dates
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(pred_days)
            
            # Convert to DataFrame with date index
            result = pd.DataFrame({
                'Close': predictions['yhat'],
                'Lower': predictions['yhat_lower'],
                'Upper': predictions['yhat_upper']
            }, index=pd.DatetimeIndex(predictions['ds']))
            
            return result
        
        except Exception as e:
            logger.error(f"Error making predictions with Prophet model: {e}")
            return pd.DataFrame()
    
    def evaluate(self, test_data):
        """
        Evaluate Prophet model performance.
        
        Args:
            test_data (DataFrame): Test data with actual prices
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained and not self.load_model():
            logger.error("Prophet model not trained and could not be loaded")
            return {}
        
        try:
            # Create dataframe for the test period
            prophet_data = pd.DataFrame({
                'ds': test_data.index,
            })
            
            # Make predictions
            predictions = self.model.predict(prophet_data)
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(test_data['Close'], predictions['yhat'])
            mse = mean_squared_error(test_data['Close'], predictions['yhat'])
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((test_data['Close'] - predictions['yhat']) / test_data['Close'])) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
        
        except Exception as e:
            logger.error(f"Error evaluating Prophet model: {e}")
            return {}


class LSTMModel(PricePredictionModel):
    """
    Price prediction model using LSTM (Long Short-Term Memory) neural networks.
    """
    def __init__(self, prediction_days=30, lookback=60, units=50, epochs=50, batch_size=32):
        super().__init__(model_name="lstm", prediction_days=prediction_days)
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM training.
        
        Args:
            data (array): Scaled price data
        
        Returns:
            tuple: (X, y) sequences for training
        """
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, historical_data):
        """
        Train LSTM model on historical data.
        
        Args:
            historical_data (DataFrame): Historical price data with 'Close' column
        
        Returns:
            bool: Success status
        """
        try:
            # Import TensorFlow inside this method for better isolation and memory management
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Preprocess data
            scaled_data = self.preprocess_data(historical_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            self.model = Sequential()
            
            # First LSTM layer with return sequences
            self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)))
            self.model.add(Dropout(0.2))
            
            # Second LSTM layer
            self.model.add(LSTM(units=self.units))
            self.model.add(Dropout(0.2))
            
            # Output layer
            self.model.add(Dense(units=1))
            
            # Compile model
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            
            self.is_trained = True
            logger.info("LSTM model trained successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def predict(self, historical_data, days=None):
        """
        Make predictions using LSTM model.
        
        Args:
            historical_data (DataFrame): Historical price data
            days (int): Number of days to predict
        
        Returns:
            DataFrame: Predicted prices
        """
        if not self.is_trained and not self.load_model():
            logger.error("LSTM model not trained and could not be loaded")
            return pd.DataFrame()
        
        try:
            # Use TensorFlow inside the method
            import tensorflow as tf
            
            # Use provided days or default
            pred_days = days if days is not None else self.prediction_days
            
            # Preprocess data
            scaled_data = self.preprocess_data(historical_data)
            
            # Get the last sequence
            last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Make predictions one by one
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(pred_days):
                # Predict next value
                next_pred = self.model.predict(current_sequence)[0][0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction (roll the window)
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                           [[next_pred]], 
                                           axis=1)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Get the last date in the historical data
            last_date = historical_data.index[-1]
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
            
            # Create a DataFrame with predictions
            result = pd.DataFrame(predictions, index=future_dates, columns=['Close'])
            
            return result
        
        except Exception as e:
            logger.error(f"Error making predictions with LSTM model: {e}")
            return pd.DataFrame()
    
    def evaluate(self, test_data):
        """
        Evaluate LSTM model performance.
        
        Args:
            test_data (DataFrame): Test data with actual prices
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained and not self.load_model():
            logger.error("LSTM model not trained and could not be loaded")
            return {}
        
        try:
            # Split test data into sequences
            scaled_data = self.preprocess_data(test_data)
            
            # Create sequences
            X_test, y_test = self._create_sequences(scaled_data)
            
            # Reshape for LSTM [samples, time steps, features]
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            predictions = self.model.predict(X_test)
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(predictions)
            y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
        
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {e}")
            return {}
    
    def save_model(self):
        """
        Save LSTM model to disk.
        
        Returns:
            bool: Success status
        """
        if not self.is_trained:
            logger.warning("LSTM model not trained, cannot save")
            return False
        
        try:
            # Save model architecture and weights separately
            model_path = os.path.join(self.model_dir, f"{self.model_name}")
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info(f"LSTM model saved to {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return False
    
    def load_model(self):
        """
        Load LSTM model from disk.
        
        Returns:
            bool: Success status
        """
        try:
            import tensorflow as tf
            
            # Load model
            model_path = os.path.join(self.model_dir, f"{self.model_name}")
            if not os.path.exists(model_path):
                logger.warning(f"LSTM model file {model_path} not found")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"LSTM model loaded from {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            return False


class EnsembleModel(PricePredictionModel):
    """
    Ensemble model that combines predictions from multiple models.
    """
    def __init__(self, prediction_days=30, models=None, weights=None):
        super().__init__(model_name="ensemble", prediction_days=prediction_days)
        
        # Initialize models if provided, otherwise use default (ARIMA, Prophet, LSTM)
        if models is None:
            self.models = [
                ARIMAModel(prediction_days=prediction_days),
                ProphetModel(prediction_days=prediction_days),
                LSTMModel(prediction_days=prediction_days)
            ]
        else:
            self.models = models
        
        # Initialize weights if provided, otherwise use equal weights
        if weights is None:
            self.weights = [1/len(self.models)] * len(self.models)
        else:
            # Normalize weights
            self.weights = [w / sum(weights) for w in weights]
    
    def train(self, historical_data):
        """
        Train all models in the ensemble.
        
        Args:
            historical_data (DataFrame): Historical price data
        
        Returns:
            bool: Success status
        """
        success = True
        
        # Train each model
        for model in self.models:
            if not model.train(historical_data):
                logger.warning(f"Failed to train {model.model_name} model")
                success = False
        
        self.is_trained = success
        
        return success
    
    def predict(self, historical_data, days=None):
        """
        Make predictions using weighted combination of all models.
        
        Args:
            historical_data (DataFrame): Historical price data
            days (int): Number of days to predict
        
        Returns:
            DataFrame: Predicted prices
        """
        # Use provided days or default
        pred_days = days if days is not None else self.prediction_days
        
        # Get predictions from each model
        all_predictions = []
        for i, model in enumerate(self.models):
            if model.is_trained or model.load_model():
                pred = model.predict(historical_data, days=pred_days)
                if not pred.empty:
                    all_predictions.append(pred)
                    logger.info(f"Got predictions from {model.model_name} model")
                else:
                    logger.warning(f"No predictions from {model.model_name} model")
            else:
                logger.warning(f"{model.model_name} model not trained or could not be loaded")
        
        if not all_predictions:
            logger.error("No predictions available from any model")
            return pd.DataFrame()
        
        # Align predictions to the same dates
        common_index = all_predictions[0].index
        for pred in all_predictions[1:]:
            common_index = common_index.intersection(pred.index)
        
        # Get weighted average of predictions
        weighted_sum = pd.Series(0, index=common_index)
        adjusted_weights = self.weights[:len(all_predictions)]
        adjusted_weights = [w / sum(adjusted_weights) for w in adjusted_weights]
        
        for i, pred in enumerate(all_predictions):
            if pred.index.equals(common_index):
                weighted_sum += pred['Close'] * adjusted_weights[i]
            else:
                # Reindex prediction to common index
                reindexed_pred = pred.reindex(common_index)
                weighted_sum += reindexed_pred['Close'] * adjusted_weights[i]
        
        # Create final prediction DataFrame
        result = pd.DataFrame({'Close': weighted_sum})
        
        return result
    
    def evaluate(self, test_data):
        """
        Evaluate ensemble model performance.
        
        Args:
            test_data (DataFrame): Test data with actual prices
        
        Returns:
            dict: Evaluation metrics for each model and the ensemble
        """
        # Evaluate each model
        model_metrics = {}
        for model in self.models:
            if model.is_trained or model.load_model():
                metrics = model.evaluate(test_data)
                if metrics:
                    model_metrics[model.model_name] = metrics
        
        # Evaluate ensemble (make predictions on test data and compare)
        ensemble_pred = self.predict(test_data[:-self.prediction_days], days=self.prediction_days)
        
        if not ensemble_pred.empty:
            # Align predictions with actual test data
            common_index = ensemble_pred.index.intersection(test_data.index)
            if len(common_index) > 0:
                ensemble_pred = ensemble_pred.loc[common_index]
                actual = test_data.loc[common_index, 'Close']
                
                # Calculate metrics
                mae = mean_absolute_error(actual, ensemble_pred['Close'])
                mse = mean_squared_error(actual, ensemble_pred['Close'])
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual - ensemble_pred['Close']) / actual)) * 100
                
                model_metrics['ensemble'] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape
                }
        
        return model_metrics


def calculate_prophet_metrics(model, test_data):
    """
    Calculate evaluation metrics for a Prophet model
    
    Args:
        model: Trained Prophet model
        test_data (DataFrame): Test data with actual prices
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        
        # Create dataframe for model prediction on test data
        prophet_data = pd.DataFrame({
            'ds': test_data.index
        })
        
        # Make predictions
        predictions = model.predict(prophet_data)
        
        # Get actual values
        actual = test_data['Close'].values
        
        # Get predicted values (align indices)
        predicted = []
        for date in test_data.index:
            pred_row = predictions[predictions['ds'] == pd.Timestamp(date)]
            if not pred_row.empty:
                predicted.append(pred_row['yhat'].iloc[0])
            else:
                # If no prediction for this date, use the last known prediction
                predicted.append(predicted[-1] if predicted else actual[0])
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle division by zero by excluding zero values
        non_zero_indices = actual != 0
        if np.any(non_zero_indices):
            mape = np.mean(np.abs((actual[non_zero_indices] - np.array(predicted)[non_zero_indices]) / actual[non_zero_indices])) * 100
        else:
            mape = 0.0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    except Exception as e:
        print(f"Error calculating Prophet metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'error': str(e)
        }

# Make sure this is AFTER the function definition, not before
def train_price_prediction_models(symbol, lookback_period="1y"):
    """
    Train price prediction models for a specific symbol.
    
    Args:
        symbol (str): Stock symbol to train models for
        lookback_period (str): Historical period to use for training (e.g., "1y", "2y")
    
    Returns:
        dict: Trained models and performance metrics
    """
    try:
        # Check if Prophet is installed
        if not check_prophet_installed():
            raise ImportError("Prophet package is required for model training")
            
        # Get historical data
        from modules.fmp_api import fmp_api
        import logging
        logger = logging.getLogger(__name__)

        
        # Get historical price data using the period directly
        # The FMP API wrapper should handle period conversion
        historical_data = fmp_api.get_historical_price(symbol, period=lookback_period)
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Make sure we have the necessary columns
        required_columns = ['Close', 'Open', 'High', 'Low']
        for col in required_columns:
            if col not in historical_data.columns:
                # Try to map column names from lowercase if necessary
                lowercase_map = {'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}
                for lcol, ucol in lowercase_map.items():
                    if lcol in historical_data.columns:
                        historical_data[ucol] = historical_data[lcol]
        
        # Check if we have the required columns after mapping
        missing_cols = [col for col in required_columns if col not in historical_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Ensure we have enough data for training (at least 60 days)
        if len(historical_data) < 60:
            logger.error(f"Not enough historical data for {symbol} (got {len(historical_data)} rows, need at least 60)")
            return None
            
        # Split data into training and testing sets (80/20 split)
        split_idx = int(len(historical_data) * 0.8)
        train_data = historical_data.iloc[:split_idx]
        test_data = historical_data.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_data)} days, Testing data: {len(test_data)} days")
        
        # Make sure 'models' directory exists
        os.makedirs("models", exist_ok=True)
        
        # For simplified version, just return a basic Prophet model prediction
        # This is more reliable and doesn't require TensorFlow to be installed
        try:
            from prophet import Prophet
            
            # Prophet requires 'ds' (date) and 'y' (target) columns
            prophet_data = pd.DataFrame({
                'ds': historical_data.index,
                'y': historical_data['Close']
            })
            
            # Train Prophet model with basic settings
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_data)
            
            # Create future dataframe for 30 days
            future = model.make_future_dataframe(periods=30)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for future dates
            predictions = forecast[forecast['ds'] > historical_data.index[-1]][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Convert to DataFrame with date index
            result = pd.DataFrame({
                'Close': predictions['yhat'],
                'Lower': predictions['yhat_lower'],
                'Upper': predictions['yhat_upper']
            }, index=pd.DatetimeIndex(predictions['ds']))
            
            # Create a simplified result to return
            simplified_result = {
                'symbol': symbol,
                'model': 'prophet',
                'dates': result.index.strftime('%Y-%m-%d').tolist(),
                'values': result['Close'].tolist(),
                'confidence': {
                    'upper': result['Upper'].tolist(),
                    'lower': result['Lower'].tolist()
                }
            }
            
            # Also return a structured model result for compatibility
            model_result = calculate_prophet_metrics(model, test_data)
        
            
            # Save the model
            os.makedirs("models", exist_ok=True)
            with open(f"models/prophet_{symbol}.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Successfully trained and saved Prophet model for {symbol}")
            
            return model_result
            
        except Exception as prophet_error:
            logger.error(f"Error training Prophet model: {prophet_error}")
            
            # If Prophet fails, return a simplified non-dictionary result
            # that can be handled by the ModelIntegration class
            return {
                'symbol': symbol,
                'model': 'simple',
                'dates': historical_data.index[-30:].strftime('%Y-%m-%d').tolist(),
                'values': historical_data['Close'].iloc[-30:].tolist(),
                'confidence': {
                    'upper': None,
                    'lower': None
                }
            }
    
    except Exception as e:
        logger.error(f"Error in train_price_prediction_models for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_price_predictions(symbol, days=30):
    """
    Get price predictions for a specific symbol using the best available model.
    
    Args:
        symbol (str): Stock symbol to predict
        days (int): Number of days to predict
    
    Returns:
        dict: Predictions with dates, values, and confidence intervals
    """
    try:
        # Get recent historical data
        from modules.fmp_api import fmp_api
        historical_data = fmp_api.get_historical_price(symbol, period="1y")
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Try to load ensemble model first
        ensemble = EnsembleModel(prediction_days=days)
        
        if ensemble.load_model():
            # Make predictions using ensemble
            predictions = ensemble.predict(historical_data, days=days)
            
            if not predictions.empty:
                # Format results
                result = {
                    'symbol': symbol,
                    'model': 'ensemble',
                    'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                    'values': predictions['Close'].tolist(),
                    'confidence': {
                        'upper': predictions.get('Upper', None),
                        'lower': predictions.get('Lower', None)
                    }
                }
                
                return result
        
        # If ensemble model not available, try individual models in order of typical accuracy
        model_types = ['prophet', 'lstm', 'arima']
        
        for model_type in model_types:
            if model_type == 'prophet':
                model = ProphetModel(prediction_days=days, symbol=symbol)
            elif model_type == 'lstm':
                model = LSTMModel(prediction_days=days, symbol=symbol)
            else:  # arima
                model = ARIMAModel(prediction_days=days, symbol=symbol)
            
            if model.load_model():
                # Make predictions
                predictions = model.predict(historical_data, days=days)
                
                if not predictions.empty:
                    # Format results
                    result = {
                        'symbol': symbol,
                        'model': model_type,
                        'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                        'values': predictions['Close'].tolist(),
                        'confidence': {
                            'upper': predictions.get('Upper', None),
                            'lower': predictions.get('Lower', None)
                        }
                    }
                    
                    # Convert confidence intervals to lists if they exist
                    if result['confidence']['upper'] is not None:
                        result['confidence']['upper'] = predictions['Upper'].tolist()
                    if result['confidence']['lower'] is not None:
                        result['confidence']['lower'] = predictions['Lower'].tolist()
                    
                    return result
        
        # If no models are available, train a new one (Prophet is fastest to train)
        logger.info(f"No trained models available for {symbol}, training new Prophet model")
        model = ProphetModel(prediction_days=days)
        
        if model.train(historical_data):
            model.save_model()
            predictions = model.predict(historical_data, days=days)
            
            if not predictions.empty:
                # Format results
                result = {
                    'symbol': symbol,
                    'model': 'prophet',
                    'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                    'values': predictions['Close'].tolist(),
                    'confidence': {
                        'upper': predictions['Upper'].tolist() if 'Upper' in predictions.columns else None,
                        'lower': predictions['Lower'].tolist() if 'Lower' in predictions.columns else None
                    }
                }
                
                return result
        
        logger.error(f"Could not make predictions for {symbol}")
        return None
    
    except Exception as e:
        logger.error(f"Error getting price predictions for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def check_prophet_installed():
    """
    Check if Prophet is installed and provide installation instructions if not.
    
    Returns:
        bool: True if Prophet is installed, False otherwise
    """
    try:
        import prophet
        return True
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.error("""
        ---------------------------------------------------
        Prophet is not installed. This is required for model training.
        
        Install Prophet with:
        
        pip install prophet
        
        Note: Prophet has additional dependencies like Stan that may 
        require further installation steps on some systems.
        
        For more information, see:
        https://facebook.github.io/prophet/docs/installation.html
        ---------------------------------------------------
        """)
        return False

def create_fallback_prediction(symbol, days=30):
    """
    Create a fallback prediction when model training fails or is not available.
    This uses a simple moving average projection instead of ML models.
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days to predict
        
    Returns:
        dict: Simple prediction data
    """
    try:
        # Get historical data
        from modules.fmp_api import fmp_api
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Get recent historical data (last 90 days)
        historical_data = fmp_api.get_historical_price(symbol, period="90days")
        
        if historical_data.empty or len(historical_data) < 30:
            logger.error(f"Not enough historical data for fallback prediction for {symbol}")
            return None
        
        # Calculate average daily return over the past 30 days
        daily_returns = historical_data['Close'].pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        
        # Calculate standard deviation for confidence intervals
        std_dev = daily_returns.std()
        
        # Get the last closing price
        last_price = historical_data['Close'].iloc[-1]
        last_date = historical_data.index[-1]
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Calculate predicted prices using compound growth
        predicted_prices = [last_price * (1 + avg_daily_return)**(i+1) for i in range(days)]
        
        # Calculate upper and lower bounds (1 standard deviation)
        upper_bounds = [price * (1 + std_dev) for price in predicted_prices]
        lower_bounds = [price * (1 - std_dev) for price in predicted_prices]
        
        # Format as dictionary
        result = {
            'symbol': symbol,
            'model': 'simple_forecast',
            'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'values': predicted_prices,
            'confidence': {
                'upper': upper_bounds,
                'lower': lower_bounds
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating fallback prediction for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Modify get_price_predictions to use the fallback when needed
def get_price_predictions(symbol, days=30):
    """
    Get price predictions for a specific symbol using the best available model.
    
    Args:
        symbol (str): Stock symbol to predict
        days (int): Number of days to predict
    
    Returns:
        dict: Predictions with dates, values, and confidence intervals
    """
    try:
        # Get recent historical data
        from modules.fmp_api import fmp_api
        historical_data = fmp_api.get_historical_price(symbol, period="1y")
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Try to load ensemble model first
        ensemble = EnsembleModel(prediction_days=days)
        
        if ensemble.load_model():
            # Make predictions using ensemble
            predictions = ensemble.predict(historical_data, days=days)
            
            if not predictions.empty:
                # Format results
                result = {
                    'symbol': symbol,
                    'model': 'ensemble',
                    'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                    'values': predictions['Close'].tolist(),
                    'confidence': {
                        'upper': predictions.get('Upper', None),
                        'lower': predictions.get('Lower', None)
                    }
                }
                
                # Convert confidence intervals to lists if they exist
                if result['confidence']['upper'] is not None:
                    result['confidence']['upper'] = predictions['Upper'].tolist()
                if result['confidence']['lower'] is not None:
                    result['confidence']['lower'] = predictions['Lower'].tolist()
                
                return result
        
        # If ensemble model not available, try individual models in order of typical accuracy
        model_types = ['prophet', 'lstm', 'arima']
        
        for model_type in model_types:
            if model_type == 'prophet':
                model = ProphetModel(prediction_days=days)
            elif model_type == 'lstm':
                model = LSTMModel(prediction_days=days)
            else:  # arima
                model = ARIMAModel(prediction_days=days)
            
            if model.load_model():
                # Make predictions
                predictions = model.predict(historical_data, days=days)
                
                if not predictions.empty:
                    # Format results
                    result = {
                        'symbol': symbol,
                        'model': model_type,
                        'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                        'values': predictions['Close'].tolist(),
                        'confidence': {
                            'upper': predictions.get('Upper', None),
                            'lower': predictions.get('Lower', None)
                        }
                    }
                    
                    # Convert confidence intervals to lists if they exist
                    if result['confidence']['upper'] is not None:
                        result['confidence']['upper'] = predictions['Upper'].tolist()
                    if result['confidence']['lower'] is not None:
                        result['confidence']['lower'] = predictions['Lower'].tolist()
                    
                    return result
        
        # Use fallback prediction if no models are available
        logger.info(f"No trained models available for {symbol}, using fallback prediction")
        return create_fallback_prediction(symbol, days)
        
    except Exception as e:
        logger.error(f"Error in get_price_predictions for {symbol}: {e}")
        
        # Try the fallback as a last resort
        try:
            return create_fallback_prediction(symbol, days)
        except:
            logger.error(f"Even fallback prediction failed for {symbol}")
            return None
        
def train_price_prediction_models(symbol, lookback_period="1y"):
    """
    Train price prediction models for a specific symbol.
    
    Args:
        symbol (str): Stock symbol to train models for
        lookback_period (str): Historical period to use for training (e.g., "1y", "2y")
    
    Returns:
        dict: Trained models and performance metrics
    """
    try:
        # Check if Prophet is installed
        if not check_prophet_installed():
            raise ImportError("Prophet package is required for model training")
            
        # Get historical data
        from modules.fmp_api import fmp_api
        import logging
        logger = logging.getLogger(__name__)
        
        # Get historical price data
        historical_data = fmp_api.get_historical_price(symbol, period=lookback_period)
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Make sure we have the necessary columns
        required_columns = ['Close', 'Open', 'High', 'Low']
        for col in required_columns:
            if col not in historical_data.columns:
                # Try to map column names from lowercase if necessary
                lowercase_map = {'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}
                for lcol, ucol in lowercase_map.items():
                    if lcol in historical_data.columns:
                        historical_data[ucol] = historical_data[lcol]
        
        # Check if we have the required columns after mapping
        missing_cols = [col for col in required_columns if col not in historical_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Ensure we have enough data for training (at least 60 days)
        if len(historical_data) < 60:
            logger.error(f"Not enough historical data for {symbol} (got {len(historical_data)} rows, need at least 60)")
            return None
            
        # Split data into training and testing sets (80/20 split)
        split_idx = int(len(historical_data) * 0.8)
        train_data = historical_data.iloc[:split_idx]
        test_data = historical_data.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_data)} days, Testing data: {len(test_data)} days")
        
        # Make sure 'models' directory exists
        import os
        os.makedirs("models", exist_ok=True)
        
        # Train Prophet model
        try:
            from prophet import Prophet
            
            # Prophet requires 'ds' (date) and 'y' (target) columns
            prophet_data = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data['Close']
            })
            
            # Train Prophet model with basic settings
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_data)
            
            # Create future dataframe for 30 days
            future = model.make_future_dataframe(periods=30)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for future dates
            predictions = forecast[forecast['ds'] > historical_data.index[-1]][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Convert to DataFrame with date index
            result = pd.DataFrame({
                'Close': predictions['yhat'],
                'Lower': predictions['yhat_lower'],
                'Upper': predictions['yhat_upper']
            }, index=pd.DatetimeIndex(predictions['ds']))
            
            # Calculate actual metrics using the test data
            metrics = calculate_prophet_metrics(model, test_data)
            
            # Create a structured model result
            model_result = {
                'prophet': {
                    'model': model,
                    'metrics': metrics
                }
            }
            
            # Save the model
            import pickle
            os.makedirs("models", exist_ok=True)
            with open(f"models/prophet_{symbol}.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Successfully trained and saved Prophet model for {symbol}")
            logger.info(f"Metrics for {symbol}: {metrics}")
            
            return model_result
            
        except Exception as prophet_error:
            logger.error(f"Error training Prophet model: {prophet_error}")
            import traceback
            traceback.print_exc()
            
            # If Prophet fails, return a simplified non-dictionary result
            # that can be handled by the ModelIntegration class
            return {
                'prophet': {
                    'model': None,
                    'metrics': {
                        'mae': 0.0,
                        'mse': 0.0,
                        'rmse': 0.0,
                        'mape': 0.0,
                        'error': str(prophet_error)
                    }
                }
            }
    
    except Exception as e:
        logger.error(f"Error in train_price_prediction_models for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None