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
    Base class for price prediction models with fixed symbol handling.
    """
    def __init__(self, model_name="base", prediction_days=30, symbol=None):
        self.model_name = model_name
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.model_dir = "models"
        self.symbol = symbol  # Properly store the symbol
        
        # Print the symbol for debugging
        print(f"Initializing {model_name} model with symbol: {symbol}")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_model(self):
        """
        Load model from disk with improved error handling.
        
        Returns:
            bool: Success status
        """
        try:
            # Print debugging info
            print(f"In load_model for {self.model_name}, symbol is: {self.symbol}")
            
            if not hasattr(self, 'symbol') or not self.symbol:
                print(f"Symbol not provided for {self.model_name} model")
                return False
            
            # Create symbol-specific model path
            model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.symbol}.pkl")
            
            print(f"Looking for model at: {os.path.abspath(model_path)}")
            
            # Check if the file exists
            if os.path.exists(model_path):
                print(f"Found model file for {self.symbol}: {model_path}")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"Successfully loaded model for {self.symbol}")
                return True
                
            # List available model files for debugging
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            print(f"Available model files: {model_files}")
            
            # Only look for matching models of the current type
            # This prevents loading the wrong model type
            matching_model_files = [f for f in model_files if f.startswith(f"{self.model_name}_")]
            symbol_models = [f for f in matching_model_files if self.symbol in f]
            
            if symbol_models:
                symbol_model_path = os.path.join(self.model_dir, symbol_models[0])
                print(f"Found matching {self.model_name} model file: {symbol_models[0]}")
                with open(symbol_model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"Successfully loaded {self.model_name} model for {self.symbol}")
                return True
            
            print(f"No {self.model_name} model file found for {self.symbol}")
            return False
            
        except Exception as e:
            print(f"Error loading {self.model_name} model for {self.symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False


class ARIMAModel(PricePredictionModel):
    """
    Price prediction model using ARIMA (AutoRegressive Integrated Moving Average).
    """
    def __init__(self, prediction_days=30, order=(5,1,0), symbol=None):
        super().__init__(model_name="arima", prediction_days=prediction_days, symbol=symbol)
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
    def __init__(self, prediction_days=30, symbol=None):
        super().__init__(model_name="prophet", prediction_days=prediction_days, symbol=symbol)
        # self.symbol = symbol
    
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
    def __init__(self, prediction_days=30, lookback=60, units=50, epochs=50, batch_size=32, symbol=None):
        super().__init__(model_name="lstm", prediction_days=prediction_days, symbol=symbol)
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
    def __init__(self, prediction_days=30, models=None, weights=None, symbol=None):
        super().__init__(model_name="ensemble", prediction_days=prediction_days, symbol=symbol)
        
        # Initialize models if provided, otherwise use default (ARIMA, Prophet, LSTM)
        if models is None:
            self.models = [
                ARIMAModel(prediction_days=prediction_days, symbol=symbol),
                ProphetModel(prediction_days=prediction_days, symbol=symbol),
                LSTMModel(prediction_days=prediction_days, symbol=symbol)
            ]
        else:
            # Make sure each model gets the symbol
            for model in models:
                if hasattr(model, 'symbol'):
                    model.symbol = symbol
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
        valid_models = []
        
        for i, model in enumerate(self.models):
            # Only use models of the correct type
            # This prevents errors from trying to use the wrong model type
            model_class_name = model.__class__.__name__
            expected_model_name = model.model_name.capitalize() + "Model"
            
            if model_class_name == expected_model_name:
                if model.is_trained or model.load_model():
                    try:
                        pred = model.predict(historical_data, days=pred_days)
                        if not pred.empty:
                            all_predictions.append(pred)
                            valid_models.append(model)
                            logger.info(f"Got predictions from {model.model_name} model")
                        else:
                            logger.warning(f"No predictions from {model.model_name} model")
                    except Exception as e:
                        logger.error(f"Error getting predictions from {model.model_name} model: {e}")
                else:
                    logger.warning(f"{model.model_name} model not trained or could not be loaded")
            else:
                logger.warning(f"Model type mismatch: {model_class_name} vs expected {expected_model_name}")
        
        if not all_predictions:
            logger.error("No predictions available from any model")
            return pd.DataFrame()
        
        # Align predictions to the same dates
        common_index = all_predictions[0].index
        for pred in all_predictions[1:]:
            common_index = common_index.intersection(pred.index)
        
        # Get weighted average of predictions
        weighted_sum = pd.Series(0, index=common_index)
        
        # Adjust weights based on valid models
        if valid_models:
            adjusted_weights = [self.weights[self.models.index(model)] for model in valid_models]
            # Normalize weights
            weight_sum = sum(adjusted_weights)
            if weight_sum > 0:
                adjusted_weights = [w / weight_sum for w in adjusted_weights]
            else:
                # Equal weights if sum is zero
                adjusted_weights = [1.0/len(valid_models)] * len(valid_models)
            
            # Apply weights to predictions
            for i, pred in enumerate(all_predictions):
                if pred.index.equals(common_index):
                    weighted_sum += pred['Close'] * adjusted_weights[i]
                else:
                    # Reindex prediction to common index
                    reindexed_pred = pred.reindex(common_index)
                    weighted_sum += reindexed_pred['Close'] * adjusted_weights[i]
        else:
            # If no valid models, return empty DataFrame
            return pd.DataFrame()
        
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
    # Input validation for symbol parameter
    if not symbol:
        error_msg = "Symbol parameter cannot be None or empty"
        logger.error(error_msg)
        return {"error": error_msg}
        
    try:
        # Check if Prophet is installed
        if not check_prophet_installed():
            raise ImportError("Prophet package is required for model training")
            
        logger.info(f"Starting model training for symbol: {symbol}")
            
        # Get historical data
        from modules.fmp_api import fmp_api
        
        # Get historical price data
        historical_data = fmp_api.get_historical_price(symbol, period=lookback_period)
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return {"error": f"No historical data available for {symbol}"}
        
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
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Ensure we have enough data for training (at least 60 days)
        if len(historical_data) < 60:
            logger.error(f"Not enough historical data for {symbol} (got {len(historical_data)} rows, need at least 60)")
            return {"error": f"Not enough historical data (got {len(historical_data)} rows, need at least 60)"}
            
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
            
            # Save the model WITH THE SYMBOL in the filename (this is critical)
            import pickle
            model_filename = f"prophet_{symbol}.pkl"
            model_path = os.path.join("models", model_filename)
            
            # Log the exact path we're saving to
            logger.info(f"Saving model to path: {os.path.abspath(model_path)}")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Successfully trained and saved Prophet model for {symbol} as {model_filename}")
            logger.info(f"Metrics for {symbol}: {metrics}")
            
            return model_result
            
        except Exception as prophet_error:
            logger.error(f"Error training Prophet model: {prophet_error}")
            import traceback
            traceback.print_exc()
            
            # If Prophet fails, return a simplified result with error information
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
        return {"error": str(e)}

def get_price_predictions(symbol, days=30):
    """
    Get price predictions for a specific symbol using the best available model.
    
    Args:
        symbol (str): Stock symbol to predict
        days (int): Number of days to predict
    
    Returns:
        dict: Predictions with dates, values, and confidence intervals
    """
    print(f"get_price_predictions called with symbol: {symbol}, days: {days}")
    
    if not symbol:
        logger.error("Symbol parameter cannot be None or empty")
        return None
        
    try:
        # Get recent historical data
        from modules.fmp_api import fmp_api
        historical_data = fmp_api.get_historical_price(symbol, period="1y")
        
        if historical_data.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Check for a specific Prophet model file first since it's our primary model
        import os
        model_dir = "models"
        prophet_path = os.path.join(model_dir, f"prophet_{symbol}.pkl")
        
        print(f"Checking for specific model file: {prophet_path}")
        
        if os.path.exists(prophet_path):
            print(f"Found existing model for {symbol}")
            
            # Direct loading approach with Prophet model
            try:
                import pickle
                with open(prophet_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Make sure it's actually a Prophet model
                from prophet import Prophet
                if isinstance(model, Prophet):
                    # Create future dataframe for the prediction days
                    print(f"Creating forecast for {days} days")
                    future = model.make_future_dataframe(periods=days)
                    
                    # Make predictions
                    forecast = model.predict(future)
                    
                    # Extract predictions for future dates (only the specified number of days)
                    future_cutoff = historical_data.index[-1]
                    future_forecast = forecast[forecast['ds'] > future_cutoff]
                    
                    if len(future_forecast) > days:
                        # Take only the requested number of days
                        predictions = future_forecast.iloc[:days]
                    else:
                        # Take all available future predictions
                        predictions = future_forecast
                    
                    if len(predictions) > 0:
                        # Format results
                        result = {
                            'symbol': symbol,
                            'model': 'prophet',
                            'dates': predictions['ds'].dt.strftime('%Y-%m-%d').tolist(),
                            'values': predictions['yhat'].tolist(),
                            'confidence': {
                                'upper': predictions['yhat_upper'].tolist(),
                                'lower': predictions['yhat_lower'].tolist()
                            }
                        }
                        
                        print(f"Successfully generated predictions for {symbol} using direct Prophet loading ({len(result['dates'])} days)")
                        return handle_nan_values(result)
                    else:
                        print(f"No future predictions generated for {symbol}")
                else:
                    print(f"Loaded model is not a Prophet model: {type(model)}")
            except Exception as e:
                print(f"Error with direct Prophet model loading: {e}")
                import traceback
                traceback.print_exc()
                # Continue to try other approaches
        
        # If direct approach failed, use the standalone ProphetModel rather than EnsembleModel
        print(f"Trying standalone ProphetModel for {symbol}")
        
        # Initialize ProphetModel with explicit symbol
        prophet_model = ProphetModel(prediction_days=days, symbol=symbol)
        
        # Try to load and use the model
        if prophet_model.load_model():
            print(f"Successfully loaded Prophet model for {symbol}")
            predictions = prophet_model.predict(historical_data, days=days)
            
            if not predictions.empty:
                # Format results
                result = {
                    'symbol': symbol,
                    'model': 'prophet',
                    'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                    'values': predictions['Close'].tolist()
                }
                
                # Add confidence intervals if they exist
                if 'Upper' in predictions.columns and 'Lower' in predictions.columns:
                    result['confidence'] = {
                        'upper': predictions['Upper'].tolist(),
                        'lower': predictions['Lower'].tolist()
                    }
                else:
                    # Create simple confidence intervals if not provided
                    values = predictions['Close'].tolist()
                    result['confidence'] = {
                        'upper': [val * 1.05 for val in values],
                        'lower': [val * 0.95 for val in values]
                    }
                
                print(f"Successfully generated predictions for {symbol} using ProphetModel ({len(result['dates'])} days)")
                return handle_nan_values(result)
        
        # If Prophet failed, try other models individually (no ensemble)
        print(f"Trying other model types for {symbol}")
        
        models_to_try = [
            ARIMAModel(prediction_days=days, symbol=symbol),
            LSTMModel(prediction_days=days, symbol=symbol)
        ]
        
        for model in models_to_try:
            if model.load_model():
                print(f"Successfully loaded {model.model_name} model for {symbol}")
                predictions = model.predict(historical_data, days=days)
                
                if not predictions.empty:
                    # Format results
                    result = {
                        'symbol': symbol,
                        'model': model.model_name,
                        'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                        'values': predictions['Close'].tolist()
                    }
                    
                    # Add confidence intervals if they exist
                    if 'Upper' in predictions.columns and 'Lower' in predictions.columns:
                        result['confidence'] = {
                            'upper': predictions['Upper'].tolist(),
                            'lower': predictions['Lower'].tolist()
                        }
                    else:
                        # Create simple confidence intervals if not provided
                        values = predictions['Close'].tolist()
                        result['confidence'] = {
                            'upper': [val * 1.05 for val in values],
                            'lower': [val * 0.95 for val in values]
                        }
                    
                    print(f"Successfully generated predictions for {symbol} using {model.model_name} ({len(result['dates'])} days)")
                    return handle_nan_values(result)
        
        # Use fallback prediction if no models are available
        print(f"No trained models available for {symbol}, using fallback prediction with {days} days")
        return handle_nan_values(create_fallback_prediction(symbol, days))
        
    except Exception as e:
        print(f"Error in get_price_predictions for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try the fallback as a last resort
        try:
            fallback_result = create_fallback_prediction(symbol, days)
            return handle_nan_values(fallback_result)
        except Exception as fallback_error:
            print(f"Even fallback prediction failed for {symbol}: {fallback_error}")
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
    print(f"Using fallback prediction for {symbol} with {days} days")
    
    try:
        # Get historical data
        from modules.fmp_api import fmp_api
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Get recent historical data (last 90 days)
        historical_data = fmp_api.get_historical_price(symbol, period="90days")
        
        if historical_data.empty or len(historical_data) < 30:
            print(f"Not enough historical data for fallback prediction for {symbol}")
            # Create a very basic fallback if no data
            last_price = 100.0  # Default price
            last_date = datetime.now()
            # Use very small random changes
            predicted_prices = [last_price * (1 + np.random.normal(0.0001, 0.001)) for _ in range(days)]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Format as dictionary
            result = {
                'symbol': symbol,
                'model': 'basic_fallback',
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'values': predicted_prices,
                'confidence': {
                    'upper': [price * 1.05 for price in predicted_prices],
                    'lower': [price * 0.95 for price in predicted_prices]
                }
            }
            
            return result
        
        # Calculate average daily return over the past 30 days
        daily_returns = historical_data['Close'].pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        
        # Calculate standard deviation for confidence intervals
        std_dev = daily_returns.std()
        
        # Get the last closing price
        last_price = historical_data['Close'].iloc[-1]
        last_date = historical_data.index[-1]
        
        # Generate future dates - use the specified number of days
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
        
        print(f"Successfully created fallback prediction for {symbol} with {days} days")
        return result
        
    except Exception as e:
        print(f"Error creating fallback prediction for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        
        # Last resort: create a minimal prediction
        try:
            # Generate very basic synthetic data
            last_price = 100.0  # Arbitrary price
            last_date = datetime.now()
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            predicted_prices = [last_price * (1 + 0.001 * i) for i in range(days)]
            
            result = {
                'symbol': symbol,
                'model': 'minimal_fallback',
                'dates': [date.strftime('%Y-%m-%d') for date in future_dates],
                'values': predicted_prices,
                'confidence': {
                    'upper': [price * 1.05 for price in predicted_prices],
                    'lower': [price * 0.95 for price in predicted_prices]
                }
            }
            
            return result
        except:
            return None

# Modify get_price_predictions to use the fallback when needed
# This fixes the get_asset_analysis method in the ModelIntegration class to properly pass
# the symbol to prediction functions

def get_asset_analysis(self, symbol, days_to_predict=30, force_refresh=False):
    """
    Get comprehensive analysis for a specific asset, including price predictions
    and trend analysis.
    
    Args:
        symbol (str): Asset symbol
        days_to_predict (int): Number of days to predict prices for
        force_refresh (bool): Force refresh of cached data
    
    Returns:
        dict: Comprehensive asset analysis
    """
    # Debug output
    print(f"get_asset_analysis called for symbol: {symbol}")
    
    # Check if we have recent cached results
    cache_key = f"{symbol}_analysis"
    if cache_key in self.prediction_cache and not force_refresh:
        cache_time, data = self.prediction_cache[cache_key]
        # Return cached data if less than 6 hours old
        if (datetime.now() - cache_time).total_seconds() < 21600:  # 6 hours
            return data
    
    try:
        # Get historical data for analysis
        historical_data = self.data_collector.get_market_data(symbols=[symbol], timeframe="1y")
        
        if symbol not in historical_data or historical_data[symbol].empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        # Run trend analysis
        df = historical_data[symbol]
        trend_analysis = self.trend_analyzer.detect_trend(df)
        support_resistance = self.trend_analyzer.identify_support_resistance(df)
        patterns = self.trend_analyzer.detect_patterns(df)
        breakout = self.trend_analyzer.predict_breakout(df, support_resistance)
        market_regime = self.trend_analyzer.get_market_regime(df)
        
        # Get price predictions with explicit symbol passing
        from modules.price_prediction import get_price_predictions
        predictions = get_price_predictions(symbol=symbol, days=days_to_predict)
        print(f"Predictions received for {symbol}: {predictions is not None}")
        
        # Check if model training is in progress
        training_status = self.model_training_status.get(symbol, "not_started")
        
        # Combine all analyses
        analysis = {
            'symbol': symbol,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price': {
                'current': df['Close'].iloc[-1],
                'change_1d': (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0,
                'change_1w': (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) > 5 else 0,
                'change_1m': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 if len(df) > 20 else 0,
            },
            'trend': trend_analysis,
            'support_resistance': support_resistance,
            'patterns': patterns,
            'breakout': breakout,
            'market_regime': market_regime,
            'price_predictions': predictions,
            'model_training': {
                'status': training_status,
                'last_updated': self.model_training_status.get(f"{symbol}_updated", "never")
            }
        }
        
        # Cache the results
        self.prediction_cache[cache_key] = (datetime.now(), analysis)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in get_asset_analysis for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def handle_nan_values(prediction_data):
    """
    Clean prediction data by handling NaN values.
    
    Args:
        prediction_data (dict): The prediction data dictionary
        
    Returns:
        dict: Cleaned prediction data with NaN values removed or replaced
    """
    import numpy as np
    import pandas as pd
    
    if not prediction_data:
        return None
    
    # Make a copy to avoid modifying the original
    cleaned_data = prediction_data.copy()
    
    # Check and clean values
    if 'values' in cleaned_data:
        # Convert values to float and replace NaN with None
        cleaned_values = []
        for val in cleaned_data['values']:
            try:
                float_val = float(val)
                if np.isnan(float_val):
                    # For plotting, we'll use the previous valid value or the first historical price
                    continue
                cleaned_values.append(float_val)
            except (ValueError, TypeError):
                continue
        
        # If we lost all values, return None
        if not cleaned_values:
            return None
            
        # Update the values list
        cleaned_data['values'] = cleaned_values
        
        # Ensure dates and values have same length
        if 'dates' in cleaned_data and len(cleaned_data['dates']) != len(cleaned_values):
            # Keep only the dates corresponding to valid values
            cleaned_data['dates'] = cleaned_data['dates'][:len(cleaned_values)]
    
    # Check and clean confidence intervals
    if 'confidence' in cleaned_data:
        confidence = cleaned_data['confidence']
        if confidence:
            # Clean upper bounds
            if 'upper' in confidence:
                cleaned_upper = []
                for val in confidence['upper']:
                    try:
                        float_val = float(val)
                        if np.isnan(float_val):
                            continue
                        cleaned_upper.append(float_val)
                    except (ValueError, TypeError):
                        continue
                confidence['upper'] = cleaned_upper
            
            # Clean lower bounds
            if 'lower' in confidence:
                cleaned_lower = []
                for val in confidence['lower']:
                    try:
                        float_val = float(val)
                        if np.isnan(float_val):
                            continue
                        cleaned_lower.append(float_val)
                    except (ValueError, TypeError):
                        continue
                confidence['lower'] = cleaned_lower
            
            # Update confidence data
            cleaned_data['confidence'] = confidence
    
    return cleaned_data