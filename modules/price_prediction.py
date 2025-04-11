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
from modules.data_provider import data_provider

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
        
class PricePredictionModel:
    """Base class for price prediction models with fixed symbol handling."""
    def __init__(self, model_name="base", prediction_days=30, symbol=None):
        self.model_name = model_name
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.model_dir = "models"
        self.symbol = symbol # Store the symbol
        os.makedirs(self.model_dir, exist_ok=True)
        logger.debug(f"Initializing {model_name} model with symbol: {symbol}") # Use logger

    def load_model(self):
        """Load model from disk with improved error handling."""
        try:
            if not hasattr(self, 'symbol') or not self.symbol:
                logger.error(f"Symbol not provided for {self.model_name} model during load") # Use logger
                return False

            model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.symbol}.pkl")
            logger.debug(f"Looking for model at: {os.path.abspath(model_path)}") # Use logger

            if os.path.exists(model_path):
                logger.info(f"Found model file for {self.symbol}: {model_path}") # Use logger
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Successfully loaded model for {self.symbol}") # Use logger
                return True
            else:
                logger.warning(f"No {self.model_name} model file found for {self.symbol} at {model_path}") # Use logger
                return False
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model for {self.symbol}: {e}") # Use logger
            logger.error(traceback.format_exc()) # Log traceback
            return False
        
    def preprocess_data(self, historical_data):
        """Preprocess data using the scaler."""
        if 'close' not in historical_data.columns:
            raise ValueError("Data must contain 'close' column")
        close_prices = historical_data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        return scaled_data

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
            if 'close' in data.columns:
                close_prices = data['close']
            else:
                raise ValueError("Data must contain 'close' column")
            
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
            predictions = pd.DataFrame(forecast, index=future_dates, columns=['close'])
            
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
            mae = mean_absolute_error(test_data['close'], predictions)
            mse = mean_squared_error(test_data['close'], predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((test_data['close'] - predictions) / test_data['close'])) * 100
            
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
    """Price prediction model using Facebook Prophet."""
    def __init__(self, prediction_days=30, symbol=None):
        # --- Ensure super().__init__ is called correctly ---
        super().__init__(model_name="prophet", prediction_days=prediction_days, symbol=symbol)

    def train(self, historical_data):
        """Train Prophet model on historical data."""
        try:
            from prophet import Prophet
            data = historical_data.sort_index().copy()
            if 'close' not in data.columns: raise ValueError("Data must contain 'close' column")

            prophet_data = pd.DataFrame({'ds': data.index, 'y': data['close']})
            self.model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            self.model.fit(prophet_data)
            self.is_trained = True
            logger.info(f"Prophet model trained successfully for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error training Prophet model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def predict(self, historical_data, days=None):
        """Make predictions using Prophet model."""
        if not self.is_trained and not self.load_model():
            logger.error(f"Prophet model for {self.symbol} not trained and could not be loaded")
            return pd.DataFrame()
        try:
            pred_days = days if days is not None else self.prediction_days
            hist_data = historical_data.copy()
            if not isinstance(hist_data.index, pd.DatetimeIndex): raise ValueError("Historical data must have DatetimeIndex")

            future = self.model.make_future_dataframe(periods=pred_days)
            forecast = self.model.predict(future)
            last_historical_date = hist_data.index[-1]
            future_predictions = forecast[forecast['ds'] > last_historical_date].copy()

            if future_predictions.empty:
                logger.error(f"No future predictions generated by Prophet for {self.symbol}")
                return pd.DataFrame()

            result = pd.DataFrame({
                'Close': future_predictions['yhat'],
                'Upper': future_predictions['yhat_upper'],
                'Lower': future_predictions['yhat_lower']
            }, index=pd.DatetimeIndex(future_predictions['ds']))

            logger.info(f"Prophet prediction successful for {self.symbol}: {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error making predictions with Prophet model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def evaluate(self, test_data):
        """Evaluate Prophet model performance."""
        if not self.is_trained and not self.load_model():
            logger.error(f"Prophet model for {self.symbol} not trained and could not be loaded")
            return {}
        try:
            # --- Call the standalone calculate_prophet_metrics function ---
            return calculate_prophet_metrics(self.model, test_data)
        except Exception as e:
            logger.error(f"Error evaluating Prophet model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
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
    Calculate evaluation metrics for a Prophet model.
    Ensures it uses the 'close' column and handles alignment issues.
    """
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # Ensure 'close' column exists (standardized name)
        if 'close' not in test_data.columns:
             raise ValueError("Test data must contain 'close' column for metrics calculation")

        prophet_data = pd.DataFrame({'ds': test_data.index})
        predictions = model.predict(prophet_data)

        # Align actual and predicted values using merge on date index
        merged_data = pd.merge(test_data[['close']], predictions[['ds', 'yhat']], left_index=True, right_on='ds', how='inner')

        if merged_data.empty or len(merged_data) < 2:
             logger.warning("No matching dates or insufficient overlap between test data and predictions for metrics calculation.")
             return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': 0.0, 'error': 'No matching dates or insufficient overlap'}

        actual = merged_data['close'].values
        predicted = merged_data['yhat'].values

        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        # Calculate MAPE safely
        non_zero_indices = actual != 0
        if np.any(non_zero_indices):
            mape = np.mean(np.abs((actual[non_zero_indices] - predicted[non_zero_indices]) / actual[non_zero_indices])) * 100
        else:
            mape = 0.0 # Avoid division by zero if all actual values are zero

        return {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'mape': float(mape)}

    except Exception as e:
        logger.error(f"Error calculating Prophet metrics: {e}")
        logger.error(traceback.format_exc())
        return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': 0.0, 'error': str(e)}

# Make sure this is AFTER the function definition, not before
def train_price_prediction_models(symbol, lookback_period="1y"):
    """
    Train price prediction models (currently Prophet) for a specific symbol.
    Saves the model file and returns necessary info for metadata saving.

    Args:
        symbol (str): Stock symbol to train models for
        lookback_period (str): Historical period to use for training (e.g., "1y", "2y")

    Returns:
        tuple: (model_object, metrics_dict, model_type_str, model_filename_str)
               Returns (None, {'error': msg}, None, None) on failure.
    """
    if not symbol:
        error_msg = "Symbol parameter cannot be None or empty"
        logger.error(error_msg)
        return None, {"error": error_msg}, None, None # Return consistent tuple

    try:
        # Check if Prophet is installed
        if not check_prophet_installed():
            raise ImportError("Prophet package is required for model training")

        logger.info(f"Starting model training for symbol: {symbol}")

        # Get historical data using DataProvider (standardized)
        historical_data = data_provider.get_historical_price(symbol, period=lookback_period)
        if historical_data.empty:
            raise ValueError(f"No historical data available for {symbol} via DataProvider")

        # Ensure required columns ('close') and enough data
        if 'close' not in historical_data.columns:
            raise ValueError("Missing required column: 'close'")
        if len(historical_data) < 60: # Need sufficient data for train/test split
            raise ValueError(f"Not enough historical data for {symbol} (got {len(historical_data)} rows, need at least 60)")

        # Split data (80/20)
        split_idx = int(len(historical_data) * 0.8)
        train_data = historical_data.iloc[:split_idx]
        test_data = historical_data.iloc[split_idx:]
        logger.info(f"Training data: {len(train_data)} days, Testing data: {len(test_data)} days")

        # --- Train Prophet Model ---
        try:
            # Instantiate the ProphetModel class
            prophet_trainer = ProphetModel(symbol=symbol) # Pass symbol here

            # Train the model using its train method
            train_success = prophet_trainer.train(train_data)

            if not train_success:
                raise RuntimeError("Prophet model training failed.")

            # Evaluate the trained model
            metrics = prophet_trainer.evaluate(test_data)
            if not metrics or 'error' in metrics:
                 logger.warning(f"Metrics calculation failed or returned error for {symbol}. Metrics: {metrics}")
                 # Decide if this is critical - maybe proceed but log error in metrics?
                 if isinstance(metrics, dict) and 'error' in metrics:
                     pass # Keep error metrics
                 else:
                     metrics = {'error': 'Metrics calculation failed'}

            # --- Save the Prophet model ---
            model_type = 'prophet' # Define model type
            model_filename = f"{model_type}_{symbol}.pkl"
            model_dir = "models" # Ensure this matches ModelIntegration
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, model_filename)

            logger.info(f"Saving Prophet model to path: {os.path.abspath(model_path)}")
            with open(model_path, 'wb') as f:
                # --- Save the TRAINED model object from the class instance ---
                pickle.dump(prophet_trainer.model, f)
            logger.info(f"Successfully trained and saved Prophet model for {symbol} as {model_filename}")

            # --- Return the necessary info ---
            # Return the trained model object itself, metrics, type, and filename
            return prophet_trainer.model, metrics, model_type, model_filename

        except Exception as prophet_error:
            logger.error(f"Error during Prophet model training/saving for {symbol}: {prophet_error}")
            logger.error(traceback.format_exc())
            # Return None and error metrics
            return None, {'error': str(prophet_error)}, 'prophet', None # Keep type if known

    except Exception as e:
        logger.error(f"Error in train_price_prediction_models for {symbol}: {e}")
        logger.error(traceback.format_exc())
        # Return None and error metrics
        return None, {"error": str(e)}, None, None

def get_price_predictions(symbol, days=30):
    """Get price predictions, using Prophet model with fallbacks."""
    logger.info(f"Getting price predictions for {symbol} for {days} days.")
    if not symbol: return create_simple_fallback("UNKNOWN", days)

    try:
        historical_data = data_provider.get_historical_price(symbol, period="1y")
        if historical_data.empty:
            logger.error(f"No historical data for {symbol} in get_price_predictions.")
            return create_fallback_prediction(symbol, days)

        # --- Use ProphetModel class instance ---
        prophet_predictor = ProphetModel(symbol=symbol, prediction_days=days)
        predictions_df = pd.DataFrame()

        if prophet_predictor.load_model():
            logger.info(f"Loaded existing Prophet model for {symbol}.")
            predictions_df = prophet_predictor.predict(historical_data, days=days)
        else:
            logger.warning(f"No pre-trained Prophet model found for {symbol}. Using fallback.")
            # Optionally trigger training here if desired, but for prediction, use fallback
            # self.train_models_for_symbol(symbol) # This would block or run async
            return create_fallback_prediction(symbol, days)

        # --- Process predictions ---
        if not predictions_df.empty:
            result = {
                'symbol': symbol, 'model': 'prophet',
                'dates': predictions_df.index.strftime('%Y-%m-%d').tolist(),
                'values': predictions_df['Close'].tolist()
            }
            if 'Upper' in predictions_df.columns and 'Lower' in predictions_df.columns:
                result['confidence'] = {'upper': predictions_df['Upper'].tolist(), 'lower': predictions_df['Lower'].tolist()}
            else: # Add simple confidence if missing
                result['confidence'] = {'upper': [v*1.05 for v in result['values']], 'lower': [v*0.95 for v in result['values']]}

            cleaned_result = handle_nan_values(result)
            return cleaned_result if cleaned_result else create_simple_fallback(symbol, days)
        else:
            logger.warning(f"Prophet prediction returned empty DataFrame for {symbol}. Using fallback.")
            return create_fallback_prediction(symbol, days)

    except Exception as e:
        logger.error(f"Error in get_price_predictions for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return create_simple_fallback(symbol, days) # Simplest fallback on error

def check_prophet_installed():
    """Check if Prophet is installed."""
    try:
        import prophet
        return True
    except ImportError:
        logger.error("Prophet is not installed. Required for model training. Run: pip install prophet")
        return False

def create_simple_fallback(symbol, days=30):
    """Create a very simple fallback prediction."""
    logger.warning(f"Using SIMPLE fallback prediction for {symbol}")
    try:
        last_price = 100.0 # Default base price
        try:
            quote = data_provider.get_current_quote(symbol)
            if quote and quote.get('price'): last_price = float(quote['price'])
        except: pass # Ignore errors getting current price for fallback

        last_date = datetime.now()
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        values = [last_price] * days # Flat prediction

        return {
            'symbol': symbol, 'model': 'simple_fallback', 'dates': future_dates, 'values': values,
            'confidence': {'upper': [v * 1.05 for v in values], 'lower': [v * 0.95 for v in values]}
        }
    except Exception as e:
         logger.error(f"CRITICAL ERROR in simple fallback for {symbol}: {e}")
         # Absolute last resort
         return {'symbol': symbol, 'model': 'error_fallback', 'dates': [], 'values': [], 'confidence': {}}

def create_fallback_prediction(symbol, days=30):
    """Create a fallback prediction using simple projection."""
    logger.warning(f"Using standard fallback prediction for {symbol}")
    try:
        historical_data = data_provider.get_historical_price(symbol, period="90d")
        if historical_data.empty or len(historical_data) < 10 or 'close' not in historical_data.columns:
            logger.error(f"Insufficient data for standard fallback for {symbol}")
            return create_simple_fallback(symbol, days)

        last_price = historical_data['close'].iloc[-1]
        last_date = historical_data.index[-1]
        daily_returns = historical_data['close'].pct_change().dropna()
        avg_daily_return = np.clip(daily_returns.mean(), -0.005, 0.005) # Bounded average
        std_dev = np.clip(daily_returns.std(), 0.005, 0.02) # Bounded std dev

        future_dates = [(last_date + timedelta(days=i+1)) for i in range(days)]
        predicted_prices = []
        current_price = last_price
        for _ in range(days):
            daily_return = np.clip(avg_daily_return + np.random.normal(0, std_dev/3), -0.01, 0.01) # Add noise, clip
            current_price *= (1 + daily_return)
            predicted_prices.append(current_price)

        return {
            'symbol': symbol, 'model': 'fallback_projection',
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'values': predicted_prices,
            'confidence': {'upper': [p * (1 + std_dev) for p in predicted_prices], 'lower': [p * (1 - std_dev) for p in predicted_prices]}
        }
    except Exception as e:
        logger.error(f"Error creating standard fallback prediction for {symbol}: {e}")
        return create_simple_fallback(symbol, days) # Use simplest if this fails


        
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
                'current': df['close'].iloc[-1],
                'change_1d': (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0,
                'change_1w': (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) > 5 else 0,
                'change_1m': (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100 if len(df) > 20 else 0,
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
    """Clean prediction data by handling NaN values."""
    if not prediction_data: return None
    cleaned_data = prediction_data.copy()
    cleaned_values = []
    valid_indices = []

    if 'values' in cleaned_data and isinstance(cleaned_data['values'], list):
        for i, val in enumerate(cleaned_data['values']):
            try:
                float_val = float(val)
                if not pd.isna(float_val):
                    cleaned_values.append(float_val)
                    valid_indices.append(i)
            except (ValueError, TypeError): continue # Skip non-numeric

        if not cleaned_values: return None # All values were invalid

        cleaned_data['values'] = cleaned_values
        if 'dates' in cleaned_data and isinstance(cleaned_data['dates'], list):
            cleaned_data['dates'] = [cleaned_data['dates'][i] for i in valid_indices if i < len(cleaned_data['dates'])]
        if 'confidence' in cleaned_data and isinstance(cleaned_data['confidence'], dict):
            if 'upper' in cleaned_data['confidence'] and isinstance(cleaned_data['confidence']['upper'], list):
                 cleaned_data['confidence']['upper'] = [cleaned_data['confidence']['upper'][i] for i in valid_indices if i < len(cleaned_data['confidence']['upper'])]
            if 'lower' in cleaned_data['confidence'] and isinstance(cleaned_data['confidence']['lower'], list):
                 cleaned_data['confidence']['lower'] = [cleaned_data['confidence']['lower'][i] for i in valid_indices if i < len(cleaned_data['confidence']['lower'])]
    return cleaned_data
