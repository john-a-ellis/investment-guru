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
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA as ARIMA_Statsmodels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PricePredictionModel:
    """Base class for price prediction models with fixed symbol handling."""
    def __init__(self, model_name="base", prediction_days=30, symbol=None):
        self.model_name = model_name
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) # Scaler for LSTM
        self.is_trained = False
        self.model_dir = "models"
        self.symbol = symbol # Store the symbol
        os.makedirs(self.model_dir, exist_ok=True)
        logger.debug(f"Initializing {model_name} model with symbol: {symbol}")

    def get_model_filename(self):
        """Generates the standard model filename."""
        if not self.symbol:
            return None
        # Default to .pkl, override in subclasses if needed
        return f"{self.model_name}_{self.symbol}.pkl"

    def get_scaler_filename(self):
        """Generates the standard scaler filename (used by LSTM)."""
        if not self.symbol:
            return None
        return f"{self.model_name}_{self.symbol}_scaler.pkl"

    def save_model(self):
        """Save model to disk. Base implementation for pickle."""
        if not self.is_trained:
            logger.warning(f"{self.model_name} model for {self.symbol} not trained, cannot save")
            return None # Return None for filename on failure
        if not self.symbol:
            logger.error(f"Symbol not provided for {self.model_name} model during save")
            return None

        model_filename = self.get_model_filename()
        if not model_filename: return None
        model_path = os.path.join(self.model_dir, model_filename)

        try:
            logger.info(f"Saving {self.model_name} model for {self.symbol} to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"{self.model_name} model saved successfully for {self.symbol}")
            return model_filename # Return filename on success
        except Exception as e:
            logger.error(f"Error saving {self.model_name} model for {self.symbol}: {e}")
            return None

    def load_model(self):
        """Load model from disk. Base implementation for pickle."""
        if not self.symbol:
            logger.error(f"Symbol not provided for {self.model_name} model during load")
            return False

        model_filename = self.get_model_filename()
        if not model_filename: return False
        model_path = os.path.join(self.model_dir, model_filename)
        logger.debug(f"Looking for {self.model_name} model at: {os.path.abspath(model_path)}")

        if not os.path.exists(model_path):
            logger.warning(f"No {self.model_name} model file found for {self.symbol} at {model_path}")
            return False

        try:
            logger.info(f"Found {self.model_name} model file for {self.symbol}: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"Successfully loaded {self.model_name} model for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def preprocess_data(self, historical_data):
        """Preprocess data using the scaler (primarily for LSTM)."""
        if 'close' not in historical_data.columns:
            raise ValueError("Data must contain 'close' column")
        close_prices = historical_data['close'].values.reshape(-1, 1)
        # Fit AND transform here. Assumes this is called during training.
        # Load_model should load the fitted scaler for prediction.
        scaled_data = self.scaler.fit_transform(close_prices)
        return scaled_data

    # --- Abstract methods (or provide default implementations) ---
    def train(self, historical_data):
        """Train Prophet model."""
        try:
            # Check if Prophet is available
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet package not installed. Please install with: pip install prophet")
                
            data = historical_data.sort_index().copy()
            if 'close' not in data.columns: raise ValueError("Data must contain 'close' column")

            prophet_data = pd.DataFrame({'ds': data.index, 'y': data['close']})
            # Instantiate Prophet model here
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
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def evaluate(self, test_data):
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")


class ARIMAModel(PricePredictionModel):
    """Price prediction model using ARIMA."""
    def __init__(self, prediction_days=30, order=(5,1,0), symbol=None):
        super().__init__(model_name="arima", prediction_days=prediction_days, symbol=symbol)
        self.order = order
        # Note: ARIMA doesn't use the scaler from the base class

    # save_model and load_model use the base class implementation (pickle)

    def train(self, historical_data):
        """Train ARIMA model."""
        try:
            data = historical_data.sort_index()
            if 'close' not in data.columns: raise ValueError("Data must contain 'close' column")
            close_prices = data['close']

            # Use the renamed import
            self.model = ARIMA_Statsmodels(close_prices, order=self.order)
            self.model = self.model.fit()

            self.is_trained = True
            logger.info(f"ARIMA model trained successfully for {self.symbol} with order {self.order}")
            return True
        except Exception as e:
            logger.error(f"Error training ARIMA model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def predict(self, historical_data, days=None):
        """Make predictions using ARIMA model."""
        if not self.is_trained and not self.load_model():
            logger.error(f"ARIMA model for {self.symbol} not trained/loaded")
            return pd.DataFrame()
        try:
            pred_days = days if days is not None else self.prediction_days
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
            forecast_result = self.model.get_forecast(steps=pred_days)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05) # 95% confidence interval

            predictions = pd.DataFrame({
                'close': forecast_values,
                'lower': conf_int.iloc[:, 0],
                'upper': conf_int.iloc[:, 1]
            }, index=future_dates)

            logger.info(f"ARIMA prediction successful for {self.symbol}")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions with ARIMA model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def evaluate(self, test_data):
        """Evaluate ARIMA model performance."""
        if not self.is_trained and not self.load_model():
            logger.error(f"ARIMA model for {self.symbol} not trained/loaded")
            return {}
        try:
            if test_data.empty: return {'error': 'Test data is empty'}
            start_idx = test_data.index[0]
            end_idx = test_data.index[-1]

            # Ensure predictions cover the test data range
            predictions = self.model.predict(start=start_idx, end=end_idx)

            # Align actual and predicted values
            actual = test_data['close']
            common_index = actual.index.intersection(predictions.index)
            if len(common_index) < 2:
                 return {'error': f'Insufficient overlap between test data and predictions ({len(common_index)} points)'}

            actual_aligned = actual.loc[common_index]
            predictions_aligned = predictions.loc[common_index]

            mae = mean_absolute_error(actual_aligned, predictions_aligned)
            mse = mean_squared_error(actual_aligned, predictions_aligned)
            rmse = np.sqrt(mse)
            non_zero_actual = actual_aligned[actual_aligned != 0]
            mape = np.mean(np.abs((non_zero_actual - predictions_aligned.loc[non_zero_actual.index]) / non_zero_actual)) * 100 if not non_zero_actual.empty else 0.0

            return {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'mape': float(mape)}
        except Exception as e:
            logger.error(f"Error evaluating ARIMA model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}


class ProphetModel(PricePredictionModel):
    """Price prediction model using Facebook Prophet."""
    def __init__(self, prediction_days=30, symbol=None):
        super().__init__(model_name="prophet", prediction_days=prediction_days, symbol=symbol)
        # Prophet doesn't use the base scaler

    # --- Add save_model and load_model using base class pickle implementation ---
    def save_model(self):
        """Save Prophet model using base pickle method."""
        return super().save_model()

    def load_model(self):
        """Load Prophet model using base pickle method."""
        return super().load_model()

    def train(self, historical_data):
        """Train Prophet model."""
        try:
            data = historical_data.sort_index().copy()
            if 'close' not in data.columns: raise ValueError("Data must contain 'close' column")

            prophet_data = pd.DataFrame({'ds': data.index, 'y': data['close']})
            # Instantiate Prophet model here
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
            logger.error(f"Prophet model for {self.symbol} not trained/loaded")
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
                'close': future_predictions['yhat'],
                'upper': future_predictions['yhat_upper'],
                'lower': future_predictions['yhat_lower']
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
            logger.error(f"Prophet model for {self.symbol} not trained/loaded")
            return {}
        try:
            return calculate_prophet_metrics(self.model, test_data) # Use the standalone helper
        except Exception as e:
            logger.error(f"Error evaluating Prophet model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}


class LSTMModel(PricePredictionModel):
    """Price prediction model using LSTM neural networks."""
    def __init__(self, prediction_days=30, lookback=60, units=50, epochs=50, batch_size=32, symbol=None):
        super().__init__(model_name="lstm", prediction_days=prediction_days, symbol=symbol)
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        # Scaler is inherited from base class

    def get_model_filename(self):
        """Override to use .h5 extension for Keras models."""
        if not self.symbol: return None
        return f"{self.model_name}_{self.symbol}.h5" # Use .h5

    def _create_sequences(self, data):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # preprocess_data uses the base class implementation

    def train(self, historical_data):
        """Train LSTM model."""
        try:
            if not self.symbol:
                 logger.error("Cannot train LSTM model without a symbol.")
                 return False
            logger.info(f"Starting LSTM training for {self.symbol}...")

            # Preprocess data (fits the scaler)
            scaled_data = self.preprocess_data(historical_data)

            X, y = self._create_sequences(scaled_data)
            if X.size == 0 or y.size == 0:
                raise ValueError(f"Not enough data after sequencing for lookback {self.lookback}")

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            self.model = Sequential([
                LSTM(units=self.units, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=self.units),
                Dropout(0.2),
                Dense(units=1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')

            logger.info(f"Fitting LSTM model for {self.symbol}...")
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            self.is_trained = True
            logger.info(f"LSTM model trained successfully for {self.symbol}")
            return True # Return success status
        except ImportError:
             logger.error("TensorFlow/Keras not installed. Cannot train LSTM model.")
             return False
        except Exception as e:
            logger.error(f"Error training LSTM model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def predict(self, historical_data, days=None):
        """Make predictions using LSTM model."""
        if not self.is_trained and not self.load_model():
            logger.error(f"LSTM model for {self.symbol} not trained/loaded")
            return pd.DataFrame()
        try:
            pred_days = days if days is not None else self.prediction_days

            # --- Use the SCALER LOADED by load_model ---
            full_history_close = historical_data['close'].values.reshape(-1, 1)
            # --- Use transform ONLY, assuming scaler is fitted ---
            scaled_full_history = self.scaler.transform(full_history_close)

            last_sequence = scaled_full_history[-self.lookback:].reshape(1, self.lookback, 1)
            predictions_scaled = []
            current_sequence = last_sequence.copy()

            for _ in range(pred_days):
                next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions_scaled.append(next_pred_scaled)
                current_sequence = np.append(current_sequence[:, 1:, :], [[[next_pred_scaled]]], axis=1)

            # Inverse transform using the loaded scaler
            predictions = self.scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

            last_date = historical_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_days)
            result = pd.DataFrame(predictions, index=future_dates, columns=['close'])
            # Add simple confidence bounds for LSTM
            std_dev_factor = 0.05 # Example: 5% std dev factor
            result['upper'] = result['close'] * (1 + std_dev_factor)
            result['lower'] = result['close'] * (1 - std_dev_factor)
            logger.info(f"LSTM prediction successful for {self.symbol}")
            return result
        except ImportError:
             logger.error("TensorFlow/Keras not installed. Cannot predict with LSTM model.")
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error making predictions with LSTM model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def evaluate(self, test_data):
        """Evaluate LSTM model performance."""
        if not self.is_trained and not self.load_model():
            logger.error(f"LSTM model for {self.symbol} not trained/loaded")
            return {}
        try:
            if test_data.empty: return {'error': 'Test data is empty'}

            # --- Use transform ONLY with the loaded scaler ---
            scaled_data = self.scaler.transform(test_data['close'].values.reshape(-1, 1))

            X_test, y_test_scaled = self._create_sequences(scaled_data)
            if X_test.size == 0 or y_test_scaled.size == 0:
                 return {'error': f'Insufficient test data after sequencing for lookback {self.lookback}'}

            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predictions_scaled = self.model.predict(X_test, verbose=0)

            # Inverse transform both predictions and actual values
            predictions = self.scaler.inverse_transform(predictions_scaled)
            y_test = self.scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            non_zero_y = y_test[y_test != 0]
            mape = np.mean(np.abs((non_zero_y - predictions[y_test != 0]) / non_zero_y)) * 100 if non_zero_y.size > 0 else 0.0

            return {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'mape': float(mape)}
        except ImportError:
             logger.error("TensorFlow/Keras not installed.")
             return {'error': 'TensorFlow/Keras not installed'}
        except Exception as e:
            logger.error(f"Error evaluating LSTM model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def save_model(self):
        """Save LSTM model (.h5) and scaler (.pkl)."""
        if not self.is_trained:
            logger.warning(f"LSTM model for {self.symbol} not trained, cannot save")
            return None
        if not self.symbol:
            logger.error(f"Symbol not provided for LSTM model during save")
            return None

        model_filename = self.get_model_filename() # Gets .h5 filename
        scaler_filename = self.get_scaler_filename() # Gets .pkl filename
        if not model_filename or not scaler_filename: return None

        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)

        try:
            logger.info(f"Saving LSTM model to {model_path}")
            self.model.save(model_path) # Save Keras model

            logger.info(f"Saving LSTM scaler to {scaler_path}")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f) # Save the scaler

            logger.info(f"LSTM model and scaler saved successfully for {self.symbol}")
            return model_filename # Return the primary model filename
        except ImportError:
             logger.error("TensorFlow/Keras not installed. Cannot save LSTM model.")
             return None
        except Exception as e:
            logger.error(f"Error saving LSTM model for {self.symbol}: {e}")
            return None

    def load_model(self):
        """Load LSTM model (.h5) and scaler (.pkl)."""
        if not self.symbol:
            logger.error(f"Symbol not provided for LSTM model during load")
            return False

        model_filename = self.get_model_filename() # Gets .h5 filename
        scaler_filename = self.get_scaler_filename() # Gets .pkl filename
        if not model_filename or not scaler_filename: return False

        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)

        logger.debug(f"Looking for LSTM model at: {os.path.abspath(model_path)}")
        logger.debug(f"Looking for LSTM scaler at: {os.path.abspath(scaler_path)}")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning(f"LSTM model ({model_filename}) or scaler ({scaler_filename}) file not found for {self.symbol}")
            return False

        try:
            logger.info(f"Loading LSTM model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)

            logger.info(f"Loading LSTM scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f) # Load the scaler

            self.is_trained = True
            logger.info(f"LSTM model and scaler loaded successfully for {self.symbol}")
            return True
        except ImportError:
             logger.error("TensorFlow/Keras not installed. Cannot load LSTM model.")
             return False
        except Exception as e:
            logger.error(f"Error loading LSTM model for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return False


class EnsembleModel(PricePredictionModel):
    """Ensemble model combining predictions."""
    def __init__(self, prediction_days=30, models_to_include=None, weights=None, symbol=None):
        super().__init__(model_name="ensemble", prediction_days=prediction_days, symbol=symbol)

        if models_to_include is None:
            models_to_include = ['arima', 'prophet', 'lstm'] # Default models

        self.models = []
        for model_type in models_to_include:
            if model_type == 'arima':
                self.models.append(ARIMAModel(prediction_days=prediction_days, symbol=symbol))
            elif model_type == 'prophet':
                self.models.append(ProphetModel(prediction_days=prediction_days, symbol=symbol))
            elif model_type == 'lstm':
                self.models.append(LSTMModel(prediction_days=prediction_days, symbol=symbol))
            else:
                logger.warning(f"Unknown model type '{model_type}' requested for ensemble.")

        if not self.models:
            raise ValueError("EnsembleModel requires at least one valid model type.")

        if weights is None or len(weights) != len(self.models):
            self.weights = [1/len(self.models)] * len(self.models) # Equal weights
        else:
            self.weights = [w / sum(weights) for w in weights] # Normalize

    def train(self, historical_data):
        """Train all models in the ensemble."""
        success_count = 0
        for model in self.models:
            logger.info(f"Training {model.model_name} for ensemble ({self.symbol})...")
            if model.train(historical_data):
                success_count += 1
            else:
                logger.warning(f"Failed to train {model.model_name} for ensemble ({self.symbol})")
        self.is_trained = success_count > 0 # Consider trained if at least one sub-model trained
        return self.is_trained

    def predict(self, historical_data, days=None):
        """Make predictions using weighted combination."""
        pred_days = days if days is not None else self.prediction_days
        all_predictions = []
        valid_model_indices = []

        for i, model in enumerate(self.models):
            # Attempt to load if not trained, then predict
            if model.is_trained or model.load_model():
                try:
                    pred = model.predict(historical_data, days=pred_days)
                    if not pred.empty:
                        all_predictions.append(pred)
                        valid_model_indices.append(i)
                        logger.info(f"Ensemble: Got predictions from {model.model_name} ({self.symbol})")
                    else:
                        logger.warning(f"Ensemble: No predictions from {model.model_name} ({self.symbol})")
                except Exception as e:
                    logger.error(f"Ensemble: Error getting predictions from {model.model_name} ({self.symbol}): {e}")
            else:
                logger.warning(f"Ensemble: {model.model_name} model not trained/loaded ({self.symbol})")

        if not all_predictions:
            logger.error(f"Ensemble: No predictions available from any sub-model for {self.symbol}")
            return pd.DataFrame()

        # Align predictions (find common index, usually future dates)
        common_index = all_predictions[0].index
        for pred in all_predictions[1:]:
            common_index = common_index.intersection(pred.index)

        if common_index.empty:
            logger.error(f"Ensemble: No common prediction dates found for {self.symbol}")
            return pd.DataFrame()

        # Get weighted average
        weighted_sum = pd.Series(0.0, index=common_index) # Initialize with float
        total_weight = 0.0

        for i, pred_idx in enumerate(valid_model_indices):
            weight = self.weights[pred_idx]
            pred_df = all_predictions[i]
            # Reindex prediction to common index and fill missing values if any (shouldn't happen ideally)
            reindexed_pred = pred_df.reindex(common_index).ffill().bfill()
            weighted_sum += reindexed_pred['close'] * weight
            total_weight += weight

        # Normalize the weighted sum if total_weight is not 1 (due to model failures)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            weighted_sum /= total_weight

        # Create final prediction DataFrame (add simple confidence bounds)
        result = pd.DataFrame({'close': weighted_sum})
        std_dev_factor = 0.07 # Slightly wider bounds for ensemble
        result['upper'] = result['close'] * (1 + std_dev_factor)
        result['lower'] = result['close'] * (1 - std_dev_factor)

        return result

    def evaluate(self, test_data):
        """Evaluate ensemble model performance."""
        # Evaluate each sub-model
        model_metrics = {}
        for model in self.models:
            try:
                 metrics = model.evaluate(test_data)
                 if metrics and 'error' not in metrics:
                     model_metrics[model.model_name] = metrics
                 elif metrics: # Keep error message if evaluation failed
                      model_metrics[model.model_name] = metrics
            except Exception as eval_err:
                 logger.error(f"Error evaluating sub-model {model.model_name}: {eval_err}")
                 model_metrics[model.model_name] = {'error': str(eval_err)}


        # Evaluate the ensemble prediction itself
        try:
            # Predict for the test period. Need historical data *before* test_data.
            # This is tricky. For simplicity, we'll predict based on data up to the start of test_data.
            # A more robust evaluation would use rolling forecasts.
            if test_data.empty: raise ValueError("Test data is empty")
            start_test_date = test_data.index[0]
            # Find historical data ending just before the test period starts
            # Assuming historical data used for prediction includes up to start_test_date - 1 day
            # This part needs careful handling in a real backtesting scenario.
            # Let's assume `historical_data` passed to `predict` ends before `test_data` starts.
            # For this function, we'll just predict the length of the test data.
            num_test_days = len(test_data)
            # We need the data *before* test_data to make the prediction
            # This requires access to the full dataset, which isn't directly available here.
            # --- Simplified Evaluation: Predict based on test_data itself (introduces lookahead bias) ---
            # THIS IS NOT IDEAL FOR REAL EVALUATION but allows calculation
            logger.warning("Evaluating ensemble using test data itself for prediction base - introduces lookahead bias.")
            ensemble_pred = self.predict(test_data, days=num_test_days) # Predict length of test data

            if not ensemble_pred.empty:
                # Align predictions with actual test data
                common_index = ensemble_pred.index.intersection(test_data.index)
                if len(common_index) > 1:
                    ensemble_pred_aligned = ensemble_pred.loc[common_index]
                    actual_aligned = test_data.loc[common_index, 'close']

                    mae = mean_absolute_error(actual_aligned, ensemble_pred_aligned['close'])
                    mse = mean_squared_error(actual_aligned, ensemble_pred_aligned['close'])
                    rmse = np.sqrt(mse)
                    non_zero_actual = actual_aligned[actual_aligned != 0]
                    mape = np.mean(np.abs((non_zero_actual - ensemble_pred_aligned['close'].loc[non_zero_actual.index]) / non_zero_actual)) * 100 if not non_zero_actual.empty else 0.0

                    model_metrics['ensemble'] = {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'mape': float(mape)}
                else:
                     model_metrics['ensemble'] = {'error': 'Insufficient overlap for ensemble evaluation'}
            else:
                 model_metrics['ensemble'] = {'error': 'Ensemble prediction failed during evaluation'}

        except Exception as e:
            logger.error(f"Error evaluating ensemble model for {self.symbol}: {e}")
            model_metrics['ensemble'] = {'error': str(e)}

        return model_metrics

    # Ensemble doesn't save a single model file, relies on sub-models
    def save_model(self):
        logger.warning("EnsembleModel does not save a single file; relies on saving sub-models.")
        return None # No single file to return

    def load_model(self):
        logger.warning("EnsembleModel does not load a single file; relies on loading sub-models.")
        # Check if sub-models can be loaded
        loaded_count = 0
        for model in self.models:
            if model.load_model():
                loaded_count += 1
        self.is_trained = loaded_count > 0 # Consider 'trained' if any sub-model loads
        return self.is_trained


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
def train_price_prediction_models(symbol, model_type_to_train='prophet', lookback_period="2y"):
    """
    Train a specified price prediction model (Prophet, ARIMA, LSTM) for a symbol.
    Saves the model file(s) and returns necessary info for metadata saving.

    Args:
        symbol (str): Stock symbol to train models for.
        model_type_to_train (str): The type of model to train ('prophet', 'arima', 'lstm').
        lookback_period (str): Historical period for training data (e.g., "1y", "2y").

    Returns:
        tuple: (model_object, metrics_dict, model_type_str, model_filename_str)
               Returns (None, {'error': msg}, model_type_str, None) on failure.
               model_object might be None even on success if saved internally.
    """
    if not symbol:
        error_msg = "Symbol parameter cannot be None or empty"
        logger.error(error_msg)
        return None, {"error": error_msg}, model_type_to_train, None

    logger.info(f"Starting model training for symbol: {symbol}, Model Type: {model_type_to_train}")

    try:
        # Get historical data
        historical_data = data_provider.get_historical_price(symbol, period=lookback_period)
        if historical_data.empty:
            raise ValueError(f"No historical data available for {symbol} via DataProvider")
        if 'close' not in historical_data.columns:
            raise ValueError("Missing required column: 'close'")
        if len(historical_data) < 60:
            raise ValueError(f"Not enough historical data for {symbol} (got {len(historical_data)} rows, need at least 60)")

        # Split data (80/20)
        split_idx = int(len(historical_data) * 0.8)
        train_data = historical_data.iloc[:split_idx]
        test_data = historical_data.iloc[split_idx:]
        logger.info(f"Training data: {len(train_data)} days, Testing data: {len(test_data)} days")

        # --- Instantiate the selected model ---
        model_trainer = None
        if model_type_to_train == 'prophet':
            if not check_prophet_installed(): raise ImportError("Prophet package required")
            model_trainer = ProphetModel(symbol=symbol)
        elif model_type_to_train == 'arima':
            model_trainer = ARIMAModel(symbol=symbol)
        elif model_type_to_train == 'lstm':
            # Check for TensorFlow/Keras installation early
            try: import tensorflow as tf
            except ImportError: raise ImportError("TensorFlow/Keras required for LSTM")
            model_trainer = LSTMModel(symbol=symbol) # Use default LSTM params
        else:
            raise ValueError(f"Unsupported model type: {model_type_to_train}")

        # --- Train the model ---
        logger.info(f"Training {model_type_to_train} model for {symbol}...")
        train_success = model_trainer.train(train_data)
        if not train_success:
            raise RuntimeError(f"{model_type_to_train} model training failed.")

        # --- Evaluate the model ---
        logger.info(f"Evaluating {model_type_to_train} model for {symbol}...")
        metrics = model_trainer.evaluate(test_data)
        if not metrics or 'error' in metrics:
             logger.warning(f"Metrics calculation failed or returned error for {symbol}. Metrics: {metrics}")
             if isinstance(metrics, dict) and 'error' in metrics: pass # Keep error metrics
             else: metrics = {'error': 'Metrics calculation failed'}

        # --- Save the model (model class handles format) ---
        logger.info(f"Saving {model_type_to_train} model for {symbol}...")
        model_filename = model_trainer.save_model() # This should return the filename
        if not model_filename:
             # Even if saving fails, log it but maybe still return metrics?
             logger.error(f"Failed to save {model_type_to_train} model for {symbol}.")
             # Decide if this is critical. Let's return metrics but no filename.
             return model_trainer.model, metrics, model_type_to_train, None

        logger.info(f"Successfully trained and saved {model_type_to_train} model for {symbol} as {model_filename}")

        # --- Return the necessary info ---
        # Return model object (might be None), metrics, type, and filename
        return model_trainer.model, metrics, model_type_to_train, model_filename

    except Exception as e:
        logger.error(f"Error in train_price_prediction_models for {symbol} ({model_type_to_train}): {e}")
        logger.error(traceback.format_exc())
        return None, {"error": str(e)}, model_type_to_train, None

def get_price_predictions(symbol, days=30, try_ensemble_fallback=True):
    """
    Get price predictions, trying Prophet first, then optionally Ensemble as fallback.
    """
    logger.info(f"Getting price predictions for {symbol} for {days} days. Ensemble Fallback: {try_ensemble_fallback}")
    if not symbol: return create_simple_fallback("UNKNOWN", days)

    prophet_predictions_df = pd.DataFrame()
    ensemble_predictions_df = pd.DataFrame()
    used_model_type = 'unknown'

    try:
        historical_data = data_provider.get_historical_price(symbol, period="1y")
        if historical_data.empty:
            logger.error(f"No historical data for {symbol} in get_price_predictions.")
            return create_fallback_prediction(symbol, days)

        # --- 1. Try Prophet Model ---
        try:
            prophet_predictor = ProphetModel(symbol=symbol, prediction_days=days)
            if prophet_predictor.load_model():
                logger.info(f"Loaded existing Prophet model for {symbol}.")
                prophet_predictions_df = prophet_predictor.predict(historical_data, days=days)
                if not prophet_predictions_df.empty:
                    used_model_type = 'prophet'
                    logger.info(f"Prophet prediction successful for {symbol}.")
                else:
                    logger.warning(f"Prophet prediction returned empty DataFrame for {symbol}.")
            else:
                logger.warning(f"No pre-trained Prophet model found for {symbol}.")
        except Exception as prophet_err:
            logger.error(f"Error during Prophet prediction for {symbol}: {prophet_err}")

        # --- 2. Try Ensemble Model (if Prophet failed and fallback enabled) ---
        if prophet_predictions_df.empty and try_ensemble_fallback:
            logger.warning(f"Prophet failed for {symbol}. Attempting Ensemble fallback.")
            try:
                ensemble_predictor = EnsembleModel(symbol=symbol, prediction_days=days)
                # predict handles loading sub-models
                ensemble_predictions_df = ensemble_predictor.predict(historical_data, days=days)
                if not ensemble_predictions_df.empty:
                    used_model_type = 'ensemble'
                    logger.info(f"Ensemble prediction successful for {symbol}.")
                else:
                    logger.warning(f"Ensemble prediction also returned empty DataFrame for {symbol}.")
            except Exception as ensemble_err:
                 logger.error(f"Error during Ensemble prediction for {symbol}: {ensemble_err}")

        # --- 3. Select Prediction DataFrame ---
        predictions_df = pd.DataFrame()
        if not prophet_predictions_df.empty:
            predictions_df = prophet_predictions_df
        elif not ensemble_predictions_df.empty:
             predictions_df = ensemble_predictions_df

        # --- 4. Process predictions or use fallback ---
        if not predictions_df.empty:
            result = {
                'symbol': symbol,
                'model': used_model_type,
                'dates': predictions_df.index.strftime('%Y-%m-%d').tolist(),
                'values': predictions_df['close'].tolist()
            }
            if 'upper' in predictions_df.columns and 'lower' in predictions_df.columns:
                result['confidence'] = {'upper': predictions_df['upper'].tolist(), 'lower': predictions_df['lower'].tolist()}
            else: # Add simple confidence if missing
                result['confidence'] = {'upper': [v*1.05 for v in result['values']], 'lower': [v*0.95 for v in result['values']]}

            cleaned_result = handle_nan_values(result)
            return cleaned_result if cleaned_result else create_simple_fallback(symbol, days)
        else:
            logger.warning(f"All prediction models failed for {symbol}. Using standard fallback.")
            return create_fallback_prediction(symbol, days)

    except Exception as e:
        logger.error(f"Critical Error in get_price_predictions for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return create_simple_fallback(symbol, days)


def check_prophet_installed():
    """Check if Prophet is installed."""
    global PROPHET_AVAILABLE
    if PROPHET_AVAILABLE:
        return True
    else:
        try:
            # Try one more time to import
            from prophet import Prophet
            PROPHET_AVAILABLE = True
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
