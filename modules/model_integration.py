# modules/model_integration.py
"""
Integration module for ML models in the Investment Recommendation System.
Combines price prediction, trend analysis, and portfolio optimization.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
import pickle # Added

# Import custom modules
# --- Ensure train_price_prediction_models is imported ---
from modules.price_prediction import get_price_predictions, train_price_prediction_models
from modules.trend_analysis import TrendAnalyzer
# --- DataProvider is already imported in __init__ ---
# from modules.data_provider import DataProvider
from modules.portfolio_utils import load_portfolio
from modules.price_prediction import ProphetModel
# --- Import save_model_metadata ---
from modules.db_utils import save_model_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelIntegration:
    """
    Integrates various ML models for investment recommendations.
    """
    def __init__(self):
        # --- Use DataProvider instance directly ---
        from modules.data_provider import data_provider
        self.data_provider = data_provider
        # --- End DataProvider change ---
        self.trend_analyzer = TrendAnalyzer()
        self.prediction_cache = {}
        self.trend_cache = {}
        self.model_training_status = {}

        self.model_dir = "models"
        self.data_dir = "data/model_data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.training_executor = ThreadPoolExecutor(max_workers=1) # Limit workers if needed

    # --- Keep get_asset_analysis (ensure it uses self.data_provider) ---
    def get_asset_analysis(self, symbol, days_to_predict=30, force_refresh=False):
        """
        Get comprehensive analysis for a specific asset, including price predictions
        and trend analysis. Ensures default values for failed sub-analyses.
        """
        cache_key = f"{symbol}_analysis_{days_to_predict}"
        # --- Cache Check (Optional: Keep disabled for debugging if needed) ---
        # force_refresh = True # Uncomment to force refresh
        if cache_key in self.prediction_cache and not force_refresh:
            cache_time, data = self.prediction_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 3600: # 1 hour cache
                logger.info(f"Returning cached analysis for {symbol}")
                return data
        # --- End Cache Check ---

        try:
            # --- Use self.data_provider ---
            historical_data_df = self.data_provider.get_historical_price(symbol=symbol, period="1y")

            if historical_data_df.empty:
                logger.error(f"No historical data available for {symbol} in get_asset_analysis")
                return None # Cannot proceed without data

            df = historical_data_df # Use the DataFrame directly

            # --- Run analyses with defaults ---
            trend_analysis = self.trend_analyzer.detect_trend(df) or {'overall_trend': 'unknown', 'trend_strength': 50, 'details': {}, 'latest_data': {}}
            support_resistance = self.trend_analyzer.identify_support_resistance(df) or {'support': [], 'resistance': [], 'current_price': df['close'].iloc[-1] if not df.empty else None}
            patterns = self.trend_analyzer.detect_patterns(df) or {'patterns': []}
            breakout = self.trend_analyzer.predict_breakout(df, support_resistance) or {'prediction': 'neutral', 'confidence': 0, 'details': 'Analysis failed'}
            market_regime = self.trend_analyzer.get_market_regime(df) or {'regime': 'unknown', 'trend': 'unknown'}

            # --- Get price predictions (already has fallbacks) ---
            predictions = get_price_predictions(symbol, days=days_to_predict)
            if predictions is None:
                 logger.warning(f"Price prediction returned None for {symbol}. Using simple fallback.")
                 from modules.price_prediction import create_simple_fallback
                 predictions = create_simple_fallback(symbol, days_to_predict)

            training_status = self.model_training_status.get(symbol, {'status': 'not_started'})

            # --- Combine analyses ---
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
                'model_training': training_status
            }

            # Cache the results
            self.prediction_cache[cache_key] = (datetime.now(), analysis)
            return analysis

        except Exception as e:
            logger.error(f"Critical Error in get_asset_analysis for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    # --- Refactored _do_training ---
    def _do_training(self, symbol, model_type, lookback_period): # Added model_type
        """
        Internal method to perform model training for a specific symbol and type.
        Calls training function and saves metadata to DB.

        Args:
            symbol (str): Asset symbol
            model_type (str): Type of model to train ('prophet', 'arima', 'lstm')
            lookback_period (str): Lookback period for training data

        Returns:
            dict: Training status and metrics
        """
        logger.info(f"Starting _do_training for {symbol} (Type: {model_type})...")
        status_key = f"{symbol}_{model_type}" # Use combined key for status tracking
        try:
            # Update training status
            self.model_training_status[status_key] = { # Use combined key
                'status': 'in_progress',
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"Status set to in_progress for {status_key}")

            # --- Call the modified training function, passing model_type ---
            model_object, performance_metrics, trained_model_type, model_filename = train_price_prediction_models(
                symbol=symbol,
                model_type_to_train=model_type, # Pass the requested type
                lookback_period=lookback_period
            )

            # Check if training and saving were successful
            if model_filename is None or (isinstance(performance_metrics, dict) and 'error' in performance_metrics):
                error_message = performance_metrics.get('error', 'Unknown training error') if isinstance(performance_metrics, dict) else "Unknown training error"
                logger.error(f"Model training/saving failed for {status_key}: {error_message}")
                self.model_training_status[status_key] = { # Use combined key
                    'status': 'failed',
                    'error': error_message,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return {"status": "failed", "error": error_message}

            logger.info(f"Model training/saving successful for {status_key}. Filename: {model_filename}")

            # --- Save Metadata to Database ---
            logger.info(f"Attempting to save metadata for {model_filename}...")
            notes = f"Training run ({model_type}) completed on {datetime.now().strftime('%Y-%m-%d')}. Lookback: {lookback_period}."
            metrics_to_save = performance_metrics if isinstance(performance_metrics, dict) else {}
            logger.debug(f"Metadata Params: filename={model_filename}, symbol={symbol}, type={trained_model_type}, metrics={metrics_to_save}, notes={notes}")

            metadata_saved = save_model_metadata(
                filename=model_filename,
                symbol=symbol,
                model_type=trained_model_type, # Use the type returned by the training function
                metrics=metrics_to_save,
                notes=notes
            )
            logger.info(f"Result of save_model_metadata for {model_filename}: {metadata_saved}")

            if metadata_saved:
                logger.info(f"Metadata saved. Training completed successfully for {status_key}")
                self.model_training_status[status_key] = { # Use combined key
                    'status': 'completed',
                    'metrics': metrics_to_save,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return {"status": "completed", "metrics": metrics_to_save}
            else:
                error_message = f"Failed to save metadata to DB for {model_filename}"
                logger.error(error_message)
                self.model_training_status[status_key] = { # Use combined key
                    'status': 'failed',
                    'error': error_message,
                    'metrics': metrics_to_save,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return {"status": "failed", "error": error_message, "metrics": metrics_to_save}

        except Exception as e:
            logger.error(f"Exception during _do_training for {status_key}: {e}")
            logger.error(traceback.format_exc())
            error_message = f"Error in model training process: {str(e)}"
            self.model_training_status[status_key] = { # Use combined key
                'status': 'failed',
                'error': error_message,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return {"status": "failed", "error": error_message}
    
    def train_models_for_symbol(self, symbol, model_type='prophet', lookback_period="2y", async_training=True): # Added model_type
        """Train ML models for a specific symbol and type."""
        logger.info(f"Received request to train model for {symbol}, Type: {model_type}. Async: {async_training}")
        status_key = f"{symbol}_{model_type}" # Use combined key
        self.model_training_status[status_key] = { # Use combined key
            'status': 'pending',
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.info(f"Status set to pending for {status_key}")

        def training_thread_wrapper():
            try:
                logger.info(f"Starting training thread for {status_key}")
                # Pass model_type to _do_training
                self._do_training(symbol, model_type, lookback_period)
                logger.info(f"Training thread finished for {status_key}")
            except Exception as e:
                logger.error(f"Exception in training thread wrapper for {status_key}: {str(e)}")
                logger.error(traceback.format_exc())
                self.model_training_status[status_key] = { # Use combined key
                    'status': 'failed',
                    'error': f"Thread execution error: {str(e)}",
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        if async_training:
            self.training_executor.submit(training_thread_wrapper)
            logger.info(f"Training submitted asynchronously for {status_key}")
            return "pending"
        else:
            logger.info(f"Training running synchronously for {status_key}")
            # Pass model_type to _do_training
            result = self._do_training(symbol, model_type, lookback_period)
            logger.info(f"Synchronous training finished for {status_key} with status: {result.get('status')}")
            return result.get("status", "failed")
        
            
    def get_portfolio_recommendations(self, use_ml=True):
        """Generate portfolio recommendations."""
        # ... (Keep existing logic, ensure load_portfolio etc. are used correctly) ...
        # ... (Ensure _generate_recommendation uses logger) ...
        try:
            portfolio = load_portfolio()
            portfolio_symbols = {details.get("symbol", "") for _, details in portfolio.items() if details.get("symbol")}
            from modules.portfolio_utils import load_user_profile, load_tracked_assets
            user_profile = load_user_profile()
            risk_level = user_profile.get("risk_level", 5)
            tracked_assets = load_tracked_assets()
            all_symbols = portfolio_symbols.union(tracked_assets.keys())

            asset_analyses = {}
            recommendations = {'buy': [], 'hold': [], 'sell': [], 'portfolio_score': 0}

            for symbol in all_symbols:
                if not symbol: continue
                analysis = self.get_asset_analysis(symbol)
                if analysis:
                    asset_analyses[symbol] = analysis
                    recommendation = self._generate_recommendation(symbol, analysis, portfolio, risk_level, use_ml)
                    if recommendation:
                        action = recommendation.get('action', 'hold')
                        if action in recommendations: recommendations[action].append(recommendation)

            recommendations['buy'].sort(key=lambda x: x.get('confidence', 0), reverse=True)
            recommendations['sell'].sort(key=lambda x: x.get('confidence', 0), reverse=True)

            # Calculate portfolio score (simplified example)
            if portfolio:
                total_value = sum(float(inv.get("current_value", 0)) for inv in portfolio.values())
                weighted_score = 0
                if total_value > 0:
                    for _, details in portfolio.items():
                        symbol = details.get("symbol", "")
                        if symbol in asset_analyses:
                             weight = float(details.get("current_value", 0)) / total_value
                             # Simplified scoring based on trend/breakout
                             trend_strength = asset_analyses[symbol]['trend'].get('trend_strength', 50)
                             breakout_conf = asset_analyses[symbol]['breakout'].get('confidence', 50)
                             breakout_dir = 1 if asset_analyses[symbol]['breakout'].get('prediction') == 'bullish' else -1 if asset_analyses[symbol]['breakout'].get('prediction') == 'bearish' else 0
                             asset_score = (trend_strength * 0.6 + (50 + breakout_dir * breakout_conf * 0.5) * 0.4)
                             weighted_score += asset_score * weight
                recommendations['portfolio_score'] = weighted_score

            return recommendations
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            logger.error(traceback.format_exc())
            return {'buy': [], 'hold': [], 'sell': [], 'portfolio_score': 0, 'error': str(e)}

    
    def _generate_recommendation(self, symbol, analysis, portfolio, risk_level, use_ml=True):
        """Generate recommendation for a specific asset."""
        # ... (Keep existing logic, ensure logging is used) ...
        try:
            in_portfolio = any(details.get("symbol") == symbol for _, details in portfolio.items())
            portfolio_details = next((details for _, details in portfolio.items() if details.get("symbol") == symbol), None)

            current_price = analysis['price']['current']
            trend = analysis['trend']
            overall_trend = trend.get('overall_trend', 'neutral')
            trend_strength = trend.get('trend_strength', 50)
            support_resistance = analysis['support_resistance']
            current_support = max([s['price'] for s in support_resistance.get('support', [])], default=None)
            current_resistance = min([r['price'] for r in support_resistance.get('resistance', [])], default=None)
            breakout = analysis['breakout']
            breakout_prediction = breakout.get('prediction', 'neutral')
            breakout_confidence = breakout.get('confidence', 0)
            market_regime = analysis['market_regime'].get('regime', 'unknown')
            patterns = analysis['patterns']
            bullish_patterns = [p for p in patterns.get('patterns', []) if p.get('type') == 'bullish']
            bearish_patterns = [p for p in patterns.get('patterns', []) if p.get('type') == 'bearish']

            prediction_direction = 'neutral'
            prediction_confidence = 0
            expected_return = 0
            if use_ml and analysis.get('price_predictions'):
                predictions = analysis['price_predictions']
                predicted_values = predictions.get('values', [])
                if predicted_values and current_price > 0:
                    final_prediction = predicted_values[-1]
                    expected_return = (final_prediction / current_price - 1) * 100
                    if expected_return > 5: prediction_direction, prediction_confidence = 'bullish', min(100, expected_return * 10)
                    elif expected_return < -5: prediction_direction, prediction_confidence = 'bearish', min(100, abs(expected_return) * 10)
                    else: prediction_direction, prediction_confidence = 'neutral', 100 - (abs(expected_return) * 10)

            # --- Scoring Logic (Simplified Example) ---
            buy_score, sell_score = 0, 0
            if overall_trend in ['bullish', 'strong_bullish']: buy_score += trend_strength * 0.3
            elif overall_trend in ['bearish', 'strong_bearish']: sell_score += trend_strength * 0.3
            if current_support and current_resistance and current_resistance > current_support:
                position = (current_price - current_support) / (current_resistance - current_support)
                if position < 0.2: buy_score += 80 * 0.15 # Near support
                elif position > 0.8: sell_score += 80 * 0.15 # Near resistance
            if breakout_prediction == 'bullish': buy_score += breakout_confidence * 0.2
            elif breakout_prediction == 'bearish': sell_score += breakout_confidence * 0.2
            if bullish_patterns: buy_score += max(p.get('strength', 50) for p in bullish_patterns) * 0.1
            if bearish_patterns: sell_score += max(p.get('strength', 50) for p in bearish_patterns) * 0.1
            if prediction_direction == 'bullish': buy_score += prediction_confidence * 0.25
            elif prediction_direction == 'bearish': sell_score += prediction_confidence * 0.25
            # --- End Scoring ---

            # Risk Adjustment
            risk_adjustment = (risk_level - 5) / 5
            buy_score *= (1 + risk_adjustment * 0.2)
            sell_score *= (1 - risk_adjustment * 0.2)
            buy_score = max(0, min(100, buy_score))
            sell_score = max(0, min(100, sell_score))

            # Determine Action
            action, confidence, trade_amount = 'hold', 0, 0
            if buy_score > 65 and buy_score > sell_score + 10: # Buy Threshold
                action, confidence = 'buy', buy_score
                trade_amount = 1000 * (confidence / 100) # Example amount
            elif sell_score > 65 and sell_score > buy_score + 10 and in_portfolio: # Sell Threshold
                action, confidence = 'sell', sell_score
                if portfolio_details: trade_amount = float(portfolio_details.get("current_value", 0)) * (confidence / 100) # Sell percentage
            else: # Hold
                confidence = 100 - max(buy_score, sell_score)

            return {
                'symbol': symbol, 'action': action, 'confidence': confidence, 'price': current_price,
                'trade_amount': trade_amount, 'expected_return': expected_return,
                'buy_signals': [], 'sell_signals': [], # Populate these if needed for UI
                'in_portfolio': in_portfolio, 'market_regime': market_regime, 'overall_trend': overall_trend
            }
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    
    def get_model_training_status(self, symbol=None, model_type=None): # Added model_type
        """Get status of model training, optionally filtered by symbol and type."""
        if symbol and model_type:
            status_key = f"{symbol}_{model_type}"
            status_info = self.model_training_status.get(status_key, {'status': 'not_started'})
            # ... (rest of the single status validation logic remains the same) ...
            if not isinstance(status_info, dict): status_info = {'status': str(status_info)}
            if 'last_updated' not in status_info: status_info['last_updated'] = "never"
            return {status_key: status_info} # Return dict with the specific key
        elif symbol: # Filter by symbol only
             filtered_status = {}
             for key, value in self.model_training_status.items():
                 if key.startswith(f"{symbol}_"):
                     # ... (validation logic) ...
                     if isinstance(value, dict):
                         if 'last_updated' not in value: value['last_updated'] = "never"
                         filtered_status[key] = value
                     else:
                         filtered_status[key] = {'status': str(value), 'last_updated': "never"}
             return filtered_status
        else: # Return all statuses
            filtered_status = {}
            for key, value in self.model_training_status.items():
                 # Exclude internal keys if any
                 if key == "batch_training": continue
                 # ... (validation logic) ...
                 if isinstance(value, dict):
                     if 'last_updated' not in value: value['last_updated'] = "never"
                     filtered_status[key] = value
                 else:
                     filtered_status[key] = {'status': str(value), 'last_updated': "never"}
            return filtered_status
        
    def get_backtesting_recommendation_signal(self, symbol, historical_data_slice):
        """Generates a simple Buy/Sell/Hold signal for backtesting."""
        # ... (Keep existing logic, ensure logging is used) ...
        try:
            if not hasattr(self, 'trend_analyzer'): self.trend_analyzer = TrendAnalyzer()
            trend_analysis = self.trend_analyzer.detect_trend(historical_data_slice) or {}
            support_resistance = self.trend_analyzer.identify_support_resistance(historical_data_slice) or {}
            patterns = self.trend_analyzer.detect_patterns(historical_data_slice) or {}
            breakout = self.trend_analyzer.predict_breakout(historical_data_slice, support_resistance) or {}
            market_regime = self.trend_analyzer.get_market_regime(historical_data_slice) or {}

            prediction_direction, prediction_confidence, expected_return = 'neutral', 0, 0
            prophet_predictor = ProphetModel(symbol=symbol, prediction_days=1)
            if prophet_predictor.load_model():
                raw_predictions = prophet_predictor.predict(historical_data_slice, days=1)
                if not raw_predictions.empty:
                    predicted_value = raw_predictions['Close'].iloc[0]
                    current_price = historical_data_slice['close'].iloc[-1] # Use standardized 'close'
                    if current_price > 0:
                        expected_return = ((predicted_value / current_price) - 1) * 100
                        if expected_return > 1.0: prediction_direction, prediction_confidence = 'bullish', min(100, expected_return * 20)
                        elif expected_return < -1.0: prediction_direction, prediction_confidence = 'bearish', min(100, abs(expected_return) * 20)
                        else: prediction_direction, prediction_confidence = 'neutral', 100 - (abs(expected_return) * 20)

            # Scoring Logic
            buy_score, sell_score = 0, 0
            if trend_analysis.get('overall_trend') in ['bullish', 'strong_bullish']: buy_score += trend_analysis.get('trend_strength', 50) * 0.3
            elif trend_analysis.get('overall_trend') in ['bearish', 'strong_bearish']: sell_score += trend_analysis.get('trend_strength', 50) * 0.3
            if breakout.get('prediction') == 'bullish': buy_score += breakout.get('confidence', 0) * 0.2
            elif breakout.get('prediction') == 'bearish': sell_score += breakout.get('confidence', 0) * 0.2
            if prediction_direction == 'bullish': buy_score += prediction_confidence * 0.3
            elif prediction_direction == 'bearish': sell_score += prediction_confidence * 0.3
            strong_bull_pattern = any(p['strength'] > 80 for p in patterns.get('patterns', []) if p['type'] == 'bullish' and p['days_ago'] <= 2)
            strong_bear_pattern = any(p['strength'] > 80 for p in patterns.get('patterns', []) if p['type'] == 'bearish' and p['days_ago'] <= 2)
            if strong_bull_pattern: buy_score += 15
            if strong_bear_pattern: sell_score += 15

            # Determine Signal
            if buy_score > 55 and buy_score > sell_score + 10: return 'BUY'
            elif sell_score > 55 and sell_score > buy_score + 10: return 'SELL'
            else: return 'HOLD'

        except Exception as e:
            logger.error(f"Error generating backtesting signal for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return 'HOLD'

