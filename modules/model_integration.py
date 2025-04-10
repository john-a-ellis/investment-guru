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

# Import custom modules
from modules.price_prediction import get_price_predictions, train_price_prediction_models
from modules.trend_analysis import TrendAnalyzer
from modules.data_provider import DataProvider
from modules.portfolio_utils import load_portfolio
from modules.price_prediction import ProphetModel

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
        self.data_provider = DataProvider()
        self.trend_analyzer = TrendAnalyzer()
        self.prediction_cache = {}
        self.trend_cache = {}
        self.model_training_status = {}
        
        # Create directories for storing model-related data
        self.model_dir = "models"
        self.data_dir = "data/model_data"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize background thread for model training
        self.training_executor = ThreadPoolExecutor(max_workers=1)
    
    def get_asset_analysis(self, symbol, days_to_predict=30, force_refresh=False):
        """
        Get comprehensive analysis for a specific asset, including price predictions
        and trend analysis. Ensures default values for failed sub-analyses.

        Args:
            symbol (str): Asset symbol
            days_to_predict (int): Number of days to predict prices for
            force_refresh (bool): Force refresh of cached data

        Returns:
            dict: Comprehensive asset analysis or None on critical failure
        """
        # --- DEFINE CACHE KEY ---
        cache_key = f"{symbol}_analysis_{days_to_predict}"

        # --- TEMPORARILY DISABLE CACHE CHECK FOR DEBUGGING ---
        force_refresh = True # ADD THIS LINE TO FORCE RECALCULATION
        # --- END TEMPORARY CHANGE ---

        # --- OPTIONAL: RE-ENABLE CACHE CHECK ---
        if cache_key in self.prediction_cache and not force_refresh:
            cache_time, data = self.prediction_cache[cache_key]
            # Return cached data if less than 1 hour old (adjust as needed)
            if (datetime.now() - cache_time).total_seconds() < 3600:
                logger.info(f"Returning cached analysis for {symbol}")
                return data
        # --- END OPTIONAL CACHE CHECK ---

        try:
            # Get historical data for analysis
            # --- Use self.data_provider, NOT self.data_collector ---
            historical_data_df = self.data_provider.get_historical_price(symbol=symbol, period="1y")
            # Convert the DataFrame back to the dictionary format expected by the rest of the function
            historical_data = {symbol: historical_data_df} if not historical_data_df.empty else {}

            if symbol not in historical_data or historical_data[symbol].empty:
                logger.error(f"No historical data available for {symbol}")
                return None # Return None if no historical data

            df = historical_data[symbol]

            # --- Run analyses and ensure default values on failure ---
            trend_analysis = self.trend_analyzer.detect_trend(df)
            if trend_analysis is None:
                logger.warning(f"Trend analysis returned None for {symbol}. Using default.")
                trend_analysis = {'overall_trend': 'unknown', 'trend_strength': 50, 'details': {}, 'latest_data': {}} # Provide default structure

            support_resistance = self.trend_analyzer.identify_support_resistance(df)
            if support_resistance is None:
                logger.warning(f"Support/Resistance analysis returned None for {symbol}. Using default.")
                support_resistance = {'support': [], 'resistance': [], 'current_price': df['close'].iloc[-1] if not df.empty else None} # Default structure

            patterns = self.trend_analyzer.detect_patterns(df)
            if patterns is None:
                logger.warning(f"Pattern detection returned None for {symbol}. Using default.")
                patterns = {'patterns': []} # Default structure

            # Pass potentially defaulted support_resistance to predict_breakout
            breakout = self.trend_analyzer.predict_breakout(df, support_resistance if support_resistance else None)
            if breakout is None:
                logger.warning(f"Breakout prediction returned None for {symbol}. Using default.")
                breakout = {'prediction': 'neutral', 'confidence': 0, 'details': 'Analysis failed'} # Default structure

            market_regime = self.trend_analyzer.get_market_regime(df)
            if market_regime is None:
                logger.warning(f"Market regime analysis returned None for {symbol}. Using default.")
                market_regime = {'regime': 'unknown', 'trend': 'unknown'} # Default structure

            # Get price predictions (already has fallback mechanisms)
            predictions = get_price_predictions(symbol, days=days_to_predict)
            if predictions is None:
                 logger.warning(f"Price prediction returned None for {symbol}. Using default.")
                 # Use the simple fallback structure if get_price_predictions fails completely
                 from modules.price_prediction import create_simple_fallback
                 predictions = create_simple_fallback(symbol, days_to_predict)

            # Check if model training is in progress
            training_status = self.model_training_status.get(symbol, {'status': 'not_started'}) # Ensure dict format

            # --- Combine all analyses ---
            analysis = {
                'symbol': symbol,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'price': {
                    'current': df['close'].iloc[-1],
                    'change_1d': (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0,
                    'change_1w': (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) > 5 else 0,
                    'change_1m': (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100 if len(df) > 20 else 0,
                },
                'trend': trend_analysis, # Now guaranteed to be a dict
                'support_resistance': support_resistance, # Now guaranteed to be a dict
                'patterns': patterns, # Now guaranteed to be a dict
                'breakout': breakout, # Now guaranteed to be a dict
                'market_regime': market_regime, # Now guaranteed to be a dict
                'price_predictions': predictions, # Now guaranteed to be a dict
                'model_training': training_status # Already handled
            }

            # Cache the results (This line will now work)
            self.prediction_cache[cache_key] = (datetime.now(), analysis)

            return analysis

        except Exception as e:
            logger.error(f"Critical Error in get_asset_analysis for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None # Return None on any critical failure
    
    def train_models_for_symbol(self, symbol, lookback_period="2y", async_training=True):
        """
        Train ML models for a specific symbol. This runs in a separate thread to avoid
        blocking the main application.
        
        Args:
            symbol (str): Asset symbol
            lookback_period (str): Period of historical data to use for training
            async_training (bool): Whether to run training asynchronously
            
        Returns:
            str: Training status
        """
        from datetime import datetime
        
        # Define the nested _do_training function with improved error handling
        def _do_training(symbol, lookback_period):
            """
            Internal method to perform model training for a specific symbol.
            
            Args:
                symbol (str): Asset symbol
                lookback_period (str): Lookback period for training data
                
            Returns:
                dict: Training status and metrics
            """
            print(f"Starting detailed model training for {symbol}")
            try:
                # Update training status
                self.model_training_status[symbol] = {
                    'status': 'in_progress',
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Train price prediction models
                from modules.price_prediction import train_price_prediction_models
                model_data = train_price_prediction_models(symbol=symbol, lookback_period=lookback_period)
                
                # Debug output to see what's coming back
                print(f"Received model_data type: {type(model_data)}")
                print(f"Model data content: {model_data}")
                
                # Check for errors or problematic model_data format
                if isinstance(model_data, str):
                    error_message = model_data
                    print(f"Error training models for {symbol}: {error_message}")
                    self.model_training_status[symbol] = {
                        'status': 'failed',
                        'error': error_message,
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    return {"status": "failed", "error": error_message}
                    
                if isinstance(model_data, dict) and "error" in model_data:
                    error_message = model_data["error"]
                    print(f"Error training models for {symbol}: {error_message}")
                    self.model_training_status[symbol] = {
                        'status': 'failed',
                        'error': error_message,
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    return {"status": "failed", "error": error_message}
                
                # Handle cases where model_data is not in expected format
                if not isinstance(model_data, dict):
                    print(f"Warning: Unexpected model_data format for {symbol}: {type(model_data)}")
                    try:
                        # Create a basic structure to continue
                        model_data = {
                            'prophet': {
                                'model': None,
                                'metrics': {
                                    'mae': 0.0,
                                    'mse': 0.0,
                                    'rmse': 0.0,
                                    'mape': 0.0,
                                    'error': f"Model data in unexpected format: {type(model_data)}"
                                }
                            }
                        }
                    except Exception as parse_error:
                        print(f"Error parsing model data for {symbol}: {parse_error}")
                        self.model_training_status[symbol] = {
                            'status': 'failed',
                            'error': f"Model data format error: {parse_error}",
                            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        return {"status": "failed", "error": f"Model data format error: {parse_error}"}
                
                # Get model metrics - adjust to handle the expected structure from train_price_prediction_models
                if 'prophet' in model_data and isinstance(model_data['prophet'], dict) and 'metrics' in model_data['prophet']:
                    metrics = model_data['prophet']['metrics']
                else:
                    metrics = model_data.get('metrics', {})
                
                # Debug output
                print(f"Metrics for {symbol}: {metrics}")
                
                # Check if we have valid metrics
                if metrics and 'error' in metrics:
                    error_message = metrics['error']
                    print(f"Error in metrics for {symbol}: {error_message}")
                    self.model_training_status[symbol] = {
                        'status': 'failed',
                        'error': error_message,
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    return {"status": "failed", "error": error_message}
                else:
                    print(f"Training completed successfully for {symbol}")
                    self.model_training_status[symbol] = {
                        'status': 'completed',
                        'metrics': metrics,
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    return {"status": "completed", "metrics": metrics}
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_message = f"Error in model training process: {str(e)}"
                print(error_message)
                
                self.model_training_status[symbol] = {
                    'status': 'failed',
                    'error': error_message,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return {"status": "failed", "error": error_message}
        
        # Initialize status with consistent nested structure
        self.model_training_status[symbol] = {
            'status': 'pending',
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Define a wrapper function that will be run in a separate thread
        def training_thread_wrapper():
            try:
                print(f"Starting training thread for {symbol}")
                _do_training(symbol, lookback_period)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Exception in training thread wrapper: {str(e)}")
                self.model_training_status[symbol] = {
                    'status': 'failed',
                    'error': f"Thread error: {str(e)}",
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # Run training either synchronously or asynchronously
        if async_training:
            # Submit the training task to the thread pool
            self.training_executor.submit(training_thread_wrapper)
            return "pending"  # Status will be updated by the thread
        else:
            # Run synchronously (mainly for testing)
            result = _do_training(symbol, lookback_period)
            return result.get("status", "failed")
            
    def get_portfolio_recommendations(self, use_ml=True):
        """
        Generate portfolio recommendations based on ML models and trend analysis.
        
        Args:
            use_ml (bool): Whether to use ML predictions or just trend analysis
        
        Returns:
            dict: Portfolio recommendations
        """
        try:
            # Load current portfolio
            portfolio = load_portfolio()
            
            # Get asset symbols from current portfolio
            portfolio_symbols = set()
            for investment_id, details in portfolio.items():
                symbol = details.get("symbol", "")
                if symbol:
                    portfolio_symbols.add(symbol)
            
            # Get user profile
            from modules.portfolio_utils import load_user_profile
            user_profile = load_user_profile()
            risk_level = user_profile.get("risk_level", 5)
            
            # Get tracked assets
            from modules.portfolio_utils import load_tracked_assets
            tracked_assets = load_tracked_assets()
            
            # Combine portfolio and tracked assets
            all_symbols = portfolio_symbols.union(tracked_assets.keys())
            
            # Analyze each asset
            asset_analyses = {}
            recommendations = {
                'buy': [],
                'hold': [],
                'sell': [],
                'portfolio_score': 0,
            }
            
            for symbol in all_symbols:
                # Skip empty symbols
                if not symbol:
                    continue
                
                # Get comprehensive analysis
                analysis = self.get_asset_analysis(symbol)
                
                if analysis:
                    asset_analyses[symbol] = analysis
                    
                    # Generate recommendation based on analysis
                    recommendation = self._generate_recommendation(symbol, analysis, portfolio, risk_level, use_ml)
                    
                    if recommendation:
                        action = recommendation.get('action', '')
                        if action == 'buy':
                            recommendations['buy'].append(recommendation)
                        elif action == 'sell':
                            recommendations['sell'].append(recommendation)
                        elif action == 'hold':
                            recommendations['hold'].append(recommendation)
            
            # Sort recommendations by confidence
            recommendations['buy'] = sorted(recommendations['buy'], key=lambda x: x.get('confidence', 0), reverse=True)
            recommendations['sell'] = sorted(recommendations['sell'], key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Calculate portfolio score
            if portfolio:
                total_value = sum(float(inv.get("current_value", 0)) for inv in portfolio.values())
                weighted_score = 0
                
                for investment_id, details in portfolio.items():
                    symbol = details.get("symbol", "")
                    current_value = float(details.get("current_value", 0))
                    
                    if symbol in asset_analyses and total_value > 0:
                        # Weight score by portfolio allocation
                        weight = current_value / total_value
                        analysis = asset_analyses[symbol]
                        
                        # Calculate asset score (0-100)
                        asset_score = 0
                        
                        # Factor 1: Trend strength (0-100)
                        trend_strength = analysis['trend'].get('trend_strength', 50)
                        
                        # Factor 2: Breakout prediction (0-100)
                        breakout = analysis['breakout']
                        breakout_confidence = breakout.get('confidence', 50)
                        breakout_direction = breakout.get('prediction', 'neutral')
                        
                        if breakout_direction == 'bullish':
                            breakout_score = 50 + (breakout_confidence / 2)
                        elif breakout_direction == 'bearish':
                            breakout_score = 50 - (breakout_confidence / 2)
                        else:
                            breakout_score = 50
                        
                        # Factor 3: Price prediction (if available)
                        prediction_score = 50
                        if use_ml and analysis['price_predictions']:
                            predictions = analysis['price_predictions']
                            current_price = analysis['price']['current']
                            
                            # Get predicted prices
                            predicted_values = predictions.get('values', [])
                            
                            if predicted_values:
                                # Calculate expected return
                                final_prediction = predicted_values[-1]
                                expected_return = (final_prediction / current_price - 1) * 100
                                
                                # Convert to score (0-100)
                                if expected_return > 10:
                                    prediction_score = 100
                                elif expected_return > 0:
                                    prediction_score = 50 + (expected_return * 5)
                                elif expected_return > -10:
                                    prediction_score = 50 + (expected_return * 5)
                                else:
                                    prediction_score = 0
                        
                        # Calculate overall asset score
                        asset_score = (trend_strength * 0.4) + (breakout_score * 0.3) + (prediction_score * 0.3)
                        
                        # Add to weighted score
                        weighted_score += asset_score * weight
                
                # Set portfolio score
                recommendations['portfolio_score'] = weighted_score
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            import traceback
            traceback.print_exc()
            return {
                'buy': [],
                'hold': [],
                'sell': [],
                'portfolio_score': 0,
                'error': str(e)
            }
    
    def _generate_recommendation(self, symbol, analysis, portfolio, risk_level, use_ml=True):
        """
        Generate a recommendation for a specific asset based on analysis.
        
        Args:
            symbol (str): Asset symbol
            analysis (dict): Comprehensive asset analysis
            portfolio (dict): Current portfolio
            risk_level (int): User risk tolerance (1-10)
            use_ml (bool): Whether to use ML predictions
        
        Returns:
            dict: Recommendation details
        """
        try:
            # Check if asset is in portfolio
            in_portfolio = False
            portfolio_details = None
            
            for investment_id, details in portfolio.items():
                if details.get("symbol", "") == symbol:
                    in_portfolio = True
                    portfolio_details = details
                    break
            
            # Get current price
            current_price = analysis['price']['current']
            
            # Get trend analysis
            trend = analysis['trend']
            overall_trend = trend.get('overall_trend', 'neutral')
            trend_strength = trend.get('trend_strength', 50)
            
            # Get support/resistance
            support_resistance = analysis['support_resistance']
            current_support = None
            current_resistance = None
            
            if 'support' in support_resistance and support_resistance['support']:
                current_support = max([s['price'] for s in support_resistance['support']])
            
            if 'resistance' in support_resistance and support_resistance['resistance']:
                current_resistance = min([r['price'] for r in support_resistance['resistance']])
            
            # Get breakout prediction
            breakout = analysis['breakout']
            breakout_prediction = breakout.get('prediction', 'neutral')
            breakout_confidence = breakout.get('confidence', 0)
            
            # Get market regime
            market_regime = analysis['market_regime']
            regime = market_regime.get('regime', 'unknown')
            
            # Get price patterns
            patterns = analysis['patterns']
            bullish_patterns = [p for p in patterns.get('patterns', []) if p.get('type') == 'bullish']
            bearish_patterns = [p for p in patterns.get('patterns', []) if p.get('type') == 'bearish']
            
            # Get ML price predictions if available and requested
            prediction_direction = 'neutral'
            prediction_confidence = 0
            expected_return = 0
            
            if use_ml and 'price_predictions' in analysis and analysis['price_predictions']:
                predictions = analysis['price_predictions']
                predicted_values = predictions.get('values', [])
                
                if predicted_values:
                    # Calculate expected return
                    final_prediction = predicted_values[-1]
                    expected_return = (final_prediction / current_price - 1) * 100
                    
                    # Determine direction and confidence
                    if expected_return > 5:  # More than 5% return
                        prediction_direction = 'bullish'
                        prediction_confidence = min(100, expected_return * 10)  # Scale up for stronger signal
                    elif expected_return < -5:  # More than 5% loss
                        prediction_direction = 'bearish'
                        prediction_confidence = min(100, abs(expected_return) * 10)
                    else:
                        prediction_direction = 'neutral'
                        prediction_confidence = 100 - (abs(expected_return) * 10)  # Higher confidence in neutrality as return approaches 0
            
            # Calculate buy/sell signals based on multiple factors
            buy_signals = []
            sell_signals = []
            
            # Factor 1: Trend Analysis
            if overall_trend in ['bullish', 'strong_bullish']:
                buy_signals.append({
                    'factor': 'trend',
                    'strength': trend_strength,
                    'description': f"{overall_trend.replace('_', ' ').title()} trend detected"
                })
            elif overall_trend in ['bearish', 'strong_bearish']:
                sell_signals.append({
                    'factor': 'trend',
                    'strength': trend_strength,
                    'description': f"{overall_trend.replace('_', ' ').title()} trend detected"
                })
            
            # Factor 2: Support/Resistance
            if current_support and current_resistance:
                # Calculate where price is in the range
                range_size = current_resistance - current_support
                if range_size > 0:
                    position = (current_price - current_support) / range_size
                    
                    if position < 0.2:  # Near support (good buying opportunity)
                        buy_signals.append({
                            'factor': 'support',
                            'strength': 80,
                            'description': f"Price near support level (${current_support:.2f})"
                        })
                    elif position > 0.8:  # Near resistance (good selling opportunity)
                        sell_signals.append({
                            'factor': 'resistance',
                            'strength': 80,
                            'description': f"Price near resistance level (${current_resistance:.2f})"
                        })
            
            # Factor 3: Breakout Prediction
            if breakout_prediction == 'bullish' and breakout_confidence > 60:
                buy_signals.append({
                    'factor': 'breakout',
                    'strength': breakout_confidence,
                    'description': f"Bullish breakout predicted with {breakout_confidence:.0f}% confidence"
                })
            elif breakout_prediction == 'bearish' and breakout_confidence > 60:
                sell_signals.append({
                    'factor': 'breakout',
                    'strength': breakout_confidence,
                    'description': f"Bearish breakout predicted with {breakout_confidence:.0f}% confidence"
                })
            
            # Factor 4: Chart Patterns
            if bullish_patterns:
                # Use the strongest bullish pattern
                strongest_pattern = max(bullish_patterns, key=lambda p: p.get('strength', 0))
                buy_signals.append({
                    'factor': 'pattern',
                    'strength': strongest_pattern.get('strength', 50),
                    'description': f"Bullish pattern detected: {strongest_pattern.get('name')}"
                })
            
            if bearish_patterns:
                # Use the strongest bearish pattern
                strongest_pattern = max(bearish_patterns, key=lambda p: p.get('strength', 0))
                sell_signals.append({
                    'factor': 'pattern',
                    'strength': strongest_pattern.get('strength', 50),
                    'description': f"Bearish pattern detected: {strongest_pattern.get('name')}"
                })
            
            # Factor 5: ML Price Prediction
            if prediction_direction == 'bullish' and prediction_confidence > 50:
                buy_signals.append({
                    'factor': 'ml_prediction',
                    'strength': prediction_confidence,
                    'description': f"ML model predicts {expected_return:.1f}% potential return"
                })
            elif prediction_direction == 'bearish' and prediction_confidence > 50:
                sell_signals.append({
                    'factor': 'ml_prediction',
                    'strength': prediction_confidence,
                    'description': f"ML model predicts {expected_return:.1f}% potential loss"
                })
            
            # Factor 6: Market Regime
            if regime in ['stable_bull', 'volatile_bull']:
                buy_signals.append({
                    'factor': 'market_regime',
                    'strength': 70,
                    'description': f"Favorable market regime: {regime.replace('_', ' ').title()}"
                })
            elif regime in ['stable_bear', 'volatile_bear']:
                sell_signals.append({
                    'factor': 'market_regime',
                    'strength': 70,
                    'description': f"Unfavorable market regime: {regime.replace('_', ' ').title()}"
                })
            
            # Calculate average signal strengths
            avg_buy_strength = sum(signal['strength'] for signal in buy_signals) / len(buy_signals) if buy_signals else 0
            avg_sell_strength = sum(signal['strength'] for signal in sell_signals) / len(sell_signals) if sell_signals else 0
            
            # Adjust for risk level (1-10)
            # Higher risk tolerance increases buy signals and decreases sell signals
            risk_adjustment = (risk_level - 5) / 5  # -1 to 1
            
            if risk_adjustment > 0:  # Higher risk tolerance
                avg_buy_strength = min(100, avg_buy_strength * (1 + risk_adjustment * 0.3))
                avg_sell_strength = max(0, avg_sell_strength * (1 - risk_adjustment * 0.3))
            elif risk_adjustment < 0:  # Lower risk tolerance
                avg_buy_strength = max(0, avg_buy_strength * (1 + risk_adjustment * 0.3))
                avg_sell_strength = min(100, avg_sell_strength * (1 - risk_adjustment * 0.3))
            
            # Determine action based on signal strengths
            action = 'hold'
            confidence = 0
            trade_amount = 0
            
            if avg_buy_strength > 70 and avg_buy_strength > avg_sell_strength:
                action = 'buy'
                confidence = avg_buy_strength
                # Higher investment for stronger signals
                trade_amount = 1000 * (confidence / 100)
            elif avg_sell_strength > 70 and avg_sell_strength > avg_buy_strength and in_portfolio:
                action = 'sell'
                confidence = avg_sell_strength
                if portfolio_details:
                    # Calculate sell amount based on confidence
                    current_value = float(portfolio_details.get("current_value", 0))
                    trade_amount = current_value * (confidence / 100) if confidence > 0 else 0
            else:
                action = 'hold'
                # For hold, confidence is how confident we are it's neither a buy nor sell
                if avg_buy_strength < 30 and avg_sell_strength < 30:
                    confidence = 80  # Confident hold (no strong signals either way)
                else:
                    confidence = 100 - max(avg_buy_strength, avg_sell_strength)  # Less confident hold
            
            # Create recommendation
            recommendation = {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'trade_amount': trade_amount,
                'expected_return': expected_return,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'in_portfolio': in_portfolio,
                'market_regime': regime,
                'overall_trend': overall_trend,
                'indicators': {
                    'rsi': trend.get('details', {}).get('rsi_value', 0),
                    'macd': trend.get('details', {}).get('macd_trend', ''),
                    'price_prediction': prediction_direction
                }
            }
            
            return recommendation
        
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_model_training_batch(self, symbols=None, lookback_period="2y"):
        """
        Run model training for a batch of symbols.
        
        Args:
            symbols (list): List of symbols to train models for
            lookback_period (str): Historical period to use for training
        
        Returns:
            dict: Training status for each symbol
        """
        if symbols is None:
            # Load current portfolio and tracked assets
            portfolio = load_portfolio()
            
            portfolio_symbols = set()
            for investment_id, details in portfolio.items():
                symbol = details.get("symbol", "")
                if symbol:
                    portfolio_symbols.add(symbol)
            
            # Get tracked assets
            from modules.portfolio_utils import load_tracked_assets
            tracked_assets = load_tracked_assets()
            
            # Combine portfolio and tracked assets
            symbols = list(portfolio_symbols.union(tracked_assets.keys()))
        
        logger.info(f"Starting batch training for {len(symbols)} symbols")
        
        # Train models for each symbol
        results = {}
        for symbol in symbols:
            try:
                status = self.train_models_for_symbol(symbol, lookback_period, async_training=True)
                results[symbol] = status
            except Exception as e:
                logger.error(f"Error starting training for {symbol}: {e}")
                results[symbol] = f"Error: {str(e)}"
        
        return results
    
    def get_model_training_status(self, symbol=None):
        """
        Get status of model training for a specific symbol or all symbols.
        
        Args:
            symbol (str): Symbol to get status for (None for all)
        
        Returns:
            dict: Training status
        """
        if symbol:
            status = self.model_training_status.get(symbol, "not_started")
            updated = self.model_training_status.get(f"{symbol}_updated", "never")
            
            return {
                'symbol': symbol,
                'status': status,
                'last_updated': updated
            }
        else:
            # Return status for all symbols
            status_dict = {}
            
            for key, value in self.model_training_status.items():
                if not key.endswith("_updated") and key != "batch_training":
                    status_dict[key] = {
                        'symbol': key,
                        'status': value,
                        'last_updated': self.model_training_status.get(f"{key}_updated", "never")
                    }
            
            return status_dict
        
    def get_backtesting_recommendation_signal(self, symbol, historical_data_slice):
        """
        Generates a simple Buy/Sell/Hold signal for backtesting based on historical data slice.

        Args:
            symbol (str): Asset symbol.
            historical_data_slice (pd.DataFrame): DataFrame with historical OHLCV data up to
                                                  the day *before* the decision date.

        Returns:
            str: 'BUY', 'SELL', or 'HOLD'.
        """
        try:
            # --- 1. Perform Trend Analysis on the Slice ---
            # Ensure TrendAnalyzer is initialized (it's in __init__)
            if not hasattr(self, 'trend_analyzer'): self.trend_analyzer = TrendAnalyzer()

            trend_analysis = self.trend_analyzer.detect_trend(historical_data_slice)
            support_resistance = self.trend_analyzer.identify_support_resistance(historical_data_slice)
            patterns = self.trend_analyzer.detect_patterns(historical_data_slice)
            breakout = self.trend_analyzer.predict_breakout(historical_data_slice, support_resistance)
            market_regime = self.trend_analyzer.get_market_regime(historical_data_slice)

            # --- 2. Get Price Prediction (using pre-trained model) ---
            prediction_direction = 'neutral'
            prediction_confidence = 0
            expected_return = 0
            predicted_value = None

            # Load the pre-trained Prophet model for this symbol
            # NOTE: Assumes model was trained *before* the backtest period started.
            # For realistic backtests, periodic retraining within the loop might be needed.
            prophet_predictor = ProphetModel(symbol=symbol, prediction_days=1) # Predict just 1 day ahead for signal
            if prophet_predictor.load_model():
                # Predict using the historical slice
                # IMPORTANT: DO NOT use the "sanity check" override here. Test the raw model output.
                raw_predictions = prophet_predictor.predict(historical_data_slice, days=1)

                if not raw_predictions.empty:
                    predicted_value = raw_predictions['Close'].iloc[0]
                    current_price = historical_data_slice['Close'].iloc[-1]

                    if current_price > 0:
                        expected_return = ((predicted_value / current_price) - 1) * 100
                        if expected_return > 1.0: # Example threshold for BUY signal based on 1-day prediction
                            prediction_direction = 'bullish'
                            prediction_confidence = min(100, expected_return * 20) # Scale confidence
                        elif expected_return < -1.0: # Example threshold for SELL
                            prediction_direction = 'bearish'
                            prediction_confidence = min(100, abs(expected_return) * 20)
                        else:
                            prediction_direction = 'neutral'
                            prediction_confidence = 100 - (abs(expected_return) * 20)

            # --- 3. Combine Factors for Simple Signal (Adapt your _generate_recommendation logic) ---
            # This is a simplified version of your _generate_recommendation heuristic logic.
            # Adapt this scoring based on your actual strategy rules.

            buy_score = 0
            sell_score = 0

            # Trend Factor
            if trend_analysis.get('overall_trend') in ['bullish', 'strong_bullish']:
                buy_score += trend_analysis.get('trend_strength', 50) * 0.3 # Weight 30%
            elif trend_analysis.get('overall_trend') in ['bearish', 'strong_bearish']:
                sell_score += trend_analysis.get('trend_strength', 50) * 0.3

            # Breakout Factor
            if breakout.get('prediction') == 'bullish':
                buy_score += breakout.get('confidence', 0) * 0.2 # Weight 20%
            elif breakout.get('prediction') == 'bearish':
                sell_score += breakout.get('confidence', 0) * 0.2

            # Prediction Factor
            if prediction_direction == 'bullish':
                 buy_score += prediction_confidence * 0.3 # Weight 30%
            elif prediction_direction == 'bearish':
                 sell_score += prediction_confidence * 0.3

            # Pattern Factor (Simple example: add bonus for strong patterns)
            strong_bull_pattern = any(p['strength'] > 80 for p in patterns.get('patterns', []) if p['type'] == 'bullish' and p['days_ago'] <= 2)
            strong_bear_pattern = any(p['strength'] > 80 for p in patterns.get('patterns', []) if p['type'] == 'bearish' and p['days_ago'] <= 2)
            if strong_bull_pattern:
                buy_score += 15 # Bonus score
            if strong_bear_pattern:
                sell_score += 15

            # --- 4. Determine Final Signal ---
            # Define thresholds for BUY/SELL signals
            buy_threshold = 55 # Example threshold
            sell_threshold = 55 # Example threshold

            if buy_score > sell_score and buy_score > buy_threshold:
                return 'BUY'
            elif sell_score > buy_score and sell_score > sell_threshold:
                return 'SELL'
            else:
                return 'HOLD'

        except Exception as e:
            logger.error(f"Error generating backtesting signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return 'HOLD' # Default to HOLD on error