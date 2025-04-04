# modules/trend_analysis.py
"""
Machine learning models for trend analysis and pattern recognition in market data.
Implements trend identification, support/resistance detection, and pattern recognition.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import talib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """
    Analyzes market data to identify trends, patterns, and key levels.
    """
    def __init__(self):
        pass
    
    def detect_trend(self, df, window=20):
        """
        Detect trend direction using moving averages and momentum indicators.
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            window (int): Lookback window for trend detection
        
        Returns:
            dict: Trend analysis results
        """
        try:
            # Create a copy to avoid modifying original data
            data = df.copy()
            
            # Add moving averages
            data['SMA_short'] = self._calculate_sma(data, 20)
            data['SMA_medium'] = self._calculate_sma(data, 50)
            data['SMA_long'] = self._calculate_sma(data, 200)
            
            # Add RSI (Relative Strength Index)
            data['RSI'] = self._calculate_rsi(data, 14)
            
            # Add MACD (Moving Average Convergence Divergence)
            data['MACD'], data['MACD_signal'], data['MACD_hist'] = self._calculate_macd(data)
            
            # Determine current trend based on multiple factors
            # 1. Short-term vs. Long-term MA
            ma_trend = "bullish" if data['SMA_short'].iloc[-1] > data['SMA_long'].iloc[-1] else "bearish"
            
            # 2. Check for crossovers (signals trend change)
            ma_crossover_signal = "none"
            if (data['SMA_short'].iloc[-2] <= data['SMA_medium'].iloc[-2] and 
                data['SMA_short'].iloc[-1] > data['SMA_medium'].iloc[-1]):
                ma_crossover_signal = "bullish"  # Golden Cross-like (short MA crosses above medium MA)
            elif (data['SMA_short'].iloc[-2] >= data['SMA_medium'].iloc[-2] and 
                  data['SMA_short'].iloc[-1] < data['SMA_medium'].iloc[-1]):
                ma_crossover_signal = "bearish"  # Death Cross-like (short MA crosses below medium MA)
            
            # 3. RSI trend
            rsi_value = data['RSI'].iloc[-1]
            rsi_trend = "neutral"
            if rsi_value > 70:
                rsi_trend = "overbought"
            elif rsi_value < 30:
                rsi_trend = "oversold"
            elif rsi_value > 50:
                rsi_trend = "bullish"
            else:
                rsi_trend = "bearish"
            
            # 4. MACD trend
            macd_trend = "neutral"
            if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                macd_trend = "bullish"
            else:
                macd_trend = "bearish"
            
            # 5. MACD histogram trend (momentum)
            macd_hist_trend = "neutral"
            if data['MACD_hist'].iloc[-1] > 0 and data['MACD_hist'].iloc[-1] > data['MACD_hist'].iloc[-2]:
                macd_hist_trend = "bullish_increasing"
            elif data['MACD_hist'].iloc[-1] > 0 and data['MACD_hist'].iloc[-1] < data['MACD_hist'].iloc[-2]:
                macd_hist_trend = "bullish_decreasing"
            elif data['MACD_hist'].iloc[-1] < 0 and data['MACD_hist'].iloc[-1] < data['MACD_hist'].iloc[-2]:
                macd_hist_trend = "bearish_increasing"
            elif data['MACD_hist'].iloc[-1] < 0 and data['MACD_hist'].iloc[-1] > data['MACD_hist'].iloc[-2]:
                macd_hist_trend = "bearish_decreasing"
            
            # 6. Price action - compare with history
            if window <= len(data):
                recent_data = data.iloc[-window:]
                price_trend = "neutral"
                
                # Calculate percentage change over the window
                price_change = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
                
                # Determine trend based on price change
                if price_change > 5:  # Strong uptrend (more than 5% gain)
                    price_trend = "strong_bullish"
                elif price_change > 1:  # Weak uptrend (1-5% gain)
                    price_trend = "weak_bullish"
                elif price_change < -5:  # Strong downtrend (more than 5% loss)
                    price_trend = "strong_bearish"
                elif price_change < -1:  # Weak downtrend (1-5% loss)
                    price_trend = "weak_bearish"
            else:
                price_trend = "neutral"
            
            # Determine overall trend strength (0-100%)
            trend_factors = {
                'ma_trend': 1 if ma_trend == "bullish" else -1,
                'ma_crossover': 1 if ma_crossover_signal == "bullish" else -1 if ma_crossover_signal == "bearish" else 0,
                'rsi_trend': 1 if rsi_trend == "bullish" else -1 if rsi_trend == "bearish" else 0,
                'macd_trend': 1 if macd_trend == "bullish" else -1 if macd_trend == "bearish" else 0,
                'price_trend': 2 if price_trend == "strong_bullish" else 1 if price_trend == "weak_bullish" else -2 if price_trend == "strong_bearish" else -1 if price_trend == "weak_bearish" else 0
            }
            
            # Calculate weighted trend score
            weights = {
                'ma_trend': 0.2,
                'ma_crossover': 0.15,
                'rsi_trend': 0.15,
                'macd_trend': 0.2,
                'price_trend': 0.3
            }
            
            trend_score = sum(trend_factors[factor] * weights[factor] for factor in trend_factors)
            
            # Normalize to 0-100% and determine overall trend
            normalized_score = (trend_score + 1) / 2 * 100
            if normalized_score > 70:
                overall_trend = "strong_bullish"
            elif normalized_score > 55:
                overall_trend = "bullish"
            elif normalized_score > 45:
                overall_trend = "neutral"
            elif normalized_score > 30:
                overall_trend = "bearish"
            else:
                overall_trend = "strong_bearish"
            
            return {
                'overall_trend': overall_trend,
                'trend_strength': normalized_score,
                'details': {
                    'ma_trend': ma_trend,
                    'ma_crossover': ma_crossover_signal,
                    'rsi_value': rsi_value,
                    'rsi_trend': rsi_trend,
                    'macd_trend': macd_trend,
                    'macd_momentum': macd_hist_trend,
                    'price_action': price_trend
                },
                'latest_data': {
                    'current_price': data['Close'].iloc[-1],
                    'sma_20': data['SMA_short'].iloc[-1],
                    'sma_50': data['SMA_medium'].iloc[-1],
                    'sma_200': data['SMA_long'].iloc[-1],
                    'rsi': data['RSI'].iloc[-1],
                    'macd': data['MACD'].iloc[-1],
                    'macd_signal': data['MACD_signal'].iloc[-1],
                    'macd_hist': data['MACD_hist'].iloc[-1]
                }
            }
        
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            import traceback
            traceback.print_exc()
            return {
                'prediction': 'neutral',
                'confidence': 0,
                'details': f'Error in prediction: {str(e)}'
            }
    
    def get_market_regime(self, df, window=252):
        """
        Identify the current market regime (bull, bear, sideways, volatile bull, volatile bear).
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            window (int): Lookback window for regime analysis
        
        Returns:
            dict: Market regime analysis results
        """
        try:
            # Get the recent data window
            if len(df) < window:
                recent_data = df.copy()
                logger.warning(f"Not enough data for full market regime analysis (needed {window}, got {len(df)})")
            else:
                recent_data = df.tail(window).copy()
            
            # Calculate returns
            recent_data['daily_return'] = recent_data['Close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            volatility = recent_data['daily_return'].std() * np.sqrt(252)  # Annualized
            
            # Calculate trend using linear regression on log prices
            import statsmodels.api as sm
            log_prices = np.log(recent_data['Close'])
            x = np.arange(len(log_prices))
            x = sm.add_constant(x)
            model = sm.OLS(log_prices, x)
            results = model.fit()
            slope = results.params[1]
            
            # Calculate R-squared of the trend line (trend strength)
            r_squared = results.rsquared
            
            # Calculate drawdowns
            peak = recent_data['Close'].expanding().max()
            drawdown = (recent_data['Close'] / peak - 1) * 100
            max_drawdown = drawdown.min()
            
            # Calculate 52-week (or max window) high and low
            high_52wk = recent_data['High'].max()
            low_52wk = recent_data['Low'].min()
            current_price = recent_data['Close'].iloc[-1]
            
            # Calculate price position relative to 52-week range
            price_position = (current_price - low_52wk) / (high_52wk - low_52wk) if high_52wk != low_52wk else 0.5
            
            # Define volatility thresholds
            high_volatility_threshold = 0.20  # 20% annualized volatility
            
            # Define trend thresholds
            annualized_trend = slope * 252
            strong_uptrend_threshold = 0.15  # 15% annualized trend
            uptrend_threshold = 0.05  # 5% annualized trend
            downtrend_threshold = -0.05  # -5% annualized trend
            strong_downtrend_threshold = -0.15  # -15% annualized trend
            
            # Define trend based on slope and volatility
            if annualized_trend > strong_uptrend_threshold:
                trend = "strong_bull"
            elif annualized_trend > uptrend_threshold:
                trend = "bull"
            elif annualized_trend < strong_downtrend_threshold:
                trend = "strong_bear"
            elif annualized_trend < downtrend_threshold:
                trend = "bear"
            else:
                trend = "sideways"
            
            # Define regime
            is_volatile = volatility > high_volatility_threshold
            
            if is_volatile:
                if trend in ["bull", "strong_bull"]:
                    regime = "volatile_bull"
                elif trend in ["bear", "strong_bear"]:
                    regime = "volatile_bear"
                else:
                    regime = "volatile_sideways"
            else:
                if trend in ["bull", "strong_bull"]:
                    regime = "stable_bull"
                elif trend in ["bear", "strong_bear"]:
                    regime = "stable_bear"
                else:
                    regime = "stable_sideways"
            
            # Calculate additional regime metrics
            recent_volatility = recent_data['daily_return'].tail(20).std() * np.sqrt(252)
            volatility_trend = "increasing" if recent_volatility > volatility else "decreasing"
            
            # Calculate the percentage of positive/negative days
            positive_days = (recent_data['daily_return'] > 0).sum() / len(recent_data)
            
            return {
                'regime': regime,
                'trend': trend,
                'volatility': volatility,
                'volatility_trend': volatility_trend,
                'max_drawdown': max_drawdown,
                'positive_days': positive_days * 100,  # Convert to percentage
                'r_squared': r_squared,
                'price_position': price_position,
                'annualized_return': annualized_trend * 100,  # Convert to percentage
                'metrics': {
                    'current_price': current_price,
                    'high_52wk': high_52wk,
                    'low_52wk': low_52wk
                }
            }
        
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            import traceback
            traceback.print_exc()
            return {
                'regime': 'unknown',
                'trend': 'unknown',
                'volatility': 0,
                'volatility_trend': 'unknown',
                'max_drawdown': 0,
                'positive_days': 0,
                'r_squared': 0,
                'price_position': 0.5,
                'annualized_return': 0,
                'metrics': {}
            }
    
    def analyze_market_cycles(self, df, window=1008):  # Default to 4 years (252 trading days * 4)
        """
        Analyze market cycles for macro trend analysis.
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            window (int): Lookback window for cycle analysis
        
        Returns:
            dict: Cycle analysis results
        """
        try:
            # Get the historical data window
            if len(df) < window:
                historical_data = df.copy()
                logger.warning(f"Not enough data for full cycle analysis (needed {window}, got {len(df)})")
            else:
                historical_data = df.tail(window).copy()
            
            # Calculate daily returns
            historical_data['daily_return'] = historical_data['Close'].pct_change()
            
            # Calculate cumulative returns
            historical_data['cum_return'] = (1 + historical_data['daily_return']).cumprod()
            
            # Identify bull and bear markets
            # A bear market is typically defined as a 20% decline from a peak
            # A bull market is typically defined as a 20% increase from a trough
            
            # Find peaks and troughs
            from scipy.signal import find_peaks
            
            # Convert to numpy array for signal processing
            prices = historical_data['Close'].values
            cum_returns = historical_data['cum_return'].values
            
            # Find peaks (potential market tops)
            peaks, _ = find_peaks(prices, distance=63)  # Minimum 3 months between peaks
            
            # Find troughs (potential market bottoms)
            # Invert prices to find troughs as peaks
            troughs, _ = find_peaks(-prices, distance=63)  # Minimum 3 months between troughs
            
            # Use the cumulative return to calculate drawdowns
            historical_data['drawdown'] = historical_data['Close'] / historical_data['Close'].expanding().max() - 1
            
            # Identify significant bear markets (20%+ drawdown)
            bear_markets = []
            significant_drawdown_threshold = -0.2  # 20% drawdown
            
            for trough_idx in troughs:
                # Find the preceding peak
                preceding_peaks = [p for p in peaks if p < trough_idx]
                if not preceding_peaks:
                    continue
                
                closest_peak = max(preceding_peaks)
                
                # Calculate drawdown from peak to trough
                peak_price = prices[closest_peak]
                trough_price = prices[trough_idx]
                drawdown = trough_price / peak_price - 1
                
                if drawdown <= significant_drawdown_threshold:
                    bear_markets.append({
                        'peak_date': historical_data.index[closest_peak].strftime('%Y-%m-%d'),
                        'trough_date': historical_data.index[trough_idx].strftime('%Y-%m-%d'),
                        'duration_days': trough_idx - closest_peak,
                        'drawdown': drawdown * 100,  # Convert to percentage
                        'peak_price': peak_price,
                        'trough_price': trough_price
                    })
            
            # Identify significant bull markets (20%+ gain from trough)
            bull_markets = []
            significant_rally_threshold = 0.2  # 20% rally
            
            for peak_idx in peaks:
                # Find the preceding trough
                preceding_troughs = [t for t in troughs if t < peak_idx]
                if not preceding_troughs:
                    continue
                
                closest_trough = max(preceding_troughs)
                
                # Calculate rally from trough to peak
                trough_price = prices[closest_trough]
                peak_price = prices[peak_idx]
                rally = peak_price / trough_price - 1
                
                if rally >= significant_rally_threshold:
                    bull_markets.append({
                        'trough_date': historical_data.index[closest_trough].strftime('%Y-%m-%d'),
                        'peak_date': historical_data.index[peak_idx].strftime('%Y-%m-%d'),
                        'duration_days': peak_idx - closest_trough,
                        'rally': rally * 100,  # Convert to percentage
                        'trough_price': trough_price,
                        'peak_price': peak_price
                    })
            
            # Perform spectral analysis for cyclical patterns
            # This is advanced, so we'll use a simple FFT (Fast Fourier Transform)
            from scipy.fft import fft
            
            # Ensure we have enough data
            if len(historical_data) >= 252:  # At least 1 year of data
                # Detrend the data for better cycle detection
                from scipy import signal
                detrended = signal.detrend(historical_data['Close'].values)
                
                # Calculate FFT
                fft_values = fft(detrended)
                
                # Get power spectrum
                power = np.abs(fft_values) ** 2
                
                # Get frequencies
                sample_freq = np.fft.fftfreq(len(detrended))
                
                # Convert frequencies to periods (in days)
                positive_freq_idx = np.where(sample_freq > 0)[0]
                freqs = sample_freq[positive_freq_idx]
                powers = power[positive_freq_idx]
                
                # Find the dominant cycle periods
                sorted_periods = sorted(zip(1/freqs, powers), key=lambda x: x[1], reverse=True)
                
                # Get the top 3 cycles
                dominant_cycles = []
                for period, power_value in sorted_periods[:3]:
                    if 10 <= period <= 1000:  # Only consider cycles between 10 and 1000 days
                        # Convert period to calendar days
                        days = int(period)
                        # Rough conversion to months/years for readability
                        if days < 30:
                            period_str = f"{days} days"
                        elif days < 365:
                            period_str = f"{days/30:.1f} months"
                        else:
                            period_str = f"{days/365:.1f} years"
                        
                        dominant_cycles.append({
                            'period_days': days,
                            'period_str': period_str,
                            'power': power_value,
                            'confidence': min(100, power_value / np.max(powers) * 100)  # Normalize to 0-100%
                        })
            else:
                dominant_cycles = []
            
            # Determine the current position in the market cycle
            current_price = historical_data['Close'].iloc[-1]
            all_time_high = historical_data['Close'].max()
            all_time_low = historical_data['Close'].min()
            current_drawdown = (current_price / all_time_high - 1) * 100
            
            # Determine if we're in a bull or bear market currently
            is_bear_market = current_drawdown <= -20
            
            # Calculate days since all time high
            all_time_high_idx = historical_data['Close'].idxmax()
            days_since_ath = (historical_data.index[-1] - all_time_high_idx).days
            
            # Determine cycle position using a combination of indicators
            if current_price == all_time_high:
                cycle_position = "peak"
            elif is_bear_market and current_drawdown > -30:
                cycle_position = "early_bear"
            elif is_bear_market and current_drawdown <= -30:
                cycle_position = "late_bear"
            elif not is_bear_market and days_since_ath < 90:
                cycle_position = "early_bull"
            elif not is_bear_market and days_since_ath >= 90:
                cycle_position = "late_bull"
            else:
                cycle_position = "uncertain"
            
            return {
                'bull_markets': bull_markets,
                'bear_markets': bear_markets,
                'dominant_cycles': dominant_cycles,
                'current_cycle': {
                    'position': cycle_position,
                    'current_price': current_price,
                    'all_time_high': all_time_high,
                    'all_time_low': all_time_low,
                    'current_drawdown': current_drawdown,
                    'days_since_ath': days_since_ath
                }
            }
        
        except Exception as e:
            logger.error(f"Error analyzing market cycles: {e}")
            import traceback
            traceback.print_exc()
            return {
                'bull_markets': [],
                'bear_markets': [],
                'dominant_cycles': [],
                'current_cycle': {
                    'position': 'unknown',
                    'current_price': 0,
                    'all_time_high': 0,
                    'all_time_low': 0,
                    'current_drawdown': 0,
                    'days_since_ath': 0
                }
            }
    
    def _calculate_sma(self, df, window=20):
        """Calculate Simple Moving Average"""
        try:
            return df['Close'].rolling(window=window).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_ema(self, df, window=20):
        """Calculate Exponential Moving Average"""
        try:
            return df['Close'].ewm(span=window, adjust=False).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_rsi(self, df, window=14):
        """Calculate Relative Strength Index"""
        try:
            # Calculate price changes
            delta = df['Close'].diff()
            
            # Separate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=df.index)
    
    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            # Calculate EMAs
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            
            # Calculate MACD
            macd = ema_fast - ema_slow
            
            # Calculate signal line
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            macd_hist = macd - macd_signal
            
            return macd, macd_signal, macd_hist
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=df.index), pd.Series(index=df.index), pd.Series(index=df.index)
    
    def _calculate_bollinger_bands(self, df, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            # Calculate middle band (SMA)
            middle_band = df['Close'].rolling(window=window).mean()
            
            # Calculate standard deviation
            std = df['Close'].rolling(window=window).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            
            return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            if 'Close' in df.columns and len(df) > 0:
                # Return current price as all bands if calculation fails
                current_price = df['Close'].iloc[-1]
                return current_price, current_price, current_price
            return 0, 0, 0
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        try:
            # Calculate True Range
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=window).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0
    
    def _find_extrema_levels(self, df, window_size=10):
        """Find local extrema (peaks and troughs) for support/resistance"""
        try:
            # Find peaks (local maxima)
            peaks_idx = argrelextrema(df['High'].values, np.greater_equal, order=window_size)[0]
            peaks = df['High'].iloc[peaks_idx]
            
            # Find troughs (local minima)
            troughs_idx = argrelextrema(df['Low'].values, np.less_equal, order=window_size)[0]
            troughs = df['Low'].iloc[troughs_idx]
            
            return {
                'peaks': peaks.values.tolist(),
                'troughs': troughs.values.tolist()
            }
        except Exception as e:
            logger.error(f"Error finding extrema levels: {e}")
            return {'peaks': [], 'troughs': []}
    
    def _find_price_clusters(self, df, n_clusters=6):
        """Find price clusters for support/resistance using K-means"""
        try:
            # Combine high and low prices for clustering
            prices = np.concatenate([df['High'].values, df['Low'].values])
            
            # Reshape for K-means
            X = prices.reshape(-1, 1)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Get cluster centers (support/resistance levels)
            levels = sorted(kmeans.cluster_centers_.flatten())
            
            return levels
        except Exception as e:
            logger.error(f"Error finding price clusters: {e}")
            return []
    
    def _calculate_pivot_points(self, df):
        """Calculate pivot points (classic method)"""
        try:
            # Get the most recent data point
            if len(df) < 1:
                return {}
            
            last_data = df.iloc[-1]
            
            # Calculate pivot points
            high = last_data['High']
            low = last_data['Low']
            close = last_data['Close']
            
            pivot = (high + low + close) / 3
            support1 = (2 * pivot) - high
            support2 = pivot - (high - low)
            resistance1 = (2 * pivot) - low
            resistance2 = pivot + (high - low)
            
            return {
                'pivot': pivot,
                'support1': support1,
                'support2': support2,
                'resistance1': resistance1,
                'resistance2': resistance2
            }
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    def _detect_double_patterns(self, df):
        """Detect double top/bottom patterns"""
        try:
            # Find peaks and troughs
            peaks_idx = argrelextrema(df['High'].values, np.greater_equal, order=5)[0]
            troughs_idx = argrelextrema(df['Low'].values, np.less_equal, order=5)[0]
            
            patterns = []
            current_idx = len(df) - 1
            
            # Look for double tops
            if len(peaks_idx) >= 2:
                # Get the last two peaks
                last_peak = peaks_idx[-1]
                second_last_peak = peaks_idx[-2]
                
                # Check if they are relatively close in time and price
                time_diff = abs(last_peak - second_last_peak)
                price_diff = abs(df['High'].iloc[last_peak] - df['High'].iloc[second_last_peak])
                price_percent_diff = price_diff / df['High'].iloc[second_last_peak]
                
                # Double top conditions
                if (time_diff <= 20 and  # Peaks within 20 bars
                    price_percent_diff <= 0.03 and  # Peaks within 3% of each other
                    current_idx - last_peak <= 5):  # Pattern is recent
                    
                    patterns.append({
                        'name': 'Double Top',
                        'type': 'bearish',
                        'date': df.index[last_peak].strftime('%Y-%m-%d'),
                        'days_ago': current_idx - last_peak,
                        'strength': 85  # Double tops are relatively reliable
                    })
            
            # Look for double bottoms
            if len(troughs_idx) >= 2:
                # Get the last two troughs
                last_trough = troughs_idx[-1]
                second_last_trough = troughs_idx[-2]
                
                # Check if they are relatively close in time and price
                time_diff = abs(last_trough - second_last_trough)
                price_diff = abs(df['Low'].iloc[last_trough] - df['Low'].iloc[second_last_trough])
                price_percent_diff = price_diff / df['Low'].iloc[second_last_trough]
                
                # Double bottom conditions
                if (time_diff <= 20 and  # Troughs within 20 bars
                    price_percent_diff <= 0.03 and  # Troughs within 3% of each other
                    current_idx - last_trough <= 5):  # Pattern is recent
                    
                    patterns.append({
                        'name': 'Double Bottom',
                        'type': 'bullish',
                        'date': df.index[last_trough].strftime('%Y-%m-%d'),
                        'days_ago': current_idx - last_trough,
                        'strength': 85  # Double bottoms are relatively reliable
                    })
            
            return patterns
        except Exception as e:
            logger.error(f"Error detecting double patterns: {e}")
            return []
    
    def _detect_head_shoulders(self, df):
        """Detect head and shoulders patterns"""
        try:
            # Find peaks and troughs
            peaks_idx = argrelextrema(df['High'].values, np.greater_equal, order=5)[0]
            troughs_idx = argrelextrema(df['Low'].values, np.less_equal, order=5)[0]
            
            patterns = []
            current_idx = len(df) - 1
            
            # Head and shoulders requires at least 3 peaks and 2 troughs in sequence
            if len(peaks_idx) >= 3 and len(troughs_idx) >= 2:
                # Check the pattern sequence for regular head and shoulders (bearish)
                for i in range(len(peaks_idx) - 2):
                    # Get 3 consecutive peaks
                    left_peak = peaks_idx[i]
                    head_peak = peaks_idx[i+1]
                    right_peak = peaks_idx[i+2]
                    
                    # Head should be higher than shoulders
                    if (df['High'].iloc[head_peak] > df['High'].iloc[left_peak] and 
                        df['High'].iloc[head_peak] > df['High'].iloc[right_peak]):
                        
                        # Left and right shoulders should be at similar height
                        shoulder_diff = abs(df['High'].iloc[left_peak] - df['High'].iloc[right_peak])
                        shoulder_percent_diff = shoulder_diff / df['High'].iloc[left_peak]
                        
                        if shoulder_percent_diff <= 0.10:  # Shoulders within 10% of each other
                            # Pattern is formed, add to list if recent
                            if current_idx - right_peak <= 10:  # Pattern completed recently
                                patterns.append({
                                    'name': 'Head and Shoulders',
                                    'type': 'bearish',
                                    'date': df.index[right_peak].strftime('%Y-%m-%d'),
                                    'days_ago': current_idx - right_peak,
                                    'strength': 90  # Head and shoulders are very reliable
                                })
                                break  # Only report the most recent pattern
            
            # Inverse head and shoulders (for troughs)
            if len(troughs_idx) >= 3 and len(peaks_idx) >= 2:
                # Check the pattern sequence for inverse head and shoulders (bullish)
                for i in range(len(troughs_idx) - 2):
                    # Get 3 consecutive troughs
                    left_trough = troughs_idx[i]
                    head_trough = troughs_idx[i+1]
                    right_trough = troughs_idx[i+2]
                    
                    # Head should be lower than shoulders
                    if (df['Low'].iloc[head_trough] < df['Low'].iloc[left_trough] and 
                        df['Low'].iloc[head_trough] < df['Low'].iloc[right_trough]):
                        
                        # Left and right shoulders should be at similar height
                        shoulder_diff = abs(df['Low'].iloc[left_trough] - df['Low'].iloc[right_trough])
                        shoulder_percent_diff = shoulder_diff / df['Low'].iloc[left_trough]
                        
                        if shoulder_percent_diff <= 0.10:  # Shoulders within 10% of each other
                            # Pattern is formed, add to list if recent
                            if current_idx - right_trough <= 10:  # Pattern completed recently
                                patterns.append({
                                    'name': 'Inverse Head and Shoulders',
                                    'type': 'bullish',
                                    'date': df.index[right_trough].strftime('%Y-%m-%d'),
                                    'days_ago': current_idx - right_trough,
                                    'strength': 90  # Inverse head and shoulders are very reliable
                                })
                                break  # Only report the most recent pattern
            
            return patterns
        except Exception as e:
            logger.error(f"Error detecting head and shoulders patterns: {e}")
            return []
            traceback.print_exc()
            return {
                'overall_trend': "unknown",
                'trend_strength': 50,
                'details': {},
                'latest_data': {}
            }
    
    def identify_support_resistance(self, df, n_levels=3, window_size=10):
        """
        Identify support and resistance levels using multiple methods.
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            n_levels (int): Number of support/resistance levels to identify
            window_size (int): Window size for local extrema detection
        
        Returns:
            dict: Support and resistance levels
        """
        try:
            data = df.copy()
            
            # Method 1: Local extrema (peaks and troughs)
            extrema_levels = self._find_extrema_levels(data, window_size)
            
            # Method 2: Historical price clustering using K-means
            cluster_levels = self._find_price_clusters(data, n_clusters=n_levels*2)
            
            # Method 3: Pivot points
            pivot_levels = self._calculate_pivot_points(data)
            
            # Method 4: Moving averages as dynamic support/resistance
            ma_levels = {
                'sma_50': self._calculate_sma(data, 50).iloc[-1],
                'sma_100': self._calculate_sma(data, 100).iloc[-1],
                'sma_200': self._calculate_sma(data, 200).iloc[-1]
            }
            
            # Combine all methods
            all_levels = []
            
            # Add extrema levels (peaks and troughs)
            for level_type, levels in extrema_levels.items():
                for level in levels:
                    all_levels.append({
                        'price': level,
                        'type': level_type,
                        'method': 'extrema',
                        'strength': 0  # Will calculate strength below
                    })
            
            # Add cluster levels
            for i, level in enumerate(cluster_levels):
                all_levels.append({
                    'price': level,
                    'type': 'cluster',
                    'method': 'kmeans',
                    'strength': 0
                })
            
            # Add pivot point levels
            for level_type, level in pivot_levels.items():
                all_levels.append({
                    'price': level,
                    'type': level_type,
                    'method': 'pivot',
                    'strength': 0
                })
            
            # Add MA levels
            for ma_type, level in ma_levels.items():
                all_levels.append({
                    'price': level,
                    'type': ma_type,
                    'method': 'moving_average',
                    'strength': 0
                })
            
            # Calculate strength for each level based on proximity to price and historical touches
            current_price = data['Close'].iloc[-1]
            
            for level in all_levels:
                # Proximity factor (closer levels are more relevant)
                proximity = abs(level['price'] - current_price) / current_price
                proximity_score = max(0, 1 - proximity * 10)  # Scale to 0-1 (0% to 100%)
                
                # Historical touches factor (more touches = stronger level)
                price_tolerance = data['Close'].std() * 0.2  # 20% of standard deviation
                
                # Count how many times price has approached this level
                touches = 0
                for i in range(1, len(data)):
                    # Check if price crossed or approached the level
                    if (min(data['Low'].iloc[i-1], data['Low'].iloc[i]) <= level['price'] <= max(data['High'].iloc[i-1], data['High'].iloc[i]) or
                        abs(data['Close'].iloc[i] - level['price']) <= price_tolerance):
                        touches += 1
                
                # Normalize touches score
                touches_score = min(1, touches / 10)  # Cap at 10 touches for a perfect score
                
                # Calculate overall strength (50% proximity, 50% historical touches)
                level['strength'] = (proximity_score * 0.5 + touches_score * 0.5) * 100
                
                # Add distance from current price
                level['distance_pct'] = ((level['price'] / current_price) - 1) * 100
            
            # Sort levels by price
            all_levels.sort(key=lambda x: x['price'])
            
            # Filter to strongest levels and categorize as support or resistance
            support_levels = []
            resistance_levels = []
            
            for level in all_levels:
                if level['price'] < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
            
            # Sort by strength (descending) and take top n_levels
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            top_support = support_levels[:n_levels] if support_levels else []
            top_resistance = resistance_levels[:n_levels] if resistance_levels else []
            
            return {
                'support': top_support,
                'resistance': top_resistance,
                'current_price': current_price
            }
        
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            import traceback
            traceback.print_exc()
            return {
                'support': [],
                'resistance': [],
                'current_price': df['Close'].iloc[-1] if not df.empty else None
            }
    
    def detect_patterns(self, df, window=25):
        """
        Detect chart patterns using TA-Lib pattern recognition.
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            window (int): Window size for pattern detection
        
        Returns:
            dict: Detected patterns with confidence scores
        """
        try:
            # Ensure we have enough data
            if len(df) < window:
                logger.warning(f"Not enough data for pattern detection (need {window}, got {len(df)})")
                return {'patterns': []}
            
            # Dictionary mapping TA-Lib pattern functions to pattern names
            pattern_functions = {
                'CDL3LINESTRIKE': '3 Line Strike',
                'CDLABANDONEDBABY': 'Abandoned Baby',
                'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
                'CDLENGULFING': 'Engulfing Pattern',
                'CDLEVENINGDOJISTAR': 'Evening Doji Star',
                'CDLEVENINGSTAR': 'Evening Star',
                'CDLHAMMER': 'Hammer',
                'CDLHANGINGMAN': 'Hanging Man',
                'CDLHARAMI': 'Harami Pattern',
                'CDLHARAMICROSS': 'Harami Cross',
                'CDLMARUBOZU': 'Marubozu',
                'CDLMORNINGDOJISTAR': 'Morning Doji Star',
                'CDLMORNINGSTAR': 'Morning Star',
                'CDLPIERCING': 'Piercing Pattern',
                'CDLSHOOTINGSTAR': 'Shooting Star',
                'CDLDOJI': 'Doji'
            }
            
            # Get the recent data window
            recent_data = df.tail(window).copy()
            
            # Detect patterns using TA-Lib
            detected_patterns = []
            
            for func_name, pattern_name in pattern_functions.items():
                try:
                    # Get the pattern recognition function from talib
                    pattern_func = getattr(talib, func_name)
                    
                    # Apply the function
                    result = pattern_func(
                        recent_data['Open'].values, 
                        recent_data['High'].values,
                        recent_data['Low'].values, 
                        recent_data['Close'].values
                    )
                    
                    # Check if pattern exists in the most recent candles (last 3)
                    for i in range(1, min(4, len(result))):
                        if result[-i] != 0:
                            bullish = result[-i] > 0
                            
                            days_ago = i - 1
                            date_str = recent_data.index[-i].strftime('%Y-%m-%d')
                            
                            # Determine pattern strength/reliability
                            strength = min(100, abs(result[-i]) * 25)  # Scale to 0-100
                            
                            detected_patterns.append({
                                'name': pattern_name,
                                'type': 'bullish' if bullish else 'bearish',
                                'date': date_str,
                                'days_ago': days_ago,
                                'strength': strength
                            })
                            
                            # Only log the most recent occurrence of each pattern
                            break
                
                except Exception as pattern_error:
                    logger.warning(f"Error detecting pattern {pattern_name}: {pattern_error}")
                    continue
            
            # Sort by days_ago (most recent first) and then by strength (strongest first)
            detected_patterns.sort(key=lambda x: (x['days_ago'], -x['strength']))
            
            # Check for double top/bottom patterns (not built into TA-Lib)
            double_patterns = self._detect_double_patterns(recent_data)
            if double_patterns:
                detected_patterns.extend(double_patterns)
            
            # Check for head and shoulders pattern
            hs_patterns = self._detect_head_shoulders(recent_data)
            if hs_patterns:
                detected_patterns.extend(hs_patterns)
            
            return {'patterns': detected_patterns}
        
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            import traceback
            traceback.print_exc()
            return {'patterns': []}
    
    def predict_breakout(self, df, support_resistance=None):
        """
        Predict potential breakout direction based on price action and indicators.
        
        Args:
            df (DataFrame): Market data with OHLCV columns
            support_resistance (dict): Support and resistance levels (optional)
        
        Returns:
            dict: Breakout prediction details
        """
        try:
            # Get recent data (last 20 days)
            recent_data = df.tail(20).copy()
            
            # Get current price
            current_price = recent_data['Close'].iloc[-1]
            
            # If support_resistance not provided, calculate it
            if support_resistance is None:
                support_resistance = self.identify_support_resistance(df)
            
            # Get closest support and resistance levels
            closest_support = None
            closest_resistance = None
            
            if support_resistance['support']:
                closest_support = max(support_resistance['support'], key=lambda x: x['price'])['price']
            
            if support_resistance['resistance']:
                closest_resistance = min(support_resistance['resistance'], key=lambda x: x['price'])['price']
            
            # If no levels found, return no prediction
            if closest_support is None or closest_resistance is None:
                return {
                    'prediction': 'neutral',
                    'confidence': 0,
                    'details': 'Insufficient support/resistance levels'
                }
            
            # Calculate price position within the range
            range_size = closest_resistance - closest_support
            if range_size <= 0:
                return {
                    'prediction': 'neutral',
                    'confidence': 0,
                    'details': 'Invalid price range'
                }
            
            # Calculate where the current price is in the range (0-1)
            price_position = (current_price - closest_support) / range_size
            
            # Calculate volatility (ATR - Average True Range)
            atr = self._calculate_atr(recent_data)
            
            # Calculate volume trends
            volume_change = recent_data['Volume'].pct_change().mean() * 100
            
            # Get momentum indicators
            recent_data['RSI'] = self._calculate_rsi(recent_data)
            recent_data['MACD'], recent_data['MACD_signal'], recent_data['MACD_hist'] = self._calculate_macd(recent_data)
            
            # Compile factors that suggest breakout direction
            factors = {}
            
            # Factor 1: Price position in the range
            if price_position > 0.7:
                factors['price_position'] = {'direction': 'bullish', 'strength': (price_position - 0.5) * 2}
            elif price_position < 0.3:
                factors['price_position'] = {'direction': 'bearish', 'strength': (0.5 - price_position) * 2}
            else:
                factors['price_position'] = {'direction': 'neutral', 'strength': 0}
            
            # Factor 2: Volume trend
            if volume_change > 10:  # Volume increasing
                factors['volume_trend'] = {'direction': 'bullish', 'strength': min(1, volume_change / 50)}
            elif volume_change < -10:  # Volume decreasing
                factors['volume_trend'] = {'direction': 'bearish', 'strength': min(1, abs(volume_change) / 50)}
            else:
                factors['volume_trend'] = {'direction': 'neutral', 'strength': 0}
            
            # Factor 3: RSI direction
            rsi = recent_data['RSI'].iloc[-1]
            rsi_change = recent_data['RSI'].diff().mean()
            
            if rsi > 50 and rsi_change > 0:
                factors['rsi'] = {'direction': 'bullish', 'strength': min(1, (rsi - 50) / 20)}
            elif rsi < 50 and rsi_change < 0:
                factors['rsi'] = {'direction': 'bearish', 'strength': min(1, (50 - rsi) / 20)}
            else:
                factors['rsi'] = {'direction': 'neutral', 'strength': 0}
            
            # Factor 4: MACD direction
            macd = recent_data['MACD'].iloc[-1]
            macd_signal = recent_data['MACD_signal'].iloc[-1]
            macd_hist = recent_data['MACD_hist'].iloc[-1]
            
            if macd > macd_signal and macd_hist > 0:
                factors['macd'] = {'direction': 'bullish', 'strength': min(1, macd_hist / 0.5)}
            elif macd < macd_signal and macd_hist < 0:
                factors['macd'] = {'direction': 'bearish', 'strength': min(1, abs(macd_hist) / 0.5)}
            else:
                factors['macd'] = {'direction': 'neutral', 'strength': 0}
            
            # Factor 5: Bollinger Band position
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(recent_data)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            if current_price > bb_upper:
                factors['bollinger'] = {'direction': 'bullish', 'strength': min(1, (current_price - bb_upper) / bb_width)}
            elif current_price < bb_lower:
                factors['bollinger'] = {'direction': 'bearish', 'strength': min(1, (bb_lower - current_price) / bb_width)}
            else:
                # Neutral but with a slight bias based on position within the bands
                band_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                if band_position > 0.5:
                    factors['bollinger'] = {'direction': 'slightly_bullish', 'strength': (band_position - 0.5) * 2}
                else:
                    factors['bollinger'] = {'direction': 'slightly_bearish', 'strength': (0.5 - band_position) * 2}
            
            # Combine all factors to predict breakout direction
            bullish_score = 0
            bearish_score = 0
            factor_count = 0
            
            # Weights for different factors
            weights = {
                'price_position': 0.25,
                'volume_trend': 0.15,
                'rsi': 0.2,
                'macd': 0.2,
                'bollinger': 0.2
            }
            
            for factor, data in factors.items():
                factor_weight = weights.get(factor, 0.2)  # Default weight if not specified
                if data['direction'] == 'bullish':
                    bullish_score += data['strength'] * factor_weight
                elif data['direction'] == 'bearish':
                    bearish_score += data['strength'] * factor_weight
                elif data['direction'] == 'slightly_bullish':
                    bullish_score += data['strength'] * factor_weight * 0.5
                elif data['direction'] == 'slightly_bearish':
                    bearish_score += data['strength'] * factor_weight * 0.5
                factor_count += 1
            
            # Calculate net score (-1 to 1)
            net_score = bullish_score - bearish_score
            
            # Determine breakout prediction and confidence
            if net_score > 0.2:
                prediction = 'bullish'
                confidence = net_score * 100
            elif net_score < -0.2:
                prediction = 'bearish'
                confidence = abs(net_score) * 100
            else:
                prediction = 'neutral'
                confidence = (1 - abs(net_score)) * 100
            
            # Limit confidence to 0-100%
            confidence = max(0, min(100, confidence))
            
            # Additional details for the prediction
            details = {
                'factors': factors,
                'price_range': {
                    'support': closest_support,
                    'resistance': closest_resistance,
                    'current': current_price,
                    'position': price_position
                },
                'indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_hist': macd_hist,
                    'bollinger': {
                        'upper': bb_upper,
                        'middle': bb_middle,
                        'lower': bb_lower,
                        'width': bb_width
                    },
                    'atr': atr
                }
            }
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'details': details
            }
        
        except Exception as e:
            logger.error(f"Error predicting breakout: {e}")
            import traceback