# modules/market_analyzer.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta  # Technical Analysis library

class MarketAnalyzer:
    """
    Analyzes market data to identify trends, correlations, and generate trading signals.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_technical_indicators(self, market_data):
        """
        Calculate technical indicators for the given market data.
        
        Args:
            market_data (DataFrame): OHLCV market data
            
        Returns:
            DataFrame: Market data with technical indicators
        """
        df = market_data.copy()
        
        # Add RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # Add MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_high'] = bollinger.bollinger_hband()
        df['bollinger_low'] = bollinger.bollinger_lband()
        
        # Add Simple Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        
        # Add Average True Range (ATR)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        return df
    
    def identify_market_regime(self, market_data):
        """
        Identify the current market regime (bullish, bearish, volatile, etc.)
        
        Args:
            market_data (DataFrame): Market data with technical indicators
            
        Returns:
            str: Market regime
        """
        df = market_data.copy()
        
        # Calculate volatility (rolling standard deviation)
        if 'Close' in df.columns:
            volatility = df['Close'].pct_change().rolling(window=20).std()
            current_volatility = volatility.iloc[-1] if not volatility.empty else 0
            
            # Calculate trend using SMA
            if 'sma_50' in df.columns and 'sma_200' in df.columns:
                # Check for golden cross or death cross
                golden_cross = df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] <= df['sma_200'].iloc[-2]
                death_cross = df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1] and df['sma_50'].iloc[-2] >= df['sma_200'].iloc[-2]
                
                # Determine trend
                if df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]:
                    trend = "bullish"
                else:
                    trend = "bearish"
                
                # Determine regime based on trend and volatility
                if trend == "bullish":
                    if current_volatility > 0.02:  # High volatility
                        return "volatile_bullish"
                    else:
                        return "stable_bullish"
                else:  # bearish
                    if current_volatility > 0.02:
                        return "volatile_bearish"
                    else:
                        return "stable_bearish"
            
        return "unknown"
    
    def generate_trading_signals(self, market_data):
        """
        Generate trading signals based on technical indicators.
        
        Args:
            market_data (DataFrame): Market data with technical indicators
            
        Returns:
            DataFrame: Trading signals
        """
        signals = pd.DataFrame(index=market_data.index)
        signals['price'] = market_data['Close']
        
        # Initialize signals
        signals['signal'] = 0
        
        # MACD Signal
        if 'macd' in market_data.columns and 'macd_signal' in market_data.columns:
            # Buy signal when MACD crosses above signal line
            signals.loc[(market_data['macd'] > market_data['macd_signal']) & 
                       (market_data['macd'].shift(1) <= market_data['macd_signal'].shift(1)), 'signal'] = 1
            
            # Sell signal when MACD crosses below signal line
            signals.loc[(market_data['macd'] < market_data['macd_signal']) & 
                       (market_data['macd'].shift(1) >= market_data['macd_signal'].shift(1)), 'signal'] = -1
        
        # RSI signals
        if 'rsi' in market_data.columns:
            # Oversold condition (RSI < 30)
            signals.loc[market_data['rsi'] < 30, 'rsi_signal'] = 1
            
            # Overbought condition (RSI > 70)
            signals.loc[market_data['rsi'] > 70, 'rsi_signal'] = -1
            
        # Simple Moving Average signals
        if 'sma_20' in market_data.columns and 'sma_50' in market_data.columns:
            # Golden Cross: Short-term SMA crosses above long-term SMA
            signals.loc[(market_data['sma_20'] > market_data['sma_50']) & 
                       (market_data['sma_20'].shift(1) <= market_data['sma_50'].shift(1)), 'sma_signal'] = 1
            
            # Death Cross: Short-term SMA crosses below long-term SMA
            signals.loc[(market_data['sma_20'] < market_data['sma_50']) & 
                       (market_data['sma_20'].shift(1) >= market_data['sma_50'].shift(1)), 'sma_signal'] = -1
            
        return signals
    
    def analyze_correlations(self, market_data_dict):
        """
        Analyze correlations between different assets.
        
        Args:
            market_data_dict (dict): Dictionary of market data for different assets
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Extract closing prices for each asset
        closing_prices = {}
        for symbol, data in market_data_dict.items():
            if not data.empty and 'Close' in data.columns:
                closing_prices[symbol] = data['Close']
        
        # Create a DataFrame with all closing prices
        if closing_prices:
            df = pd.DataFrame(closing_prices)
            correlation_matrix = df.corr()
            return correlation_matrix
        
        return pd.DataFrame()
