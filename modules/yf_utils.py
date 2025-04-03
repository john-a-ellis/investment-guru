# modules/yf_utils.py
"""
Utility functions for interacting with Yahoo Finance API.
Provides consistent session management to avoid connection pool warnings.
"""
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import pandas as pd

# Global session object that can be reused
_yf_session = None

def get_yf_session():
    """
    Get or create a session for Yahoo Finance API calls.
    Reuses the same session across the application to manage connection pools.
    
    Returns:
        requests.Session: Configured session object
    """
    global _yf_session
    
    if _yf_session is None:
        _yf_session = create_yf_session()
    
    return _yf_session

def create_yf_session():
    """
    Create a custom session for Yahoo Finance API calls with proper retry logic
    and connection pool settings to avoid warnings.
    
    Returns:
        requests.Session: Configured session object
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=1,  # Time factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Allow retries on these methods
    )
    
    # Create adapter with the retry strategy and larger pool size
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=50,  # Increased from default
        pool_maxsize=50  # Increased from default
    )
    
    # Mount for both http and https
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def download_yf_data(symbols, start=None, end=None, period="1mo", auto_adjust=False):
    """
    Download Yahoo Finance data with proper session management.
    
    Args:
        symbols (str or list): Symbol or list of symbols
        start (datetime): Start date (optional)
        end (datetime): End date (optional)
        period (str): Period (optional, default is "1mo")
        auto_adjust (bool): Whether to auto-adjust data (optional)
        
    Returns:
        DataFrame: Historical data
    """
    session = get_yf_session()
    
    try:
        data = yf.download(
            symbols, 
            start=start, 
            end=end, 
            period=period, 
            auto_adjust=auto_adjust,
            progress=False,  # Disable progress bar to avoid noise in logs
            session=session  # Use our managed session
        )
        return data
    except Exception as e:
        print(f"Error downloading Yahoo Finance data: {e}")
        try:
            # If session approach fails, fall back to default 
            # but still disable progress bar
            return yf.download(
                symbols, 
                start=start, 
                end=end, 
                period=period, 
                auto_adjust=auto_adjust,
                progress=False
            )
        except Exception as e2:
            print(f"Fallback download also failed: {e2}")
            # Return empty DataFrame with proper structure
            if isinstance(symbols, list) and len(symbols) > 1:
                # Multi-symbol case
                return pd.DataFrame(columns=pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], symbols]))
            else:
                # Single symbol case
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

def get_ticker_info(symbol):
    """
    Get ticker info with proper session management.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Ticker info
    """
    session = get_yf_session()
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        return ticker.info
    except Exception as e:
        print(f"Error getting ticker info for {symbol}: {e}")
        try:
            # Fall back to standard approach if session fails
            ticker = yf.Ticker(symbol)
            return ticker.info
        except:
            return {}

def get_ticker_history(symbol, start=None, end=None, period="1mo", auto_adjust=False):
    """
    Get ticker history with proper session management.
    
    Args:
        symbol (str): Stock symbol
        start (datetime): Start date (optional)
        end (datetime): End date (optional)
        period (str): Period (optional, default is "1mo")
        auto_adjust (bool): Whether to auto-adjust data (optional)
        
    Returns:
        DataFrame: Historical data
    """
    session = get_yf_session()
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        return ticker.history(start=start, end=end, period=period, auto_adjust=auto_adjust)
    except Exception as e:
        print(f"Error getting history for {symbol}: {e}")
        try:
            # Fall back to standard approach
            ticker = yf.Ticker(symbol)
            return ticker.history(start=start, end=end, period=period, auto_adjust=auto_adjust)
        except:
            return pd.DataFrame()