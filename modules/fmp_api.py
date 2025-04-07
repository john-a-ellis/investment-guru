# modules/fmp_api.py
"""
Financial Modeling Prep (FMP) API integration module.
Provides unified access to FMP API endpoints with proper error handling and caching.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FMPApi:
    """
    Financial Modeling Prep API client with caching and error handling.
    """
    
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            logger.warning("FMP_API_KEY not found in environment variables")
        
        # Base URL for API endpoints
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Initialize caches with expiration times
        self.price_cache = {}  # Cache for price data
        self.profile_cache = {}  # Cache for company profiles
        self.financial_cache = {}  # Cache for financial statements
        self.news_cache = {}  # Cache for news articles
        
        # Configurable cache expiration (in seconds)
        self.cache_expiry = {
            'price': 300,  # 5 minutes for price data
            'daily_price': 3600,  # 1 hour for daily price data
            'profile': 86400,  # 24 hours for company profiles
            'financial': 86400,  # 24 hours for financial statements
            'news': 1800  # 30 minutes for news
        }
        
        # Rate limiting settings
        self.request_interval = 0.2  # Minimum time between requests (200ms)
        self.last_request_time = 0
        
        # Maximum retries
        self.max_retries = 3
    
    def _make_request(self, endpoint, params=None, retries=0):
        """
        Make an API request with rate limiting and retry logic.
        
        Args:
            endpoint (str): API endpoint to request
            params (dict): Query parameters
            retries (int): Current retry count
            
        Returns:
            dict or None: JSON response or None on failure
        """
        # Ensure we don't exceed rate limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Add API key to parameters
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        # Construct full URL
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Make the request
            response = requests.get(url, params=params, timeout=10)
            
            # Check for successful response
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                if retries < self.max_retries:
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.warning(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._make_request(endpoint, params, retries + 1)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                    return None
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error making API request to {endpoint}: {e}")
            if retries < self.max_retries:
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return self._make_request(endpoint, params, retries + 1)
            return None
    
    def get_historical_price(self, symbol, period=None, start_date=None, end_date=None):
        """
        Get historical price data for a symbol with caching and YFinance fallback.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y')
            start_date (datetime or str): Start date (if period not provided)
            end_date (datetime or str): End date (if period not provided)
            
        Returns:
            pandas.DataFrame: Historical price data or empty DataFrame on failure
        """
        # Format dates if provided
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Generate cache key
        cache_key = f"{symbol}_{period}_{start_date}_{end_date}"
        
        # Check cache
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['daily_price']:
                return data
        
        # Handle period parameter for FMP API
        if period:
            # Convert period to 'from' parameter for FMP API
            days = 0
            if period == '1d':
                days = 1
            elif period == '5d':
                days = 5
            elif period == '1mo':
                days = 30
            elif period == '3mo':
                days = 90
            elif period == '6mo':
                days = 180
            elif period == '1y':
                days = 365
            elif period == '5y':
                days = 1825
            
            # If from is specified by days
            if days > 0:
                endpoint = f"historical-price-full/{symbol}"
                params = {'from': f"{days}days"}
            else:
                # Default to 1 year if period not recognized
                endpoint = f"historical-price-full/{symbol}"
                params = {'from': '365days'}
        elif start_date:
            # If specific date range is provided
            endpoint = f"historical-price-full/{symbol}"
            params = {}
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
        else:
            # Default to 1 year of data
            endpoint = f"historical-price-full/{symbol}"
            params = {'from': '365days'}
        
        # Make the request to FMP API
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and 'historical' in response:
            # Convert to DataFrame
            df = pd.DataFrame(response['historical'])
            
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date (ascending)
            df = df.sort_index()
            
            # Rename columns to match expected format from yfinance
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjClose': 'Adj Close',
                'volume': 'Volume'
            })
            
            # Add Adj Close if not present (for FMP free tier)
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            
            # Cache the result
            self.price_cache[cache_key] = (time.time(), df)
            
            return df
        else:
            # FMP API couldn't find data - try YFinance as fallback
            logger.info(f"Data for {symbol} not found in FMP API, trying YFinance as fallback")
            
            try:
                import yfinance as yf
                from modules.yf_utils import get_ticker_history
                
                # Convert period to YFinance format if needed
                yf_period = period
                if not yf_period and start_date and end_date:
                    # Use start and end dates instead of period
                    yf_start = pd.to_datetime(start_date)
                    yf_end = pd.to_datetime(end_date)
                    hist = get_ticker_history(symbol, start=yf_start, end=yf_end)
                else:
                    # Use period parameter
                    hist = get_ticker_history(symbol, period=yf_period)
                
                if not hist.empty:
                    # IMPORTANT FIX: Convert to timezone-naive DatetimeIndex to match FMP data
                    # This ensures compatibility with the rest of the application
                    if hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    
                    # Cache the result
                    self.price_cache[cache_key] = (time.time(), hist)
                    logger.info(f"Successfully retrieved {symbol} data from YFinance")
                    return hist
                else:
                    logger.warning(f"No data found for {symbol} in YFinance either")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error getting data from YFinance for {symbol}: {e}")
                return pd.DataFrame()
    
    def get_quote(self, symbol):
        """
        Get the latest quote for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Quote data or empty dict on failure
        """
        # Generate cache key
        cache_key = f"quote_{symbol}"
        
        # Check cache
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            # Return cached data if within expiry period (shorter for quotes)
            if time.time() - cache_time < self.cache_expiry['price']:
                return data
        
        # Make the request
        endpoint = f"quote/{symbol}"
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list) and len(response) > 0:
            quote = response[0]
            
            # Cache the result
            self.price_cache[cache_key] = (time.time(), quote)
            
            return quote
        else:
            # Return empty dict on failure
            return {}
    
    def get_company_profile(self, symbol):
        """
        Get company profile information.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Company profile or empty dict on failure
        """
        # Check cache
        if symbol in self.profile_cache:
            cache_time, data = self.profile_cache[symbol]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['profile']:
                return data
        
        # Make the request
        endpoint = f"profile/{symbol}"
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list) and len(response) > 0:
            profile = response[0]
            
            # Cache the result
            self.profile_cache[symbol] = (time.time(), profile)
            
            return profile
        else:
            # Return empty dict on failure
            return {}
    
    def get_income_statement(self, symbol, period='annual', limit=4):
        """
        Get income statement data.
        
        Args:
            symbol (str): Stock symbol
            period (str): 'annual' or 'quarter'
            limit (int): Number of periods to retrieve
            
        Returns:
            list: Income statements or empty list on failure
        """
        # Generate cache key
        cache_key = f"income_{symbol}_{period}_{limit}"
        
        # Check cache
        if cache_key in self.financial_cache:
            cache_time, data = self.financial_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['financial']:
                return data
        
        # Make the request
        endpoint = f"income-statement/{symbol}"
        params = {'limit': limit}
        if period == 'quarter':
            params['period'] = 'quarter'
        
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.financial_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []
    
    def get_balance_sheet(self, symbol, period='annual', limit=4):
        """
        Get balance sheet data.
        
        Args:
            symbol (str): Stock symbol
            period (str): 'annual' or 'quarter'
            limit (int): Number of periods to retrieve
            
        Returns:
            list: Balance sheets or empty list on failure
        """
        # Generate cache key
        cache_key = f"balance_{symbol}_{period}_{limit}"
        
        # Check cache
        if cache_key in self.financial_cache:
            cache_time, data = self.financial_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['financial']:
                return data
        
        # Make the request
        endpoint = f"balance-sheet-statement/{symbol}"
        params = {'limit': limit}
        if period == 'quarter':
            params['period'] = 'quarter'
        
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.financial_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []
    
    def get_cash_flow(self, symbol, period='annual', limit=4):
        """
        Get cash flow statement data.
        
        Args:
            symbol (str): Stock symbol
            period (str): 'annual' or 'quarter'
            limit (int): Number of periods to retrieve
            
        Returns:
            list: Cash flow statements or empty list on failure
        """
        # Generate cache key
        cache_key = f"cashflow_{symbol}_{period}_{limit}"
        
        # Check cache
        if cache_key in self.financial_cache:
            cache_time, data = self.financial_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['financial']:
                return data
        
        # Make the request
        endpoint = f"cash-flow-statement/{symbol}"
        params = {'limit': limit}
        if period == 'quarter':
            params['period'] = 'quarter'
        
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.financial_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []
    
    def get_key_metrics(self, symbol, period='annual', limit=1):
        """
        Get key metrics for a company.
        
        Args:
            symbol (str): Stock symbol
            period (str): 'annual' or 'quarter'
            limit (int): Number of periods to retrieve
            
        Returns:
            list: Key metrics or empty list on failure
        """
        # Generate cache key
        cache_key = f"metrics_{symbol}_{period}_{limit}"
        
        # Check cache
        if cache_key in self.financial_cache:
            cache_time, data = self.financial_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['financial']:
                return data
        
        # Make the request
        endpoint = f"key-metrics/{symbol}"
        params = {'limit': limit}
        if period == 'quarter':
            params['period'] = 'quarter'
        
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.financial_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []
    
    def get_news(self, tickers=None, limit=50):
        """
        Get news articles for specific tickers or general market news.
        
        Args:
            tickers (list): List of stock symbols or None for general news
            limit (int): Maximum number of articles to retrieve
            
        Returns:
            list: News articles or empty list on failure
        """
        # Generate cache key
        cache_key = f"news_{'_'.join(tickers) if tickers else 'general'}_{limit}"
        
        # Check cache
        if cache_key in self.news_cache:
            cache_time, data = self.news_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['news']:
                return data
        
        # Make the request
        if tickers and len(tickers) > 0:
            # Format tickers for the API (comma-separated)
            tickers_str = ','.join(tickers)
            endpoint = f"stock_news"
            params = {'tickers': tickers_str, 'limit': limit}
        else:
            # General market news
            endpoint = f"stock_news"
            params = {'limit': limit}
        
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and isinstance(response, list):
            # Format news articles to match expected format
            articles = []
            for article in response:
                # Transform to match expected format
                formatted_article = {
                    'title': article.get('title', ''),
                    'description': article.get('text', ''),
                    'publishedAt': article.get('publishedDate', ''),
                    'url': article.get('url', ''),
                    'source': {'name': article.get('site', '')}
                }
                articles.append(formatted_article)
            
            # Cache the result
            self.news_cache[cache_key] = (time.time(), articles)
            
            return articles
        else:
            # Return empty list on failure
            return []
    
    def get_exchange_rate(self, from_currency, to_currency):
        """
        Get exchange rate between two currencies.
        
        Args:
            from_currency (str): Base currency code (e.g., 'USD')
            to_currency (str): Target currency code (e.g., 'CAD')
            
        Returns:
            float: Exchange rate or None on failure
        """
        # Generate cache key
        cache_key = f"fx_{from_currency}_{to_currency}"
        
        # Check cache
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['price']:
                return data
        
        # Make the request
        endpoint = f"fx/{from_currency}{to_currency}"
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list) and len(response) > 0:
            rate = response[0].get('rate')
            if rate is not None:
                # Cache the result
                self.price_cache[cache_key] = (time.time(), rate)
                return rate
        
        # Return default on failure
        logger.warning(f"Failed to get exchange rate from {from_currency} to {to_currency}")
        return None
    
    def get_historical_exchange_rates(self, from_currency, to_currency, days=365):
        """
        Get historical exchange rates between two currencies.
        
        Args:
            from_currency (str): Base currency code (e.g., 'USD')
            to_currency (str): Target currency code (e.g., 'CAD')
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame: Historical exchange rates or empty DataFrame on failure
        """
        # Generate cache key
        cache_key = f"hist_fx_{from_currency}_{to_currency}_{days}"
        
        # Check cache
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['daily_price']:
                return data
        
        # Make the request
        endpoint = f"historical-price-full/{from_currency}{to_currency}"
        params = {'from': f"{days}days"}
        response = self._make_request(endpoint, params)
        
        # Process the response
        if response and 'historical' in response:
            # Convert to DataFrame
            df = pd.DataFrame(response['historical'])
            
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Sort by date (ascending)
            df = df.sort_index()
            
            # Rename columns for clarity
            df = df.rename(columns={
                'close': 'rate',
                'high': 'high_rate',
                'low': 'low_rate',
                'open': 'open_rate'
            })
            
            # Cache the result
            self.price_cache[cache_key] = (time.time(), df)
            
            return df
        else:
            # Return empty DataFrame on failure
            return pd.DataFrame()
    
    def get_mutual_fund_profile(self, symbol):
        """
        Get mutual fund profile information.
        
        Args:
            symbol (str): Mutual fund symbol
            
        Returns:
            dict: Mutual fund profile or empty dict on failure
        """
        # FMP's mutual fund data is accessed through the same endpoint as stocks
        return self.get_company_profile(symbol)
    
    def get_etf_profile(self, symbol):
        """
        Get ETF profile information.
        
        Args:
            symbol (str): ETF symbol
            
        Returns:
            dict: ETF profile or empty dict on failure
        """
        # FMP has a specific ETF holder endpoint
        
        # Generate cache key
        cache_key = f"etf_profile_{symbol}"
        
        # Check cache
        if cache_key in self.profile_cache:
            cache_time, data = self.profile_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['profile']:
                return data
        
        # First get the basic profile
        profile = self.get_company_profile(symbol)
        
        # Then get ETF holdings to enhance the profile
        endpoint = f"etf-holder/{symbol}"
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list):
            # Add holdings to the profile
            profile['holdings'] = response
            
            # Cache the result
            self.profile_cache[cache_key] = (time.time(), profile)
            
            return profile
        else:
            # Return basic profile if holdings aren't available
            return profile
    
    def get_sector_performance(self):
        """
        Get sector performance data.
        
        Returns:
            list: Sector performance data or empty list on failure
        """
        # Generate cache key
        cache_key = "sector_performance"
        
        # Check cache
        if cache_key in self.price_cache:
            cache_time, data = self.price_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['daily_price']:
                return data
        
        # Make the request
        endpoint = "sector-performance"
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.price_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []
    
    def get_economic_indicators(self, indicator=None):
        """
        Get economic indicators data.
        
        Args:
            indicator (str): Specific indicator or None for all indicators
            
        Returns:
            list: Economic indicators data or empty list on failure
        """
        # Generate cache key
        cache_key = f"economic_{indicator if indicator else 'all'}"
        
        # Check cache
        if cache_key in self.financial_cache:
            cache_time, data = self.financial_cache[cache_key]
            # Return cached data if within expiry period
            if time.time() - cache_time < self.cache_expiry['financial']:
                return data
        
        # Make the request
        if indicator:
            endpoint = f"economic/{indicator}"
        else:
            endpoint = "economic"
        
        response = self._make_request(endpoint)
        
        # Process the response
        if response and isinstance(response, list):
            # Cache the result
            self.financial_cache[cache_key] = (time.time(), response)
            
            return response
        else:
            # Return empty list on failure
            return []

# Initialize a single instance to be imported by other modules
fmp_api = FMPApi()