# data_collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables from .env file
load_dotenv()

class DataCollector:
    """
    Responsible for collecting market data, financial statements, economic indicators,
    and news from Financial Modeling Prep API.
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        
        # Initialize caches
        self.market_data_cache = {}
        self.financial_data_cache = {}
        self.news_cache = {}
        
    def get_market_data(self, symbols=None, timeframe="1mo"):
        """
        Retrieve market data for specified symbols over the given timeframe.
        """
        if symbols is None:
            # Load tracked assets
            from components.asset_tracker import load_tracked_assets
            tracked_assets = load_tracked_assets()
            symbols = list(tracked_assets.keys())
            
            # Add default indices if no user assets
            if not symbols:
                symbols = ['^GSPTSE', '^TXCX', 'XIU.TO', 'XIC.TO']
        
        # Convert timeframe to days for FMP API
        days = 30  # default for 1mo
        if timeframe == "1d":
            days = 1
        elif timeframe == "1w":
            days = 7
        elif timeframe == "3mo":
            days = 90
        elif timeframe == "6mo":
            days = 180
        elif timeframe == "1y":
            days = 365
            
        # Check cache first
        cache_key = f"{'-'.join(symbols)}_{timeframe}"
        if cache_key in self.market_data_cache:
            cache_time, data = self.market_data_cache[cache_key]
            # Return cached data if less than 1 hour old
            if datetime.now() - cache_time < timedelta(hours=1):
                return data
        
        # Get fresh data
        data = {}
        base_url = "https://financialmodelingprep.com/api/v3"
        
        # Create a custom session for Yahoo Finance API calls
        yf_session = create_yf_session()
        
        for symbol in symbols:
            try:
                # Handle indices differently
                if symbol.startswith('^'):
                    # For indices, use the historical price endpoint
                    url = f"{base_url}/historical-price-full/{symbol}?from={days}days&apikey={self.fmp_api_key}"
                else:
                    # For stocks and ETFs
                    url = f"{base_url}/historical-price-full/{symbol}?from={days}days&apikey={self.fmp_api_key}"
                
                # Use the custom session
                with yf_session as session:
                    response = session.get(url)
                    if response.status_code == 200:
                        json_data = response.json()
                        if 'historical' in json_data:
                            # Convert to dataframe
                            df = pd.DataFrame(json_data['historical'])
                            # Convert date strings to datetime
                            df['date'] = pd.to_datetime(df['date'])
                            # Set date as index
                            df = df.set_index('date')
                            # Sort by date
                            df = df.sort_index()
                            # Rename columns to match yfinance format
                            df = df.rename(columns={
                                'open': 'Open',
                                'high': 'High',
                                'low': 'Low',
                                'close': 'Close',
                                'volume': 'Volume'
                            })
                            data[symbol] = df
                        else:
                            data[symbol] = pd.DataFrame()
                    else:
                        data[symbol] = pd.DataFrame()
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")
                data[symbol] = pd.DataFrame()
        
        # Update cache
        self.market_data_cache[cache_key] = (datetime.now(), data)
        return data
        
    def get_company_financials(self, symbols):
        """
        Retrieve financial statements for specified companies using FMP.
        """
        financial_data = {}
        base_url = "https://financialmodelingprep.com/api/v3"
        
        for symbol in symbols:
            try:
                # Check cache first
                if symbol in self.financial_data_cache:
                    cache_time, data = self.financial_data_cache[symbol]
                    # Return cached data if less than 1 day old
                    if datetime.now() - cache_time < timedelta(days=1):
                        financial_data[symbol] = data
                        continue
                
                # Get income statement
                income_url = f"{base_url}/income-statement/{symbol}?limit=4&apikey={self.fmp_api_key}"
                income_response = requests.get(income_url)
                
                # Get balance sheet
                balance_url = f"{base_url}/balance-sheet-statement/{symbol}?limit=4&apikey={self.fmp_api_key}"
                balance_response = requests.get(balance_url)
                
                # Get cash flow
                cashflow_url = f"{base_url}/cash-flow-statement/{symbol}?limit=4&apikey={self.fmp_api_key}"
                cashflow_response = requests.get(cashflow_url)
                
                # Get key metrics
                metrics_url = f"{base_url}/key-metrics/{symbol}?limit=1&apikey={self.fmp_api_key}"
                metrics_response = requests.get(metrics_url)
                
                # Combine all data
                financial_data[symbol] = {
                    'income_statement': income_response.json() if income_response.status_code == 200 else [],
                    'balance_sheet': balance_response.json() if balance_response.status_code == 200 else [],
                    'cash_flow': cashflow_response.json() if cashflow_response.status_code == 200 else [],
                    'key_metrics': metrics_response.json() if metrics_response.status_code == 200 else []
                }
                
                # Update cache
                self.financial_data_cache[symbol] = (datetime.now(), financial_data[symbol])
            except Exception as e:
                print(f"Error retrieving financial data for {symbol}: {e}")
                financial_data[symbol] = {}
                
        return financial_data
    
    def get_latest_news(self, keywords=None, sectors=None):
        """
        Retrieve latest financial news based on keywords or sectors using FMP.
        
        Args:
            keywords (list): List of keywords to filter news
            sectors (list): List of sectors to filter news
            
        Returns:
            list: List of news articles
        """
        if keywords is None:
            keywords = ["finance", "stocks", "economy", "market", "canada", "tsx"]
        
        # Create cache key
        cache_key = f"news_{'_'.join(keywords)}"
        if cache_key in self.news_cache:
            cache_time, data = self.news_cache[cache_key]
            # Return cached data if less than 3 hours old
            if datetime.now() - cache_time < timedelta(hours=3):
                return data
        
        # Using FMP for news data
        base_url = "https://financialmodelingprep.com/api/v3/stock_news"
        
        articles = []
        try:
            # Get general stock news
            params = {
                'limit': 50,  # Get more articles to have enough after filtering
                'apikey': self.fmp_api_key
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                all_articles = response.json()
                
                # Filter articles based on keywords
                for article in all_articles:
                    if any(keyword.lower() in article.get('text', '').lower() or 
                        keyword.lower() in article.get('title', '').lower() 
                        for keyword in keywords):
                        # Transform to match expected format in news_analyzer
                        transformed_article = {
                            'title': article.get('title', ''),
                            'description': article.get('text', ''),
                            'publishedAt': article.get('publishedDate', ''),
                            'url': article.get('url', ''),
                            'source': {'name': article.get('site', '')}
                        }
                        articles.append(transformed_article)
            
            # If we want more Canadian-specific news, we can also get news for Canadian indices
            canadian_tickers = ['XIU.TO', '^GSPTSE']  # iShares S&P/TSX 60 ETF and TSX Composite Index
            for ticker in canadian_tickers:
                ticker_url = f"https://financialmodelingprep.com/api/v3/stock_news/{ticker}?limit=20&apikey={self.fmp_api_key}"
                ticker_response = requests.get(ticker_url)
                if ticker_response.status_code == 200:
                    ticker_articles = ticker_response.json()
                    for article in ticker_articles:
                        # Transform to match expected format in news_analyzer
                        transformed_article = {
                            'title': article.get('title', ''),
                            'description': article.get('text', ''),
                            'publishedAt': article.get('publishedDate', ''),
                            'url': article.get('url', ''),
                            'source': {'name': article.get('site', '')}
                        }
                        # Check if this is a duplicate
                        if not any(existing['title'] == transformed_article['title'] for existing in articles):
                            articles.append(transformed_article)
            
        except Exception as e:
            print(f"Error retrieving news: {e}")
        
        # Limit to 20 articles
        articles = articles[:20]
        
        # Update cache
        self.news_cache[cache_key] = (datetime.now(), articles)
        return articles
    
    def get_economic_indicators(self):
        """
        Retrieve key economic indicators like GDP, inflation, unemployment, etc. using FMP.
        
        Returns:
            dict: Dictionary of economic indicators
        """
        indicators = {}
        base_url = "https://financialmodelingprep.com/api/v4"
        
        try:
            # Get economic indicators
            indicators_url = f"{base_url}/economic?apikey={self.fmp_api_key}"
            response = requests.get(indicators_url)
            
            if response.status_code == 200:
                all_indicators = response.json()
                
                # Process indicators into categories
                for indicator in all_indicators:
                    name = indicator.get('name', '')
                    
                    # Categorize indicators
                    if 'gdp' in name.lower():
                        category = 'gdp'
                    elif 'inflation' in name.lower() or 'cpi' in name.lower():
                        category = 'inflation'
                    elif 'unemployment' in name.lower():
                        category = 'unemployment'
                    elif 'interest rate' in name.lower():
                        category = 'interest_rates'
                    elif 'housing' in name.lower() or 'home' in name.lower():
                        category = 'housing'
                    else:
                        category = 'other'
                    
                    # Add to appropriate category
                    if category not in indicators:
                        indicators[category] = []
                    
                    indicators[category].append(indicator)
                    
        except Exception as e:
            print(f"Error retrieving economic indicators: {e}")
            
        return indicators
    

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
        pool_connections=30,  # Increased from 20
        pool_maxsize=30  # Increased from 20
    )
    
    # Mount for both http and https
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session