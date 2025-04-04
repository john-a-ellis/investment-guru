# modules/data_collector.py
"""
Data collection module using Financial Modeling Prep API.
"""
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Import the FMP API module
from modules.fmp_api import fmp_api

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Responsible for collecting market data, financial statements, economic indicators,
    and news from Financial Modeling Prep API.
    """
    
    def __init__(self):
        # Initialize caches
        self.market_data_cache = {}
        self.financial_data_cache = {}
        self.news_cache = {}
        
    def get_market_data(self, symbols=None, timeframe="1mo"):
        """
        Retrieve market data for specified symbols over the given timeframe.
        
        Args:
            symbols (list): List of stock symbols
            timeframe (str): Time period ('1d', '1w', '1mo', '3mo', '6mo', '1y')
            
        Returns:
            dict: Dictionary of DataFrames with market data for each symbol
        """
        if symbols is None:
            # Load tracked assets
            from components.asset_tracker import load_tracked_assets
            tracked_assets = load_tracked_assets()
            symbols = list(tracked_assets.keys())
            
            # Add default indices if no user assets
            if not symbols:
                symbols = ['^GSPTSE', '^TXCX', 'XIU.TO', 'XIC.TO']
        
        # Check cache first
        cache_key = f"{'-'.join(symbols)}_{timeframe}"
        if cache_key in self.market_data_cache:
            cache_time, data = self.market_data_cache[cache_key]
            # Return cached data if less than 1 hour old
            if datetime.now() - cache_time < timedelta(hours=1):
                return data
        
        # Get fresh data
        data = {}
        
        for symbol in symbols:
            try:
                # Get historical price data from FMP API
                df = fmp_api.get_historical_price(symbol, period=timeframe)
                
                if not df.empty:
                    data[symbol] = df
                else:
                    logger.warning(f"No data found for symbol: {symbol}")
                    data[symbol] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol}: {e}")
                data[symbol] = pd.DataFrame()
        
        # Update cache
        self.market_data_cache[cache_key] = (datetime.now(), data)
        return data
        
    def get_company_financials(self, symbols):
        """
        Retrieve financial statements for specified companies using FMP.
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary of financial data for each symbol
        """
        financial_data = {}
        
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
                income_statement = fmp_api.get_income_statement(symbol, limit=4)
                
                # Get balance sheet
                balance_sheet = fmp_api.get_balance_sheet(symbol, limit=4)
                
                # Get cash flow
                cash_flow = fmp_api.get_cash_flow(symbol, limit=4)
                
                # Get key metrics
                key_metrics = fmp_api.get_key_metrics(symbol, limit=1)
                
                # Combine all data
                financial_data[symbol] = {
                    'income_statement': income_statement,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'key_metrics': key_metrics
                }
                
                # Update cache
                self.financial_data_cache[symbol] = (datetime.now(), financial_data[symbol])
            except Exception as e:
                logger.error(f"Error retrieving financial data for {symbol}: {e}")
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
        
        # Get news data from FMP
        try:
            # First try to get news for specific symbols related to keywords
            # For Canadian focus, we'll add some TSX symbols
            canadian_tickers = ['XIU.TO', '^GSPTSE', 'XIC.TO', 'RY.TO', 'TD.TO', 'ENB.TO']
            
            # Get general financial news
            articles = fmp_api.get_news(limit=50)
            
            # Get Canada-specific news using Canadian tickers
            canadian_articles = fmp_api.get_news(tickers=canadian_tickers, limit=20)
            
            # Combine and remove duplicates
            all_articles = articles + canadian_articles
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                title = article.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    # Filter based on keywords
                    if any(keyword.lower() in article.get('description', '').lower() or 
                           keyword.lower() in title.lower() 
                           for keyword in keywords):
                        unique_articles.append(article)
            
            # Limit to 20 articles
            unique_articles = unique_articles[:20]
            
            # Update cache
            self.news_cache[cache_key] = (datetime.now(), unique_articles)
            
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error retrieving news: {e}")
            return []
    
    def get_economic_indicators(self):
        """
        Retrieve key economic indicators like GDP, inflation, unemployment, etc. using FMP.
        
        Returns:
            dict: Dictionary of economic indicators
        """
        indicators = {}
        
        try:
            # Get broad economic indicators
            all_indicators = fmp_api.get_economic_indicators()
            
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
            logger.error(f"Error retrieving economic indicators: {e}")
            
        return indicators
    
    def get_exchange_rate(self, from_currency="USD", to_currency="CAD"):
        """
        Get the current exchange rate between two currencies.
        
        Args:
            from_currency (str): Base currency code
            to_currency (str): Target currency code
            
        Returns:
            float: Exchange rate
        """
        try:
            rate = fmp_api.get_exchange_rate(from_currency, to_currency)
            
            if rate is not None:
                return rate
            else:
                # Default rate if API call fails
                logger.warning(f"Failed to get exchange rate, using default")
                return 1.33 if from_currency == "USD" and to_currency == "CAD" else 1.0
        except Exception as e:
            logger.error(f"Error retrieving exchange rate: {e}")
            # Default rate as fallback
            return 1.33 if from_currency == "USD" and to_currency == "CAD" else 1.0
    
    def get_historical_exchange_rates(self, from_currency="USD", to_currency="CAD", days=365):
        """
        Get historical exchange rates between two currencies.
        
        Args:
            from_currency (str): Base currency code
            to_currency (str): Target currency code
            days (int): Number of days of historical data
            
        Returns:
            DataFrame: Historical exchange rates
        """
        try:
            df = fmp_api.get_historical_exchange_rates(from_currency, to_currency, days)
            
            if not df.empty:
                return df
            else:
                # Create a default series if API call fails
                logger.warning(f"Failed to get historical exchange rates, using default")
                date_range = pd.date_range(end=datetime.now(), periods=days)
                return pd.Series([1.33] * len(date_range), index=date_range)
        except Exception as e:
            logger.error(f"Error retrieving historical exchange rates: {e}")
            # Default series as fallback
            date_range = pd.date_range(end=datetime.now(), periods=days)
            return pd.Series([1.33] * len(date_range), index=date_range)