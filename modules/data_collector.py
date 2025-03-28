# modules/data_collector.py
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class DataCollector:
    """
    Responsible for collecting market data, financial statements, and economic indicators
    from various APIs and data sources.
    """
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        
        # Cache to store data and reduce API calls
        self.market_data_cache = {}
        self.financial_data_cache = {}
        self.news_cache = {}
        
    def get_market_data(self, symbols=None, timeframe="1mo"):
        """
        Retrieve market data for specified symbols over the given timeframe.
        
        Args:
            symbols (list): List of ticker symbols to retrieve data for. If None, uses default indices.
            timeframe (str): Timeframe for data retrieval (e.g., '1d', '1mo', '1y')
            
        Returns:
            dict: Dictionary of dataframes containing market data for each symbol
        """
        if symbols is None:
            symbols = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225']  # Default indices
            
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
                ticker = yf.Ticker(symbol)
                data[symbol] = ticker.history(period=timeframe)
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")
                data[symbol] = pd.DataFrame()  # Empty dataframe for failed retrieval
        
        # Update cache
        self.market_data_cache[cache_key] = (datetime.now(), data)
        return data
    
    def get_company_financials(self, symbols):
        """
        Retrieve financial statements for specified companies.
        
        Args:
            symbols (list): List of ticker symbols
            
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
                
                # Get data using yfinance
                ticker = yf.Ticker(symbol)
                financial_data[symbol] = {
                    'income_statement': ticker.financials,
                    'balance_sheet': ticker.balance_sheet,
                    'cash_flow': ticker.cashflow,
                    'key_stats': ticker.info
                }
                
                # Update cache
                self.financial_data_cache[symbol] = (datetime.now(), financial_data[symbol])
            except Exception as e:
                print(f"Error retrieving financial data for {symbol}: {e}")
                financial_data[symbol] = {}
                
        return financial_data
    
    def get_economic_indicators(self):
        """
        Retrieve key economic indicators like GDP, inflation, unemployment, etc.
        
        Returns:
            dict: Dictionary of economic indicators
        """
        # Using Alpha Vantage for economic indicators
        indicators = {}
        
        try:
            # Get real GDP
            gdp_url = f"https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={self.alpha_vantage_key}"
            gdp_response = requests.get(gdp_url)
            if gdp_response.status_code == 200:
                indicators['gdp'] = gdp_response.json()
            
            # Get inflation data (CPI)
            cpi_url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={self.alpha_vantage_key}"
            cpi_response = requests.get(cpi_url)
            if cpi_response.status_code == 200:
                indicators['inflation'] = cpi_response.json()
            
            # Get unemployment rate
            unemp_url = f"https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={self.alpha_vantage_key}"
            unemp_response = requests.get(unemp_url)
            if unemp_response.status_code == 200:
                indicators['unemployment'] = unemp_response.json()
                
        except Exception as e:
            print(f"Error retrieving economic indicators: {e}")
            
        return indicators
    
    def get_latest_news(self, keywords=None, sectors=None):
        """
        Retrieve latest financial news based on keywords or sectors.
        
        Args:
            keywords (list): List of keywords to filter news
            sectors (list): List of sectors to filter news
            
        Returns:
            list: List of news articles
        """
        if keywords is None:
            keywords = ["finance", "stocks", "economy", "market"]
        
        # Create cache key
        cache_key = f"news_{'_'.join(keywords)}"
        if cache_key in self.news_cache:
            cache_time, data = self.news_cache[cache_key]
            # Return cached data if less than 3 hours old
            if datetime.now() - cache_time < timedelta(hours=3):
                return data
        
        # Using NewsAPI for news data
        news_url = f"https://newsapi.org/v2/everything"
        
        articles = []
        try:
            for keyword in keywords:
                params = {
                    'q': keyword,
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10
                }
                
                response = requests.get(news_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'articles' in data:
                        articles.extend(data['articles'])
        except Exception as e:
            print(f"Error retrieving news: {e}")
        
        # Update cache
        self.news_cache[cache_key] = (datetime.now(), articles)
        return articles