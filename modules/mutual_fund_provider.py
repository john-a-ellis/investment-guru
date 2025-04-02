# modules/mutual_fund_provider.py
"""
Provider for retrieving and managing Canadian mutual fund price data.
Uses PostgreSQL database for storage instead of JSON files.
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from modules.mutual_fund_db import (
    add_mutual_fund_price, 
    get_mutual_fund_prices, 
    get_latest_mutual_fund_price
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MutualFundProvider:
    """
    Provider for retrieving and managing Canadian mutual fund price data.
    Falls back to manual entry if no data source is available.
    """
    
    def __init__(self):
        self.data_cache = {}  # Runtime cache to minimize database hits
    
    def get_fund_data_morningstar(self, fund_code):
        """
        Attempt to retrieve mutual fund data from Morningstar
        Note: This is a simplified example, in production might use a licensed API
        """
        try:
            # This is a placeholder - real implementation would use Morningstar API
            return None
        except Exception as e:
            logger.error(f"Error retrieving Morningstar data for {fund_code}: {e}")
            return None
    
    def get_fund_data_tmx(self, fund_code):
        """
        Attempt to retrieve mutual fund data from TMX Money
        Note: Web scraping in production would require proper permission
        """
        try:
            # This is a placeholder - real implementation would require proper scraping
            return None
        except Exception as e:
            logger.error(f"Error retrieving TMX data for {fund_code}: {e}")
            return None
    
    def add_manual_price(self, fund_code, date, price):
        """
        Add a manually entered price point for a mutual fund
        
        Args:
            fund_code (str): Fund code/symbol
            date (datetime or str): Date of the price point
            price (float): NAV price
        
        Returns:
            bool: Success status
        """
        # Add to database
        success = add_mutual_fund_price(fund_code, date, price)
        
        # If successful, update runtime cache
        if success:
            # Initialize fund in cache if needed
            if fund_code not in self.data_cache:
                self.data_cache[fund_code] = {}
            
            # Convert date to string for cache key if it's a datetime
            date_key = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            
            # Add to cache
            self.data_cache[fund_code][date_key] = float(price)
        
        return success
    
    def get_historical_data(self, fund_code, start_date=None, end_date=None):
        """
        Get historical price data for a mutual fund
        
        Args:
            fund_code (str): Fund code/symbol
            start_date (datetime): Start date (optional)
            end_date (datetime): End date (optional)
        
        Returns:
            DataFrame: Historical price data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Try to get data from external source first
        external_data = self.get_fund_data_morningstar(fund_code) or self.get_fund_data_tmx(fund_code)
        
        if external_data:
            # If we got external data, add it to our database
            for date, price in external_data.items():
                add_mutual_fund_price(fund_code, date, price)
        
        # Get data from database
        price_data = get_mutual_fund_prices(fund_code, start_date, end_date)
        
        if price_data:
            # Convert to DataFrame
            df_data = []
            for item in price_data:
                df_data.append({
                    'Date': datetime.strptime(item['date'], '%Y-%m-%d'),
                    'Close': float(item['price'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            return df.sort_index()
        
        # If we don't have data, return empty DataFrame
        return pd.DataFrame()
    
    def get_current_price(self, fund_code):
        """
        Get the most recent price for a mutual fund
        
        Args:
            fund_code (str): Fund code/symbol
        
        Returns:
            float: Most recent price (or None if not available)
        """
        # Check runtime cache first
        if fund_code in self.data_cache and self.data_cache[fund_code]:
            # Find the most recent date
            most_recent_date = max(self.data_cache[fund_code].keys())
            return self.data_cache[fund_code][most_recent_date]
        
        # Check database
        return get_latest_mutual_fund_price(fund_code)