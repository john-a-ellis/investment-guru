# modules/mutual_fund_provider.py
import pandas as pd
import json
import os
from datetime import datetime, timedelta

class MutualFundProvider:
    """
    Provider for retrieving and managing Canadian mutual fund price data.
    Falls back to manual entry if no data source is available.
    """
    
    def __init__(self):
        self.data_cache = {}
        self.cache_file = 'data/mutual_fund_cache.json'
        self.load_cache()
    
    def load_cache(self):
        """Load cached mutual fund data"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Convert string dates back to datetime
                for fund_code, entries in cache_data.items():
                    self.data_cache[fund_code] = {}
                    for date_str, price in entries.items():
                        try:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            self.data_cache[fund_code][date] = float(price)
                        except:
                            continue
        except Exception as e:
            print(f"Error loading mutual fund cache: {e}")
            self.data_cache = {}
    
    def save_cache(self):
        """Save mutual fund data to cache"""
        try:
            # Create a serializable version of the cache
            serializable_cache = {}
            for fund_code, entries in self.data_cache.items():
                serializable_cache[fund_code] = {}
                for date, price in entries.items():
                    serializable_cache[fund_code][date.strftime('%Y-%m-%d')] = price
            
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=4)
        except Exception as e:
            print(f"Error saving mutual fund cache: {e}")
    
    def get_fund_data_morningstar(self, fund_code):
        """
        Attempt to retrieve mutual fund data from Morningstar
        Note: This is a simplified example, in production might use a licensed API
        """
        try:
            # This is a placeholder - real implementation would use Morningstar API
            return None
        except Exception as e:
            print(f"Error retrieving Morningstar data for {fund_code}: {e}")
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
            print(f"Error retrieving TMX data for {fund_code}: {e}")
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
        try:
            # Convert string date to datetime if needed
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d')
            
            # Initialize fund in cache if needed
            if fund_code not in self.data_cache:
                self.data_cache[fund_code] = {}
            
            # Add the price point
            self.data_cache[fund_code][date] = float(price)
            
            # Save the updated cache
            self.save_cache()
            return True
        except Exception as e:
            print(f"Error adding manual price for {fund_code}: {e}")
            return False
    
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
            # If we got external data, merge it with our cache
            for date, price in external_data.items():
                if fund_code not in self.data_cache:
                    self.data_cache[fund_code] = {}
                self.data_cache[fund_code][date] = price
            
            # Save the updated cache
            self.save_cache()
        
        # Use cached data (which now includes any new external data)
        if fund_code in self.data_cache:
            # Filter by date range
            filtered_data = {
                date: price for date, price in self.data_cache[fund_code].items()
                if start_date <= date <= end_date
            }
            
            # Convert to DataFrame
            if filtered_data:
                df = pd.DataFrame(
                    {'Close': [price for price in filtered_data.values()]},
                    index=[date for date in filtered_data.keys()]
                )
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
        if fund_code in self.data_cache:
            # Get the most recent date
            if self.data_cache[fund_code]:
                most_recent_date = max(self.data_cache[fund_code].keys())
                return self.data_cache[fund_code][most_recent_date]
        
        return None