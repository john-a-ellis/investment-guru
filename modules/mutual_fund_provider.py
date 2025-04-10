# modules/mutual_fund_provider.py
"""
Provider for retrieving and managing Canadian mutual fund price data.
Manages manual price entries in the database. External fetching is handled by DataProvider.
"""
import pandas as pd
import logging
from datetime import datetime, timedelta

# Import database functions specific to mutual funds
from modules.mutual_fund_db import (
    add_mutual_fund_price,
    get_mutual_fund_prices,
    get_latest_mutual_fund_price
)

# No longer need fmp_api directly here
# from modules.fmp_api import fmp_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MutualFundProvider:
    """
    Provider for managing manual Canadian mutual fund price data in the database.
    External data fetching is now handled by the DataProvider.
    """

    def __init__(self):
        # No runtime cache needed here anymore, DataProvider handles caching.
        pass

    # Removed get_fund_data_fmp - DataProvider handles this.
    # Removed get_fund_data_morningstar - DataProvider would handle this if implemented.

    def add_manual_price(self, fund_code, date, price):
        """
        Add a manually entered price point for a mutual fund to the database.

        Args:
            fund_code (str): Fund code/symbol
            date (datetime or str): Date of the price point
            price (float): NAV price

        Returns:
            bool: Success status
        """
        # Add to database using the dedicated db function
        success = add_mutual_fund_price(fund_code, date, price)
        # No cache update needed here
        return success

    def get_historical_data(self, fund_code, start_date=None, end_date=None):
        """
        Get historical price data for a mutual fund *from the internal database*.
        This is used by DataProvider as a fallback or primary source if configured.

        Args:
            fund_code (str): Fund code/symbol
            start_date (datetime or str): Start date (optional)
            end_date (datetime or str): End date (optional)

        Returns:
            pd.DataFrame: Historical price data (Date index, 'Close' column) or empty DataFrame.
        """
        logger.info(f"MFP: Getting historical data for {fund_code} from internal DB.")

        # Get data from the specific mutual fund prices table
        price_data = get_mutual_fund_prices(fund_code, start_date, end_date)

        if price_data:
            df_data = []
            for item in price_data:
                try:
                    # Ensure date parsing is robust
                    date_obj = datetime.strptime(item['date'], '%Y-%m-%d')
                    df_data.append({
                        'Date': date_obj,
                        'Close': float(item['price']) # Standardize column name
                    })
                except ValueError:
                    logger.warning(f"MFP: Skipping invalid date format: {item['date']}")
                    continue
                except (TypeError, ValueError) as price_err:
                     logger.warning(f"MFP: Skipping invalid price format: {item['price']} ({price_err})")
                     continue

            if not df_data:
                 logger.warning(f"MFP: No valid data points found for {fund_code} in internal DB after parsing.")
                 return pd.DataFrame()

            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df = df.sort_index()

            # Ensure timezone-naive index to match DataProvider standard
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.info(f"MFP: Found {len(df)} records for {fund_code} in internal DB.")
            return df

        logger.warning(f"MFP: No data found for {fund_code} in internal DB.")
        return pd.DataFrame() # Return empty DataFrame if no data

    def get_current_price(self, fund_code):
        """
        Get the most recent price for a mutual fund *from the internal database*.
        Used by DataProvider to prioritize manual entries or as a fallback.

        Args:
            fund_code (str): Fund code/symbol

        Returns:
            float: Most recent price (or None if not available)
        """
        logger.info(f"MFP: Getting current price for {fund_code} from internal DB.")
        latest_price = get_latest_mutual_fund_price(fund_code)

        if latest_price is not None:
             logger.info(f"MFP: Found latest price {latest_price} for {fund_code} in internal DB.")
        else:
             logger.warning(f"MFP: No latest price found for {fund_code} in internal DB.")

        return latest_price

