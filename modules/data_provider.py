# modules/data_provider.py
"""
Data Abstraction Layer (DAL) for fetching financial data.
Centralizes access to external APIs (FMP, YFinance) and handles caching,
fallback logic, and data standardization.
"""
import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from cachetools import TTLCache, cached

# Import the FMP API module (will only be called from here)
from modules.fmp_api import fmp_api
# Import YFinance utilities (optional, can call yf directly too)
from modules.yf_utils import get_yf_session, get_ticker_history
# Import Mutual Fund Provider for fallback
from modules.mutual_fund_provider import MutualFundProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Caching Configuration ---
# Cache historical price data for 1 hour
hist_price_cache = TTLCache(maxsize=100, ttl=3600)
# Cache mutual fund historical data for 12 hours
mf_hist_cache = TTLCache(maxsize=50, ttl=43200)
# Cache quotes for 5 minutes
quote_cache = TTLCache(maxsize=200, ttl=300)
# Cache profile data for 24 hours
profile_cache = TTLCache(maxsize=100, ttl=86400)
etf_profile_cache = TTLCache(maxsize=100, ttl=86400)
mf_profile_cache = TTLCache(maxsize=100, ttl=86400)
# Cache financials for 24 hours
financial_cache = TTLCache(maxsize=100, ttl=86400)
# Cache news for 30 minutes
news_cache = TTLCache(maxsize=50, ttl=1800)
# Cache exchange rates for 1 hour
fx_cache = TTLCache(maxsize=20, ttl=3600)
# Cache economic data for 6 hours
econ_cache = TTLCache(maxsize=10, ttl=21600)
# Cache sector performance for 6 hours
sector_perf_cache = TTLCache(maxsize=10, ttl=21600)


class DataProvider:
    """
    Provides a unified interface to fetch financial data, handling
    primary (FMP) and fallback (YFinance/DB) sources, caching, and standardization.
    """
    def __init__(self):
        # Initialize Mutual Fund Provider for fallback
        self.mutual_fund_provider = MutualFundProvider()

    def _standardize_dataframe(self, df):
        """Ensure DataFrame has standard columns and timezone-naive index."""
        if df is None or df.empty:
            return pd.DataFrame()

        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Failed to convert index to DatetimeIndex: {e}")
                return pd.DataFrame() # Return empty if conversion fails

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Standard column names (lowercase for consistency)
        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Adj Close': 'adj_close', 'Volume': 'volume',
            # Add other potential variations if needed
            'adjClose': 'adj_close',
        }
        df = df.rename(columns=lambda c: rename_map.get(c, c.lower()))

        # Ensure essential columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Add missing columns with NA, except for volume which can be 0
                df[col] = 0 if col == 'volume' else pd.NA

        # Select and reorder standard columns
        standard_cols = [col for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume'] if col in df.columns]
        df = df[standard_cols]

        return df

    def _determine_currency(self, symbol):
        """Determine currency based on symbol convention."""
        if symbol.endswith((".TO", ".V")) or "-CAD" in symbol or symbol.startswith("MAW"): # Crude check for known CAD
             return "CAD"
        # Add more checks if needed (e.g., .L for LSE in GBP)
        # Default to USD if not obviously Canadian
        return "USD"

    @cached(hist_price_cache)
    def get_historical_price(self, symbol, start_date=None, end_date=None, period=None):
        """
        Get standardized historical price data for a symbol (Stock/ETF).
        Tries FMP first, then YFinance. Handles caching.

        Args:
            symbol (str): Stock/ETF symbol.
            start_date (str or datetime): Start date.
            end_date (str or datetime): End date.
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y').

        Returns:
            pd.DataFrame: Standardized historical price data or empty DataFrame on failure.
        """
        logger.info(f"DAL: Requesting historical price for {symbol} (Period: {period}, Start: {start_date}, End: {end_date})")
        df = None

        # --- Try FMP First ---
        try:
            # Note: fmp_api.get_historical_price itself might have a fallback,
            # but we control the primary attempt here.
            # We pass dates/period directly to the underlying FMP call.
            fmp_df = fmp_api.get_historical_price(symbol, period=period, start_date=start_date, end_date=end_date)
            if fmp_df is not None and not fmp_df.empty:
                logger.info(f"DAL: Got data for {symbol} from FMP.")
                df = self._standardize_dataframe(fmp_df)
                if not df.empty:
                    return df # Return standardized FMP data
            else:
                 logger.warning(f"DAL: No data from FMP for {symbol}.")
        except Exception as e:
            logger.error(f"DAL: Error fetching {symbol} from FMP: {e}")

        # --- Fallback to YFinance ---
        if df is None or df.empty:
            logger.warning(f"DAL: Falling back to YFinance for {symbol}.")
            try:
                # Use yfinance directly or through yf_utils
                yf_df = get_ticker_history(symbol, start=start_date, end=end_date, period=period)
                if yf_df is not None and not yf_df.empty:
                    logger.info(f"DAL: Got data for {symbol} from YFinance.")
                    df = self._standardize_dataframe(yf_df)
                    if not df.empty:
                        return df # Return standardized YFinance data
                else:
                    logger.warning(f"DAL: No data from YFinance for {symbol}.")
            except Exception as e:
                logger.error(f"DAL: Error fetching {symbol} from YFinance: {e}")

        # --- Final Return ---
        if df is not None and not df.empty:
             logger.info(f"DAL: Successfully provided historical data for {symbol}.")
             return df
        else:
             logger.error(f"DAL: Failed to get historical data for {symbol} from all sources.")
             return pd.DataFrame() # Return empty DataFrame if all sources fail

    @cached(mf_hist_cache)
    def get_mutual_fund_data(self, symbol, start_date=None, end_date=None, period=None):
        """
        Get standardized historical price data for a Mutual Fund symbol.
        Tries FMP first, then falls back to the internal MutualFundProvider (DB).

        Args:
            symbol (str): Mutual Fund symbol.
            start_date (str or datetime): Start date.
            end_date (str or datetime): End date.
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '5y').

        Returns:
            pd.DataFrame: Standardized historical price data or empty DataFrame on failure.
        """
        logger.info(f"DAL: Requesting historical data for Mutual Fund {symbol} (Period: {period}, Start: {start_date}, End: {end_date})")
        df = None

        # --- Try FMP First ---
        try:
            # Use the general historical price function which tries FMP/YF
            fmp_df = self.get_historical_price(symbol, period=period, start_date=start_date, end_date=end_date)
            if fmp_df is not None and not fmp_df.empty:
                logger.info(f"DAL: Got data for Mutual Fund {symbol} from FMP/YF.")
                df = self._standardize_dataframe(fmp_df) # Standardize again just in case
                if not df.empty:
                    return df # Return standardized FMP/YF data
            else:
                 logger.warning(f"DAL: No data from FMP/YF for Mutual Fund {symbol}.")
        except Exception as e:
            logger.error(f"DAL: Error fetching Mutual Fund {symbol} from FMP/YF: {e}")

        # --- Fallback to Internal MutualFundProvider (Database) ---
        if df is None or df.empty:
            logger.warning(f"DAL: Falling back to internal DB for Mutual Fund {symbol}.")
            try:
                # Use the MutualFundProvider instance
                db_df = self.mutual_fund_provider.get_historical_data(symbol, start_date=start_date, end_date=end_date)
                if db_df is not None and not db_df.empty:
                    logger.info(f"DAL: Got data for Mutual Fund {symbol} from internal DB.")
                    df = self._standardize_dataframe(db_df) # Standardize DB data
                    if not df.empty:
                        return df # Return standardized DB data
                else:
                    logger.warning(f"DAL: No data from internal DB for Mutual Fund {symbol}.")
            except Exception as e:
                logger.error(f"DAL: Error fetching Mutual Fund {symbol} from internal DB: {e}")

        # --- Final Return ---
        if df is not None and not df.empty:
             logger.info(f"DAL: Successfully provided historical data for Mutual Fund {symbol}.")
             return df
        else:
             logger.error(f"DAL: Failed to get historical data for Mutual Fund {symbol} from all sources.")
             return pd.DataFrame() # Return empty DataFrame if all sources fail


    @cached(quote_cache)
    def get_current_quote(self, symbol, asset_type="stock"):
        """
        Get the latest quote (price and currency) for a symbol.
        Handles different asset types (stock, etf, mutual_fund).
        Tries FMP first, then YFinance, then internal DB for mutual funds.

        Args:
            symbol (str): Asset symbol.
            asset_type (str): Type of asset ('stock', 'etf', 'mutual_fund').

        Returns:
            dict: {'price': float, 'currency': str} or None on failure.
        """
        logger.info(f"DAL: Requesting current quote for {symbol} (Type: {asset_type})")
        quote_data = None

        # --- Handle Mutual Funds Separately (Try DB first for latest manual entry) ---
        if asset_type == "mutual_fund":
            try:
                db_price = self.mutual_fund_provider.get_current_price(symbol)
                if db_price is not None:
                    price = float(db_price)
                    currency = "CAD" # Assume CAD for mutual funds from DB
                    quote_data = {'price': price, 'currency': currency}
                    logger.info(f"DAL: Got quote for Mutual Fund {symbol} from internal DB: {price} {currency}")
                    return quote_data
                else:
                    logger.warning(f"DAL: No quote for Mutual Fund {symbol} in internal DB, trying FMP/YF.")
            except Exception as e_db:
                 logger.error(f"DAL: Error fetching Mutual Fund {symbol} quote from internal DB: {e_db}")

        # --- Try FMP First (for all types, including MF fallback) ---
        try:
            fmp_quote = fmp_api.get_quote(symbol)
            if fmp_quote and 'price' in fmp_quote and fmp_quote['price'] is not None:
                 price = float(fmp_quote['price'])
                 # Determine currency based on symbol (crude)
                 currency = self._determine_currency(symbol)
                 quote_data = {'price': price, 'currency': currency}
                 logger.info(f"DAL: Got quote for {symbol} from FMP: {price} {currency}")
                 return quote_data
            else:
                 logger.warning(f"DAL: No valid quote from FMP for {symbol}.")
        except Exception as e:
            logger.error(f"DAL: Error fetching quote for {symbol} from FMP: {e}")

        # --- Fallback to YFinance (for all types, including MF fallback) ---
        if quote_data is None:
            logger.warning(f"DAL: Falling back to YFinance for {symbol} quote.")
            try:
                # Get 1 day history to find the latest close
                yf_hist = get_ticker_history(symbol, period="1d")
                if yf_hist is not None and not yf_hist.empty and 'Close' in yf_hist.columns:
                    price = float(yf_hist['Close'].iloc[-1])
                    # Try to get currency info from yfinance ticker info
                    try:
                        ticker_info = yf.Ticker(symbol, session=get_yf_session()).info
                        currency = ticker_info.get('currency', self._determine_currency(symbol)).upper()
                    except:
                        currency = self._determine_currency(symbol) # Fallback determination

                    quote_data = {'price': price, 'currency': currency}
                    logger.info(f"DAL: Got quote for {symbol} from YFinance: {price} {currency}")
                    return quote_data
                else:
                    logger.warning(f"DAL: No quote data from YFinance for {symbol}.")
            except Exception as e:
                logger.error(f"DAL: Error fetching quote for {symbol} from YFinance: {e}")

        # --- Final Return ---
        if quote_data:
             return quote_data
        else:
             logger.error(f"DAL: Failed to get quote for {symbol} from all sources.")
             return None

    @cached(profile_cache)
    def get_company_profile(self, symbol):
        logger.info(f"DAL: Requesting company profile for {symbol}")
        try:
            profile = fmp_api.get_company_profile(symbol)
            if profile:
                logger.info(f"DAL: Got profile for {symbol} from FMP.")
                return profile
            else:
                # Add YFinance fallback if desired
                logger.warning(f"DAL: No profile from FMP for {symbol}.")
                return {}
        except Exception as e:
            logger.error(f"DAL: Error fetching profile for {symbol}: {e}")
            return {}

    @cached(etf_profile_cache)
    def get_etf_profile(self, symbol):
        logger.info(f"DAL: Requesting ETF profile for {symbol}")
        try:
            profile = fmp_api.get_etf_profile(symbol) # FMP API handles this
            if profile:
                logger.info(f"DAL: Got ETF profile for {symbol} from FMP.")
                return profile
            else:
                logger.warning(f"DAL: No ETF profile from FMP for {symbol}.")
                return {}
        except Exception as e:
            logger.error(f"DAL: Error fetching ETF profile for {symbol}: {e}")
            return {}

    @cached(mf_profile_cache)
    def get_mutual_fund_profile(self, symbol):
        logger.info(f"DAL: Requesting Mutual Fund profile for {symbol}")
        try:
            # FMP uses the same profile endpoint for MFs
            profile = fmp_api.get_mutual_fund_profile(symbol)
            if profile:
                logger.info(f"DAL: Got Mutual Fund profile for {symbol} from FMP.")
                return profile
            else:
                logger.warning(f"DAL: No Mutual Fund profile from FMP for {symbol}.")
                return {}
        except Exception as e:
            logger.error(f"DAL: Error fetching Mutual Fund profile for {symbol}: {e}")
            return {}

    @cached(financial_cache)
    def get_financial_statement(self, symbol, statement_type, period='annual', limit=4):
        logger.info(f"DAL: Requesting {statement_type} for {symbol} (Period: {period}, Limit: {limit})")
        try:
            data = []
            if statement_type == 'income':
                data = fmp_api.get_income_statement(symbol, period=period, limit=limit)
            elif statement_type == 'balance':
                data = fmp_api.get_balance_sheet(symbol, period=period, limit=limit)
            elif statement_type == 'cashflow':
                data = fmp_api.get_cash_flow(symbol, period=period, limit=limit)
            else:
                logger.error(f"DAL: Unknown statement type: {statement_type}")
                return []

            if data:
                logger.info(f"DAL: Got {statement_type} for {symbol} from FMP.")
                return data
            else:
                logger.warning(f"DAL: No {statement_type} from FMP for {symbol}.")
                return []
        except Exception as e:
            logger.error(f"DAL: Error fetching {statement_type} for {symbol}: {e}")
            return []

    @cached(financial_cache)
    def get_key_metrics(self, symbol, period='annual', limit=1):
        logger.info(f"DAL: Requesting key metrics for {symbol} (Period: {period}, Limit: {limit})")
        try:
            metrics = fmp_api.get_key_metrics(symbol, period=period, limit=limit)
            if metrics:
                logger.info(f"DAL: Got key metrics for {symbol} from FMP.")
                return metrics
            else:
                logger.warning(f"DAL: No key metrics from FMP for {symbol}.")
                return []
        except Exception as e:
            logger.error(f"DAL: Error fetching key metrics for {symbol}: {e}")
            return []

    @cached(news_cache)
    def get_news(self, tickers=None, limit=50):
        logger.info(f"DAL: Requesting news (Tickers: {tickers}, Limit: {limit})")
        try:
            news = fmp_api.get_news(tickers=tickers, limit=limit)
            if news:
                logger.info(f"DAL: Got {len(news)} news articles from FMP.")
                return news
            else:
                logger.warning(f"DAL: No news from FMP.")
                return []
        except Exception as e:
            logger.error(f"DAL: Error fetching news: {e}")
            return []

    @cached(fx_cache)
    def get_exchange_rate(self, from_currency, to_currency):
        logger.info(f"DAL: Requesting exchange rate {from_currency} to {to_currency}")
        rate = None
        # --- Try FMP First ---
        try:
            fmp_rate = fmp_api.get_exchange_rate(from_currency, to_currency)
            if fmp_rate is not None:
                logger.info(f"DAL: Got exchange rate from FMP: {fmp_rate}")
                rate = fmp_rate
                return rate
            else:
                logger.warning(f"DAL: No exchange rate from FMP for {from_currency}{to_currency}.")
        except Exception as e:
            logger.error(f"DAL: Error fetching exchange rate for {from_currency}{to_currency} from FMP: {e}")

        # --- Fallback to YFinance ---
        if rate is None:
            logger.warning(f"DAL: Falling back to YFinance for exchange rate {from_currency}{to_currency}.")
            try:
                yf_ticker_str = f"{from_currency}{to_currency}=X"
                yf_ticker = yf.Ticker(yf_ticker_str, session=get_yf_session())
                yf_hist = yf_ticker.history(period="1d")
                if yf_hist is not None and not yf_hist.empty and 'Close' in yf_hist.columns:
                    yf_rate = float(yf_hist['Close'].iloc[-1])
                    logger.info(f"DAL: Got exchange rate from YFinance: {yf_rate}")
                    rate = yf_rate
                    return rate
                else:
                    logger.warning(f"DAL: No exchange rate from YFinance.")
            except Exception as e_yf:
                logger.error(f"DAL: Error fetching exchange rate from YFinance: {e_yf}")

        # --- Final Fallback (Default) ---
        if rate is None:
            default_rate = 1.33 if from_currency == "USD" and to_currency == "CAD" else 1.0
            logger.warning(f"DAL: Using default rate: {default_rate}")
            rate = default_rate

        return rate

    @cached(fx_cache) # Cache historical FX for longer
    def get_historical_exchange_rates(self, from_currency, to_currency, start_date=None, end_date=None, days=365):
        """Gets historical exchange rates, ensuring timezone-naive output."""
        logger.info(f"DAL: Requesting historical FX {from_currency} to {to_currency} (Days: {days})")
        df = None
        days_to_use = days

        # Determine days_to_use based on start/end dates if provided
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                days_calc = (end - start).days + 1 # Include end date
                days_to_use = max(days, days_calc) # Use the larger duration
            except Exception as date_err:
                 logger.error(f"DAL: Error parsing dates for historical FX: {date_err}. Using default days: {days}")
                 days_to_use = days
        elif start_date or end_date:
            # If only one date is provided, use the default 'days' for fetching, then filter
            days_to_use = days

        # --- Try FMP First ---
        try:
            fmp_df = fmp_api.get_historical_exchange_rates(from_currency, to_currency, days=days_to_use)
            if fmp_df is not None and not fmp_df.empty:
                logger.info(f"DAL: Got historical FX from FMP.")
                df = self._standardize_dataframe(fmp_df) # Standardize applies timezone fix
                if 'close' in df.columns: # FMP uses 'close'
                    df = df.rename(columns={'close': 'rate'})
                if not df.empty:
                    # Filter by date range if provided
                    if start_date: df = df.loc[start_date:]
                    if end_date: df = df.loc[:end_date]
                    if not df.empty:
                        return df[['rate']] # Return only the rate column
            else:
                logger.warning(f"DAL: No historical FX from FMP.")
        except Exception as e:
            logger.error(f"DAL: Error fetching historical FX from FMP: {e}")

        # --- Fallback to YFinance ---
        if df is None or df.empty:
            logger.warning(f"DAL: Falling back to YFinance for historical FX {from_currency}{to_currency}.")
            try:
                yf_ticker_str = f"{from_currency}{to_currency}=X"
                yf_ticker = yf.Ticker(yf_ticker_str, session=get_yf_session())
                # Fetch slightly more data if using days_to_use, then filter
                yf_df = yf_ticker.history(start=start_date, end=end_date, period=f"{days_to_use+5}d") # Add buffer
                if yf_df is not None and not yf_df.empty:
                    logger.info(f"DAL: Got historical FX from YFinance.")
                    df = self._standardize_dataframe(yf_df)
                    if 'close' in df.columns: # YFinance uses 'Close' -> 'close'
                        df = df.rename(columns={'close': 'rate'})
                    if not df.empty:
                         # Filter again precisely by requested dates
                        if start_date: df = df.loc[start_date:]
                        if end_date: df = df.loc[:end_date]
                        if not df.empty:
                            return df[['rate']]
                else:
                    logger.warning(f"DAL: No historical FX from YFinance.")
            except Exception as e_yf:
                logger.error(f"DAL: Error fetching historical FX from YFinance: {e_yf}")

        # --- Final Return (Fallback/Empty) ---
        if df is not None and not df.empty:
             logger.info(f"DAL: Successfully provided historical FX for {from_currency}{to_currency}.")
             return df[['rate']]
        else:
            logger.error(f"DAL: Failed to get historical FX for {from_currency}{to_currency}.")
            # Return empty DataFrame
            return pd.DataFrame(columns=['rate'])

    @cached(econ_cache)
    def get_economic_indicators(self, indicator=None):
        """
        Retrieve key economic indicators like GDP, inflation, unemployment, etc. using FMP.

        Args:
            indicator (str): Specific indicator name (optional).

        Returns:
            list or dict: List of indicators or specific indicator data.
        """
        logger.info(f"DAL: Requesting economic indicators (Indicator: {indicator})")
        try:
            data = fmp_api.get_economic_indicators(indicator=indicator)
            if data:
                logger.info(f"DAL: Got economic indicators from FMP.")
                return data
            else:
                logger.warning(f"DAL: No economic indicators from FMP.")
                return [] if indicator is None else {}
        except Exception as e:
            logger.error(f"DAL: Error retrieving economic indicators: {e}")
            return [] if indicator is None else {}

    @cached(sector_perf_cache)
    def get_sector_performance(self):
        """
        Get sector performance data using FMP.

        Returns:
            list: Sector performance data or empty list on failure.
        """
        logger.info("DAL: Requesting sector performance data.")
        try:
            data = fmp_api.get_sector_performance()
            if data:
                logger.info(f"DAL: Got sector performance data from FMP.")
                return data
            else:
                logger.warning(f"DAL: No sector performance data from FMP.")
                return []
        except Exception as e:
            logger.error(f"DAL: Error retrieving sector performance data: {e}")
            return []

# --- Instantiate the DataProvider ---
# This single instance will be imported and used by other modules.
data_provider = DataProvider()
