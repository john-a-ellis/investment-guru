# modules/historical_price_converter.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_historical_usd_to_cad_rates(start_date, end_date=None):
    """
    Get historical USD to CAD exchange rates for a date range
    
    Args:
        start_date (datetime or str): Start date
        end_date (datetime or str, optional): End date (defaults to current date)
        
    Returns:
        pandas.Series: Historical exchange rates indexed by date
    """
    try:
        # Handle string dates if provided
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Add a buffer day before and after to ensure we have data
        start_date_buffer = start_date - timedelta(days=5)
        end_date_buffer = end_date + timedelta(days=1)
        
        # Get exchange rate data from Yahoo Finance
        ticker = yf.Ticker("CAD=X")
        data = ticker.history(start=start_date_buffer, end=end_date_buffer)
        
        if not data.empty:
            # Extract just the close prices as our exchange rates
            rates = data['Close']
            
            # Forward-fill any missing values (weekends, holidays)
            rates = rates.fillna(method='ffill')
            
            return rates
        else:
            # Create a default series if data retrieval fails
            date_range = pd.date_range(start=start_date, end=end_date)
            return pd.Series([1.54] * len(date_range), index=date_range)
    except Exception as e:
        print(f"Error getting historical USD to CAD rates: {e}")
        # Create a default series if an exception occurs
        date_range = pd.date_range(start=start_date, end=end_date)
        return pd.Series([1.54] * len(date_range), index=date_range)

def convert_historical_prices(price_data, is_canadian=False):
    """
    Convert historical price data to the correct currency
    
    Args:
        price_data (DataFrame): Historical price data with DatetimeIndex
        is_canadian (bool): Whether the security is Canadian
        
    Returns:
        DataFrame: Price data with corrected prices
    """
    if not is_canadian:
        # For non-Canadian securities, return as is
        return price_data
    
    try:
        # Get the date range from the price data
        start_date = price_data.index.min()
        end_date = price_data.index.max()
        
        # Get historical exchange rates
        exchange_rates = get_historical_usd_to_cad_rates(start_date, end_date)
        
        # Create a copy of the price data to avoid modifying the original
        corrected_data = price_data.copy()
        
        # Apply currency conversion to each column that contains price data
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for column in price_columns:
            if column in corrected_data.columns:
                # For each date, multiply the price by the exchange rate for that date
                for date in corrected_data.index:
                    # Find the closest exchange rate date (in case of weekends/holidays)
                    closest_date = exchange_rates.index[exchange_rates.index.get_indexer([date], method='nearest')[0]]
                    rate = exchange_rates.loc[closest_date]
                    
                    # Apply the conversion
                    corrected_data.loc[date, column] *= rate
        
        return corrected_data
    except Exception as e:
        print(f"Error converting historical prices: {e}")
        return price_data  # Return original data if conversion fails

def get_corrected_historical_data(symbol, period="1y"):
    """
    Get historical data for a symbol with currency correction applied
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period (e.g., "1d", "1mo", "1y")
        
    Returns:
        DataFrame: Corrected historical price data
    """
    try:
        # Determine if this is a Canadian security
        is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
        
        # Get historical data from yfinance
        ticker = yf.Ticker(symbol)
        raw_data = ticker.history(period=period)
        
        # Apply currency correction if needed
        if is_canadian:
            return convert_historical_prices(raw_data, is_canadian=True)
        else:
            return raw_data
    except Exception as e:
        print(f"Error getting corrected historical data for {symbol}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs