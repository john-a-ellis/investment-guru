# modules/currency_utils.py
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def get_usd_to_cad_rate():
    """
    Get the current USD to CAD exchange rate.
    
    Returns:
        float: Current USD to CAD exchange rate
    """
    try:
        # Get exchange rate data from Yahoo Finance
        ticker = yf.Ticker("CAD=X")  # Yahoo Finance symbol for USD/CAD rate
        data = ticker.history(period="1d")
        
        if not data.empty:
            rate = data['Close'].iloc[-1]
            return rate
        else:
            # Default rate if data retrieval fails
            return 1.33
    except Exception as e:
        print(f"Error getting USD to CAD rate: {e}")
        # Default rate if an exception occurs
        return 1.33

def get_historical_usd_to_cad_rates(start_date=None, end_date=None):
    """
    Get historical USD to CAD exchange rates.
    
    Args:
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pandas.Series: Historical USD to CAD exchange rates
    """
    try:
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        # Get exchange rate data from Yahoo Finance
        ticker = yf.Ticker("CAD=X")
        data = ticker.history(start=start_date, end=end_date)
        
        if not data.empty:
            return data['Close']
        else:
            # Default series if data retrieval fails
            date_range = pd.date_range(start=start_date, end=end_date)
            return pd.Series([1.33] * len(date_range), index=date_range)
    except Exception as e:
        print(f"Error getting historical USD to CAD rates: {e}")
        # Default series if an exception occurs
        date_range = pd.date_range(start=start_date, end=end_date)
        return pd.Series([1.33] * len(date_range), index=date_range)

def convert_usd_to_cad(usd_value):
    """
    Convert a USD value to CAD.
    
    Args:
        usd_value (float): Value in USD
        
    Returns:
        float: Value in CAD
    """
    rate = get_usd_to_cad_rate()
    return usd_value * rate

def convert_cad_to_usd(cad_value):
    """
    Convert a CAD value to USD.
    
    Args:
        cad_value (float): Value in CAD
        
    Returns:
        float: Value in USD
    """
    rate = get_usd_to_cad_rate()
    return cad_value / rate

def format_currency(value, currency="CAD"):
    """
    Format a currency value for display.
    
    Args:
        value (float): The numeric value
        currency (str): Currency code (CAD or USD)
        
    Returns:
        str: Formatted currency string
    """
    if currency == "CAD":
        return f"${value:.2f} CAD"
    elif currency == "USD":
        return f"${value:.2f} USD"
    else:
        return f"${value:.2f}"

def get_combined_value_cad(portfolio):
    """
    Calculate the combined value of a portfolio in CAD.
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        float: Total value in CAD
    """
    # Group investments by currency
    cad_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "CAD"}
    usd_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "USD"}
    
    # Calculate total value in CAD
    total_cad = sum(inv.get("current_value", 0) for inv in cad_investments.values())
    total_usd = sum(inv.get("current_value", 0) for inv in usd_investments.values())
    
    # Convert USD to CAD and add to total
    total_value_cad = total_cad + convert_usd_to_cad(total_usd)
    
    return total_value_cad