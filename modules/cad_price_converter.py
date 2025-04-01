# modules/cad_price_converter.py
import yfinance as yf
from datetime import datetime, timedelta

def get_usd_to_cad_rate():
    """
    Get the current USD to CAD exchange rate
    
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
            return 1.54
    except Exception as e:
        print(f"Error getting USD to CAD rate: {e}")
        # Default rate if an exception occurs
        return 1.54

def convert_usd_to_cad(usd_value):
    """
    Convert a USD value to CAD
    
    Args:
        usd_value (float): Value in USD
        
    Returns:
        float: Value in CAD
    """
    rate = get_usd_to_cad_rate()
    return usd_value * rate

def get_correct_cad_price(symbol, price_from_api):
    """
    Get the correct CAD price for a Canadian security
    
    Args:
        symbol (str): Stock symbol (e.g., CGL.TO)
        price_from_api (float): Price returned from the API
        
    Returns:
        float: Corrected price in CAD
    """
    # Check if this is a Canadian security
    is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
    
    if not is_canadian:
        # Return the original price for non-Canadian securities
        return price_from_api
    
    # For Canadian securities, the API might be returning USD prices
    # Convert to CAD using the current exchange rate
    corrected_price = convert_usd_to_cad(price_from_api)
    
    # Return the corrected price
    return corrected_price