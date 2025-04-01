# cgl_price_checker.py
# Script to debug and diagnose issues with CGL.TO pricing

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

def get_usd_to_cad_rate():
    """Get current USD to CAD exchange rate"""
    try:
        # Get exchange rate from Yahoo Finance
        ticker = yf.Ticker("CAD=X")
        data = ticker.history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            print(f"Current USD/CAD exchange rate: {rate}")
            return rate
        else:
            print("Error: No exchange rate data available")
            return 1.54  # Default fallback
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return 1.54  # Default fallback

def get_yfinance_price(symbol):
    """Get price from yfinance API"""
    try:
        # Get data from yfinance
        ticker = yf.Ticker(symbol)
        
        # Get info
        info = ticker.info
        
        # Get price data
        hist = ticker.history(period="1d")
        
        print(f"\n--- {symbol} PRICE DATA FROM YFINANCE ---")
        if not hist.empty:
            print(f"Latest close price: {hist['Close'].iloc[-1]}")
        else:
            print("No historical data available")
        
        # Print relevant fields from info
        print("\nRelevant fields from ticker.info:")
        price_fields = [
            'currency', 'regularMarketPrice', 'currentPrice', 
            'previousClose', 'regularMarketPreviousClose',
            'regularMarketOpen', 'regularMarketDayHigh', 'regularMarketDayLow'
        ]
        
        for field in price_fields:
            if field in info:
                print(f"{field}: {info[field]}")
        
        # Get raw price from API
        if not hist.empty:
            raw_price = hist['Close'].iloc[-1]
            
            # Print price with conversion
            usd_to_cad = get_usd_to_cad_rate()
            converted_price = raw_price * usd_to_cad
            
            print(f"\nRaw price: {raw_price}")
            print(f"USD to CAD rate: {usd_to_cad}")
            print(f"Converted price (USD to CAD): {converted_price}")
            
            return raw_price, converted_price
        else:
            return None, None
    except Exception as e:
        print(f"Error getting yfinance price for {symbol}: {e}")
        return None, None

def scrape_tmx_price(symbol):
    """Scrape price from TMX website (as a reference)"""
    # Remove the .TO suffix for TMX
    base_symbol = symbol.replace(".TO", "")
    
    # TMX URL
    url = f"https://money.tmx.com/en/quote/{base_symbol}"
    
    try:
        print(f"\n--- ATTEMPTING TO GET {symbol} PRICE FROM TMX ---")
        print(f"URL: {url}")
        print("Note: This is just for reference; web scraping should not be used in production")
        
        # For demonstration only - not implemented to avoid web scraping issues
        print("TMX scraping not implemented to avoid terms of service violations")
        return None
    except Exception as e:
        print(f"Error scraping TMX: {e}")
        return None

def check_multiple_price_sources():
    """Check multiple price sources to diagnose the issue"""
    print("===== DEBUGGING CGL.TO PRICE ISSUES =====")
    print(f"Current time: {datetime.now()}")
    
    # Check CGL.TO
    print("\n==== CHECKING CGL.TO ====")
    raw_price, converted_price = get_yfinance_price("CGL.TO")
    
    # Compare with a known US stock as reference
    print("\n==== CHECKING AAPL (REFERENCE) ====")
    get_yfinance_price("AAPL")
    
    # Check a different Canadian ETF for comparison
    print("\n==== CHECKING XIC.TO (ANOTHER CANADIAN ETF) ====")
    get_yfinance_price("XIC.TO")
    
    # Get reference price from TMX (demo only)
    tmx_price = scrape_tmx_price("CGL.TO")
    
    print("\n===== SUMMARY =====")
    print(f"CGL.TO yfinance raw price: {raw_price}")
    print(f"CGL.TO converted to CAD: {converted_price}")
    print(f"Expected price: 37.09")
    
    if raw_price and converted_price:
        expected_rate = 37.09 / raw_price
        print(f"\nTo get 37.09 from {raw_price}, would need exchange rate of: {expected_rate}")
        print(f"Current exchange rate being used: {get_usd_to_cad_rate()}")
        
        # Check if conversion is backwards
        inverse_conversion = raw_price / get_usd_to_cad_rate()
        print(f"Inverse conversion (CAD to USD): {inverse_conversion}")
    
    print("\n===== CONCLUSION =====")
    print("Based on the above information, check if:")
    print("1. The price from yfinance is already in CAD")
    print("2. The conversion is being applied incorrectly (backwards)")
    print("3. There might be a different conversion rate for gold ETFs")
    print("4. The exchange rate source may need to be updated")

if __name__ == "__main__":
    check_multiple_price_sources()