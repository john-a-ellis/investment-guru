# modules/portfolio_data_updater.py
import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf

def update_portfolio_data():
    """
    Updates portfolio data with current market prices and performance metrics
    """
    # Load portfolio
    portfolio = load_portfolio()
    
    # First, get current prices for all unique symbols
    symbol_prices = {}
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        if symbol not in symbol_prices:
            try:
                # Get price data from yfinance
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period="1d")
                
                if not price_data.empty:
                    # Get the price directly from yfinance
                    current_price = price_data['Close'].iloc[-1]
                    
                    # Determine currency based on symbol
                    is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
                    currency = "CAD" if is_canadian else "USD"
                    
                    # Store the price and currency for this symbol
                    symbol_prices[symbol] = {
                        "price": current_price,
                        "currency": currency
                    }
                    
                    print(f"Got price for {symbol}: {current_price} {currency}")
                else:
                    print(f"No price data available for {symbol}")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
    
    # Now update each investment with the consistent price
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        shares = details.get("shares", 0)
        purchase_price = details.get("purchase_price", 0)
        
        # Get the price data for this symbol
        symbol_data = symbol_prices.get(symbol)
        
        if symbol_data:
            current_price = symbol_data["price"]
            currency = symbol_data["currency"]
            
            # Store the currency for display purposes
            details["currency"] = currency
                
            # Calculate current value and gain/loss
            current_value = shares * current_price
            gain_loss = current_value - (shares * purchase_price)
            gain_loss_percent = (current_price / purchase_price - 1) * 100
            
            # Update investment details
            portfolio[investment_id].update({
                "current_price": current_price,
                "current_value": current_value,
                "gain_loss": gain_loss,
                "gain_loss_percent": gain_loss_percent,
                "currency": currency,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save updated portfolio
    save_portfolio(portfolio)
    
    return portfolio

def load_portfolio():
    """
    Load portfolio data from storage file
    """
    try:
        if os.path.exists('data/portfolio.json'):
            with open('data/portfolio.json', 'r') as f:
                return json.load(f)
        else:
            # Default empty portfolio if no file exists
            return {}
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return {}

def save_portfolio(portfolio):
    """
    Save portfolio data to storage file
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False

def add_investment(symbol, shares, purchase_price, purchase_date):
    """
    Add a new investment to the portfolio
    
    Args:
        symbol (str): Investment symbol
        shares (float): Number of shares
        purchase_price (float): Purchase price per share
        purchase_date (str): Purchase date in YYYY-MM-DD format
        
    Returns:
        bool: Success status
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Generate unique ID for this investment
    import uuid
    investment_id = str(uuid.uuid4())
    
    # Calculate initial values
    initial_value = shares * purchase_price
    
    # Determine currency based on symbol
    is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
    currency = "CAD" if is_canadian else "USD"
    
    # Find if we already have this symbol in our portfolio for the current price
    current_price = purchase_price
    for existing_inv in portfolio.values():
        if existing_inv.get("symbol") == symbol:
            current_price = existing_inv.get("current_price", purchase_price)
            break
    
    # If we don't have an existing price, try to get it from the API
    if current_price == purchase_price:
        try:
            ticker = yf.Ticker(symbol)
            price_data = ticker.history(period="1d")
            
            if not price_data.empty:
                current_price = price_data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error getting initial price for {symbol}: {e}")
    
    # Create investment entry
    investment = {
        "symbol": symbol,
        "shares": float(shares),
        "purchase_price": float(purchase_price),
        "purchase_date": purchase_date,
        "current_price": float(current_price),
        "current_value": float(shares) * float(current_price),
        "gain_loss": float(shares) * (float(current_price) - float(purchase_price)),
        "gain_loss_percent": ((float(current_price) / float(purchase_price)) - 1) * 100 if float(purchase_price) > 0 else 0,
        "currency": currency,
        "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to portfolio
    portfolio[investment_id] = investment
    
    # Save portfolio
    return save_portfolio(portfolio)

def remove_investment(investment_id):
    """
    Remove an investment from the portfolio
    
    Args:
        investment_id (str): ID of investment to remove
        
    Returns:
        bool: Success status
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Remove investment
    if investment_id in portfolio:
        del portfolio[investment_id]
        
        # Save updated portfolio
        return save_portfolio(portfolio)
    
    return False