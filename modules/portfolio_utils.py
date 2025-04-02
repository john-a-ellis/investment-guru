# modules/portfolio_utils.py
"""
Consolidated utility functions for portfolio management, eliminating redundancy
across multiple modules.
"""
import os
import json
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import uuid

def load_portfolio():
    """
    Load portfolio data from storage file
    
    Returns:
        dict: Portfolio data
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
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False

def load_transactions():
    """
    Load transaction records from storage file
    
    Returns:
        dict: Transaction records
    """
    try:
        if os.path.exists('data/transactions.json'):
            with open('data/transactions.json', 'r') as f:
                return json.load(f)
        else:
            # Default empty transactions if no file exists
            return {}
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return {}

def save_transactions(transactions):
    """
    Save transaction records to storage file
    
    Args:
        transactions (dict): Transaction records
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/transactions.json', 'w') as f:
            json.dump(transactions, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving transactions: {e}")
        return False

def load_tracked_assets():
    """
    Load tracked assets from storage file
    
    Returns:
        dict: Tracked assets
    """
    try:
        if os.path.exists('data/tracked_assets.json'):
            with open('data/tracked_assets.json', 'r') as f:
                return json.load(f)
        else:
            # Default assets if no file exists
            default_assets = {
                "CGL.TO": {"name": "iShares Gold Bullion ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
                "XTR.TO": {"name": "iShares Diversified Monthly Income ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
                "CWW.TO": {"name": "iShares Global Water Index ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
                "MFC.TO": {"name": "Manulife Financial Corp.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")},
                "TRI.TO": {"name": "Thomson Reuters Corp.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")},
                "PNG.V": {"name": "Kraken Robotics Inc.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")}
            }
            save_tracked_assets(default_assets)
            return default_assets
    except Exception as e:
        print(f"Error loading tracked assets: {e}")
        return {}

def save_tracked_assets(assets):
    """
    Save tracked assets to storage file
    
    Args:
        assets (dict): Tracked assets data
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/tracked_assets.json', 'w') as f:
            json.dump(assets, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving tracked assets: {e}")
        return False

def load_user_profile():
    """
    Load user profile from storage file
    
    Returns:
        dict: User profile data
    """
    try:
        if os.path.exists('data/user_profile.json'):
            with open('data/user_profile.json', 'r') as f:
                return json.load(f)
        else:
            # Default profile if no file exists
            default_profile = {
                "risk_level": 5,
                "investment_horizon": "medium",
                "initial_investment": 10000,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_user_profile(default_profile)
            return default_profile
    except Exception as e:
        print(f"Error loading user profile: {e}")
        return {}

def save_user_profile(profile):
    """
    Save user profile to storage file
    
    Args:
        profile (dict): User profile data
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/user_profile.json', 'w') as f:
            json.dump(profile, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving user profile: {e}")
        return False

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
            return 1.33
    except Exception as e:
        print(f"Error getting USD to CAD rate: {e}")
        # Default rate if an exception occurs
        return 1.33

def get_historical_usd_to_cad_rates(start_date=None, end_date=None):
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
            return pd.Series([1.33] * len(date_range), index=date_range)
    except Exception as e:
        print(f"Error getting historical USD to CAD rates: {e}")
        # Create a default series if an exception occurs
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date)
            return pd.Series([1.33] * len(date_range), index=date_range)
        return pd.Series([1.33], index=[datetime.now()])

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

def convert_cad_to_usd(cad_value):
    """
    Convert a CAD value to USD
    
    Args:
        cad_value (float): Value in CAD
        
    Returns:
        float: Value in USD
    """
    rate = get_usd_to_cad_rate()
    return cad_value / rate

def format_currency(value, currency="CAD"):
    """
    Format a currency value for display
    
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
    Calculate the combined value of a portfolio in CAD
    
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

def record_transaction(transaction_type, symbol, price, shares, date=None, notes=""):
    """
    Record a buy/sell transaction and update portfolio
    
    Args:
        transaction_type (str): "buy" or "sell"
        symbol (str): Asset symbol
        price (float): Price per share/unit
        shares (float): Number of shares/units
        date (str): Transaction date (optional, defaults to current date)
        notes (str): Transaction notes (optional)
        
    Returns:
        bool: Success status
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate total amount
    amount = float(price) * float(shares)
    
    # Load existing transactions
    transactions = load_transactions()
    
    # Generate unique ID
    transaction_id = str(uuid.uuid4())
    
    # Create transaction record
    transaction = {
        "type": transaction_type,
        "symbol": symbol,
        "price": float(price),
        "shares": float(shares),
        "amount": amount,
        "date": date,
        "notes": notes,
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to transactions list
    transactions[transaction_id] = transaction
    
    # Save transactions
    save_success = save_transactions(transactions)
    
    # Update portfolio based on transaction
    if save_success:
        update_portfolio_for_transaction(transaction_type, symbol, price, shares, date)
    
    return save_success

def update_portfolio_for_transaction(transaction_type, symbol, price, shares, date):
    """
    Update portfolio based on a transaction
    
    Args:
        transaction_type (str): "buy" or "sell"
        symbol (str): Asset symbol
        price (float): Price per share
        shares (float): Number of shares
        date (str): Transaction date
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Find if we already have this symbol in our portfolio
    existing_investment = None
    for inv_id, inv in portfolio.items():
        if inv.get("symbol") == symbol:
            existing_investment = (inv_id, inv)
            break
    
    if transaction_type.lower() == "buy":
        # Buy transaction
        if existing_investment:
            # Update existing investment
            inv_id, inv = existing_investment
            
            current_shares = inv.get("shares", 0)
            current_value = current_shares * inv.get("current_price", price)
            
            # Add new shares
            new_shares = current_shares + float(shares)
            
            # Calculate new average purchase price (weighted average)
            current_cost = current_shares * inv.get("purchase_price", 0)
            new_cost = float(shares) * float(price)
            new_avg_price = (current_cost + new_cost) / new_shares if new_shares > 0 else 0
            
            # Update investment
            inv["shares"] = new_shares
            inv["purchase_price"] = new_avg_price
            
            # If purchase date is earlier than current, update it
            if date < inv.get("purchase_date", date):
                inv["purchase_date"] = date
                
            # Update current value
            inv["current_value"] = new_shares * inv.get("current_price", price)
            
            # Recalculate gain/loss
            inv["gain_loss"] = inv["current_value"] - (new_shares * new_avg_price)
            inv["gain_loss_percent"] = (inv["current_value"] / (new_shares * new_avg_price) - 1) * 100
        else:
            # Add new investment with correct handling of currency
            add_investment(symbol, shares, price, date)
    
    elif transaction_type.lower() == "sell":
        # Sell transaction
        if existing_investment:
            # Update existing investment
            inv_id, inv = existing_investment
            
            current_shares = inv.get("shares", 0)
            
            # Remove sold shares
            new_shares = current_shares - float(shares)
            
            if new_shares <= 0:
                # If all shares sold, remove investment
                del portfolio[inv_id]
            else:
                # Update shares and current value
                inv["shares"] = new_shares
                inv["current_value"] = new_shares * inv.get("current_price", price)
                
                # Recalculate gain/loss
                purchase_cost = new_shares * inv.get("purchase_price", 0)
                inv["gain_loss"] = inv["current_value"] - purchase_cost
                inv["gain_loss_percent"] = (inv["current_value"] / purchase_cost - 1) * 100
    
    # Save updated portfolio
    save_portfolio(portfolio)

def add_investment(symbol, shares, purchase_price, purchase_date, asset_type="stock"):
    """
    Add a new investment to the portfolio
    
    Args:
        symbol (str): Investment symbol
        shares (float): Number of shares
        purchase_price (float): Purchase price per share
        purchase_date (str): Purchase date in YYYY-MM-DD format
        asset_type (str): Type of asset (stock, etf, mutual_fund, etc.)
        
    Returns:
        bool: Success status
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Generate unique ID for this investment
    investment_id = str(uuid.uuid4())
    
    # Calculate initial values
    initial_value = float(shares) * float(purchase_price)
    
    # Default current price to purchase price
    current_price = float(purchase_price)
    
    # For mutual funds, try to get price from our provider
    if asset_type == "mutual_fund":
        # Import here to avoid circular imports
        from modules.mutual_fund_provider import MutualFundProvider
        mutual_fund_provider = MutualFundProvider()
        
        # Get the most recent price
        fund_price = mutual_fund_provider.get_current_price(symbol)
        if fund_price:
            current_price = fund_price
            
        # Always treat mutual funds as CAD
        currency = "CAD"
    else:
        # Determine currency based on symbol for non-mutual fund investments
        is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
        currency = "CAD" if is_canadian else "USD"
        
        # Find if we already have this symbol in our portfolio for the current price
        for existing_inv in portfolio.values():
            if existing_inv.get("symbol") == symbol:
                current_price = existing_inv.get("current_price", purchase_price)
                break
        
        # If we don't have an existing price, try to get it from the API
        if current_price == float(purchase_price) and asset_type != "mutual_fund":
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
        "asset_type": asset_type,
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

def update_portfolio_data():

    """
    Updates portfolio data with current market prices and performance metrics
    
    Returns:
        dict: Updated portfolio data
    """
    # Load portfolio
    portfolio = load_portfolio()
    
    # Import mutual fund provider
    from modules.mutual_fund_provider import MutualFundProvider
    mutual_fund_provider = MutualFundProvider()
    
    # First, get current prices for all unique symbols
    symbol_prices = {}
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        asset_type = details.get("asset_type", "stock")
        
        if symbol not in symbol_prices:
            try:
                # Handle mutual funds differently
                if asset_type == "mutual_fund":
                    # Get the most recent price from our mutual fund provider
                    current_price = mutual_fund_provider.get_current_price(symbol)
                    
                    if current_price:
                        # Assume mutual funds are in CAD
                        symbol_prices[symbol] = {
                            "price": current_price,
                            "currency": "CAD"
                        }
                    else:
                        print(f"No price data available for mutual fund {symbol}")
                        # Use the purchase price as a fallback
                        purchase_price = details.get("purchase_price", 0)
                        symbol_prices[symbol] = {
                            "price": purchase_price,
                            "currency": "CAD"
                        }
                else:
                    # For stocks, ETFs, etc., use yfinance
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
                    else:
                        print(f"No price data available for {symbol}")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
    
    # Now update each investment with the consistent price
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        shares = details.get("shares", 0)
        purchase_price = details.get("purchase_price", 0)
        asset_type = details.get("asset_type", "stock")
        
        # Get the price data for this symbol
        symbol_data = symbol_prices.get(symbol)
        
        if symbol_data:
            current_price = symbol_data["price"]
            currency = symbol_data["currency"]
            
            # Store the currency for display purposes
            details["currency"] = currency
            details["asset_type"] = asset_type  # Ensure asset_type is stored
                
            # Calculate current value and gain/loss
            current_value = shares * current_price
            gain_loss = current_value - (shares * purchase_price)
            gain_loss_percent = (current_price / purchase_price - 1) * 100 if purchase_price > 0 else 0
            
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

def calculate_twrr(portfolio, transactions=None, period="3m"):
    """
    Calculate the Time-Weighted Rate of Return (TWRR) for a portfolio.
    
    This method eliminates the distorting effects of cash flows (deposits or 
    withdrawals) by breaking the overall period into sub-periods defined by 
    the timing of cash flows.
    
    Args:
        portfolio (dict): Current portfolio data
        transactions (dict): Transaction history (optional)
        period (str): Time period to calculate for ("1m", "3m", "6m", "1y", "all")
        
    Returns:
        dict: TWRR metrics including performance data series for charting
    """
    # Get transaction data if not provided
    if transactions is None:
        transactions = load_transactions()
    
    # Define date range based on period
    end_date = datetime.now()
    
    if period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:  # "all"
        # Find earliest transaction date
        earliest_date = end_date
        for transaction_id, transaction in transactions.items():
            try:
                transaction_date = datetime.strptime(transaction.get("date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                if transaction_date < earliest_date:
                    earliest_date = transaction_date
            except Exception as e:
                print(f"Error parsing transaction date: {e}")
                continue
        
        # Go back at least 1 day before earliest transaction to get a baseline
        start_date = earliest_date - timedelta(days=1)
    
    # Get historical portfolio values
    from components.portfolio_visualizer import get_portfolio_historical_data
    historical_data = get_portfolio_historical_data(portfolio, period)
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        return {
            'twrr': 0,
            'historical_values': pd.DataFrame(),
            'normalized_series': pd.Series()
        }
    
    # Get all transactions within the period
    period_transactions = []
    for transaction_id, transaction in transactions.items():
        try:
            transaction_date = datetime.strptime(transaction.get("date", ""), "%Y-%m-%d")
            if start_date <= transaction_date <= end_date:
                period_transactions.append({
                    'date': transaction_date,
                    'type': transaction.get("type", ""),
                    'amount': transaction.get("amount", 0)
                })
        except Exception as e:
            print(f"Error processing transaction for TWRR: {e}")
            continue
    
    # Sort transactions by date
    period_transactions.sort(key=lambda x: x['date'])
    
    # Calculate TWRR by breaking into sub-periods
    sub_period_returns = []
    portfolio_values = historical_data['Total']
    
    # Create a data structure for significant dates (transactions)
    significant_dates = []
    for transaction in period_transactions:
        # Only include buys and sells, not dividends or other transaction types
        if transaction['type'].lower() in ['buy', 'sell']:
            # Find closest date in the data
            closest_date = portfolio_values.index[portfolio_values.index >= transaction['date']]
            if len(closest_date) > 0:
                significant_dates.append({
                    'date': closest_date[0],
                    'amount': transaction['amount'] if transaction['type'].lower() == 'buy' else -transaction['amount']
                })
    
    # Make sure significant dates are sorted and unique
    significant_dates.sort(key=lambda x: x['date'])
    unique_dates = []
    date_map = {}
    for item in significant_dates:
        date_str = item['date'].strftime("%Y-%m-%d")
        if date_str in date_map:
            date_map[date_str]['amount'] += item['amount']
        else:
            date_map[date_str] = item
            unique_dates.append(item)
    
    significant_dates = unique_dates
    
    # If no significant dates, just calculate the total return
    if not significant_dates:
        if len(portfolio_values) >= 2:
            first_val = portfolio_values.iloc[0]
            last_val = portfolio_values.iloc[-1]
            if first_val > 0:
                total_return = (last_val / first_val) - 1
                sub_period_returns.append((1 + total_return))
    else:
        # Calculate returns for each sub-period
        prev_date = portfolio_values.index[0]
        prev_value = portfolio_values.iloc[0]
        
        for date_item in significant_dates:
            current_date = date_item['date']
            flow_amount = date_item['amount']
            
            # Get value just before flow
            try:
                before_flow_value = portfolio_values.loc[current_date]
            except KeyError:
                # Get the closest previous date
                before_dates = portfolio_values.index[portfolio_values.index <= current_date]
                if len(before_dates) > 0:
                    before_flow_value = portfolio_values.loc[before_dates[-1]]
                else:
                    continue
            
            # Calculate return for this sub-period
            if prev_value > 0:
                sub_period_return = (before_flow_value / prev_value) - 1
                sub_period_returns.append((1 + sub_period_return))
            
            # Update for next sub-period (adjust for cash flow)
            prev_value = before_flow_value + flow_amount
            prev_date = current_date
        
        # Calculate return for the last sub-period
        last_value = portfolio_values.iloc[-1]
        if prev_value > 0:
            last_sub_period_return = (last_value / prev_value) - 1
            sub_period_returns.append((1 + last_sub_period_return))
    
    # Calculate the overall TWRR
    if sub_period_returns:
        twrr = (np.prod(sub_period_returns) - 1) * 100
    else:
        twrr = 0
    
    # Create a normalized series for charting
    normalized_series = pd.Series(index=portfolio_values.index)
    
    # Calculate the impact of cash flows on the performance line
    running_adjustment = 1.0
    last_date = portfolio_values.index[0]
    normalized_series.iloc[0] = 100  # Start at 100
    
    for i in range(1, len(portfolio_values)):
        current_date = portfolio_values.index[i]
        
        # Check if there were any cash flows between the last date and current date
        flows_between = [item for item in significant_dates 
                         if last_date < item['date'] <= current_date]
        
        if flows_between:
            # There were cash flows, need to adjust
            for flow in flows_between:
                flow_date = flow['date']
                flow_amount = flow['amount']
                
                # Get portfolio value before and after flow
                try:
                    before_flow = portfolio_values.loc[flow_date]
                except KeyError:
                    before_dates = portfolio_values.index[portfolio_values.index < flow_date]
                    before_flow = portfolio_values.loc[before_dates[-1]] if len(before_dates) > 0 else 0
                
                # Calculate adjustment factor
                if before_flow > 0:
                    flow_factor = before_flow / (before_flow + flow_amount)
                    running_adjustment *= flow_factor
        
        # Calculate the normalized value that removes the impact of cash flows
        if running_adjustment > 0:
            performance_only = portfolio_values.iloc[i] / (portfolio_values.iloc[0] * running_adjustment)
            normalized_series.iloc[i] = performance_only * 100
        else:
            normalized_series.iloc[i] = normalized_series.iloc[i-1]
        
        last_date = current_date
    
    return {
        'twrr': twrr,
        'historical_values': portfolio_values,
        'normalized_series': normalized_series
    }

def get_money_weighted_return(portfolio, transactions=None, period="3m"):
    """
    Calculate the Money-Weighted Rate of Return (Internal Rate of Return or IRR)
    for a portfolio over a given period.
    
    Args:
        portfolio (dict): Current portfolio data
        transactions (dict): Transaction history (optional)
        period (str): Time period to calculate for ("1m", "3m", "6m", "1y", "all")
        
    Returns:
        float: Money-weighted return (IRR) as a percentage
    """
    from scipy import optimize
    
    # Get transaction data if not provided
    if transactions is None:
        transactions = load_transactions()
    
    # Define date range based on period
    end_date = datetime.now()
    
    if period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:  # "all"
        # Find earliest transaction date
        earliest_date = end_date
        for transaction_id, transaction in transactions.items():
            try:
                transaction_date = datetime.strptime(transaction.get("date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                if transaction_date < earliest_date:
                    earliest_date = transaction_date
            except Exception as e:
                print(f"Error parsing transaction date: {e}")
                continue
        
        # Go back at least 1 day before earliest transaction to get a baseline
        start_date = earliest_date - timedelta(days=1)
    
    # Get all transactions within the period
    cash_flows = []
    for transaction_id, transaction in transactions.items():
        try:
            transaction_date = datetime.strptime(transaction.get("date", ""), "%Y-%m-%d")
            if start_date <= transaction_date <= end_date:
                transaction_type = transaction.get("type", "").lower()
                amount = transaction.get("amount", 0)
                
                # Buy is negative cash flow (money leaving your account)
                # Sell is positive cash flow (money coming into your account)
                if transaction_type == "buy":
                    cash_flows.append((transaction_date, -amount))
                elif transaction_type == "sell":
                    cash_flows.append((transaction_date, amount))
        except Exception as e:
            print(f"Error processing transaction for IRR: {e}")
            continue
    
    # Current portfolio value is a positive cash flow (as if you were to sell everything)
    portfolio_value = sum(inv.get("current_value", 0) for inv in portfolio.values())
    cash_flows.append((end_date, portfolio_value))
    
    # Initial portfolio value (at start of period) is a negative cash flow
    # (as if you had just bought everything)
    # We'll need to estimate this if not explicitly provided
    from components.portfolio_visualizer import get_portfolio_historical_data
    historical_data = get_portfolio_historical_data(portfolio, period)
    
    if not historical_data.empty and 'Total' in historical_data.columns:
        initial_dates = historical_data.index[historical_data.index >= start_date]
        if len(initial_dates) > 0:
            initial_date = initial_dates[0]
            initial_value = historical_data.loc[initial_date, 'Total']
            cash_flows.append((initial_date, -initial_value))
    
    # Sort cash flows by date
    cash_flows.sort(key=lambda x: x[0])
    
    # Convert to days since first cash flow for IRR calculation
    if cash_flows:
        first_date = cash_flows[0][0]
        days_values = [(d - first_date).days for d, v in cash_flows]
        values = [v for _, v in cash_flows]
        
        # Function to calculate NPV with a given rate
        def npv(rate):
            return sum(values[i] / (1 + rate) ** (days_values[i] / 365) for i in range(len(values)))
        
        # Find IRR (rate where NPV is zero)
        try:
            irr = optimize.newton(npv, 0.1)  # Use 10% as initial guess
            return irr * 100  # Convert to percentage
        except:
            try:
                # Fall back to a more robust but slower method
                irr = optimize.brentq(npv, -0.999, 5)  # Reasonable range for returns
                return irr * 100  # Convert to percentage
            except:
                # If all else fails
                return 0
    
    return 0