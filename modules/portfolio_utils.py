# modules/portfolio_utils.py
"""
Consolidated utility functions for portfolio management, using the PostgreSQL database.
"""
import logging
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import uuid
from modules.db_utils import execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_portfolio():
    """
    Load portfolio data from database
    
    Returns:
        dict: Portfolio data
    """
    query = "SELECT * FROM portfolio;"
    portfolios = execute_query(query, fetchall=True)
    
    # Convert to dictionary keyed by investment ID
    result = {}
    if portfolios:
        for inv in portfolios:
            # Convert RealDictRow to regular dict
            inv_dict = dict(inv)
            
            # Format dates as strings
            inv_dict['purchase_date'] = inv_dict['purchase_date'].strftime("%Y-%m-%d")
            inv_dict['added_date'] = inv_dict['added_date'].strftime("%Y-%m-%d %H:%M:%S")
            if inv_dict['last_updated']:
                inv_dict['last_updated'] = inv_dict['last_updated'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to result dictionary
            result[inv_dict['id']] = inv_dict
    
    return result

def load_tracked_assets():
    """
    Load tracked assets from database
    
    Returns:
        dict: Tracked assets
    """
    query = "SELECT * FROM tracked_assets;"
    assets = execute_query(query, fetchall=True)
    
    # Convert to dictionary keyed by symbol
    result = {}
    if assets:
        for asset in assets:
            # Convert RealDictRow to regular dict
            asset_dict = dict(asset)
            
            # Format dates as strings
            asset_dict['added_date'] = asset_dict['added_date'].strftime("%Y-%m-%d")
            
            # Add to result dictionary
            result[asset_dict['symbol']] = {
                'name': asset_dict['name'],
                'type': asset_dict['type'],
                'added_date': asset_dict['added_date']
            }
    
    return result

def save_tracked_assets(assets):
    """
    Save tracked assets to database
    
    Args:
        assets (dict): Tracked assets data
        
    Returns:
        bool: Success status
    """
    try:
        # First, clear existing assets
        clear_query = "DELETE FROM tracked_assets;"
        execute_query(clear_query, commit=True)
        
        # Insert each asset
        for symbol, details in assets.items():
            insert_query = """
            INSERT INTO tracked_assets (
                symbol, name, type, added_date
            ) VALUES (
                %s, %s, %s, %s
            );
            """
            
            params = (
                symbol,
                details.get('name', ''),
                details.get('type', 'stock'),
                details.get('added_date', datetime.now().strftime("%Y-%m-%d"))
            )
            
            result = execute_query(insert_query, params, commit=True)
            if result is None:
                logger.error(f"Failed to save tracked asset: {symbol}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error saving tracked assets: {e}")
        return False

def load_user_profile():
    """
    Load user profile from database
    
    Returns:
        dict: User profile data
    """
    query = "SELECT * FROM user_profile ORDER BY id LIMIT 1;"
    profile = execute_query(query, fetchone=True)
    
    if profile:
        # Convert RealDictRow to regular dict
        profile_dict = dict(profile)
        
        # Format dates as strings
        profile_dict['last_updated'] = profile_dict['last_updated'].strftime("%Y-%m-%d %H:%M:%S")
        
        return profile_dict
    
    # Return default profile if none exists
    return {
        "risk_level": 5,
        "investment_horizon": "medium",
        "initial_investment": 10000,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def save_user_profile(profile):
    """
    Save user profile to database
    
    Args:
        profile (dict): User profile data
        
    Returns:
        bool: Success status
    """
    try:
        # Check if a profile already exists
        check_query = "SELECT COUNT(*) as count FROM user_profile;"
        result = execute_query(check_query, fetchone=True)
        
        if result and result['count'] == 0:
            # Insert new profile
            insert_query = """
            INSERT INTO user_profile (
                risk_level, investment_horizon, initial_investment, last_updated
            ) VALUES (
                %s, %s, %s, %s
            );
            """
            
            params = (
                profile.get("risk_level", 5),
                profile.get("investment_horizon", "medium"),
                float(profile.get("initial_investment", 10000)),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            result = execute_query(insert_query, params, commit=True)
        else:
            # Update existing profile
            update_query = """
            UPDATE user_profile SET
                risk_level = %s,
                investment_horizon = %s,
                initial_investment = %s,
                last_updated = %s
            WHERE id = (SELECT id FROM user_profile ORDER BY id LIMIT 1);
            """
            
            params = (
                profile.get("risk_level", 5),
                profile.get("investment_horizon", "medium"),
                float(profile.get("initial_investment", 10000)),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            result = execute_query(update_query, params, commit=True)
        
        return result is not None
    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        return False

def update_portfolio_data():
    """
    Updates portfolio data with current market prices and performance metrics
    
    Returns:
        dict: Updated portfolio data
    """
    # Get all portfolio investments
    query = "SELECT * FROM portfolio;"
    investments = execute_query(query, fetchall=True)
    
    if not investments:
        return {}
    
    # Import mutual fund provider
    from modules.mutual_fund_provider import MutualFundProvider
    mutual_fund_provider = MutualFundProvider()
    
    # First, get current prices for all unique symbols
    unique_symbols = {inv['symbol'] for inv in investments}
    symbol_prices = {}
    
    for symbol in unique_symbols:
        # Find asset type for this symbol (use the first occurrence)
        asset_type = next((inv['asset_type'] for inv in investments if inv['symbol'] == symbol), "stock")
        
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
                    logger.warning(f"No price data available for mutual fund {symbol}")
                    # Use the purchase price as a fallback (from the first matching investment)
                    purchase_price = next((float(inv['purchase_price']) for inv in investments if inv['symbol'] == symbol), 0)
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
                    logger.warning(f"No price data available for {symbol}")
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
    
    # Now update each investment with the consistent price
    updates = []
    for inv in investments:
        symbol = inv['symbol']
        shares = float(inv['shares'])
        purchase_price = float(inv['purchase_price'])
        asset_type = inv['asset_type']
        investment_id = inv['id']
        
        # Get the price data for this symbol
        symbol_data = symbol_prices.get(symbol)
        
        if symbol_data:
            current_price = symbol_data["price"]
            currency = symbol_data["currency"]
            
            # Calculate current value and gain/loss
            current_value = shares * current_price
            gain_loss = current_value - (shares * purchase_price)
            gain_loss_percent = (current_price / purchase_price - 1) * 100 if purchase_price > 0 else 0
            
            # Update investment details in database
            update_query = """
            UPDATE portfolio SET
                current_price = %s,
                current_value = %s,
                gain_loss = %s,
                gain_loss_percent = %s,
                currency = %s,
                last_updated = %s
            WHERE id = %s;
            """
            
            params = (
                current_price,
                current_value,
                gain_loss,
                gain_loss_percent,
                currency,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                investment_id
            )
            
            execute_query(update_query, params, commit=True)
    
    # Return updated portfolio
    return load_portfolio()

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
        logger.error(f"Error getting USD to CAD rate: {e}")
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
        logger.error(f"Error getting historical USD to CAD rates: {e}")
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

def get_combined_value_cad(portfolio):
    """
    Calculate the combined value of a portfolio in CAD
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        float: Total value in CAD
    """
    # Group investments by currency
    cad_investments = [inv for inv in portfolio.values() if inv.get("currency", "USD") == "CAD"]
    usd_investments = [inv for inv in portfolio.values() if inv.get("currency", "USD") == "USD"]
    
    # Calculate total value in CAD
    total_cad = sum(float(inv.get("current_value", 0)) for inv in cad_investments)
    total_usd = sum(float(inv.get("current_value", 0)) for inv in usd_investments)
    
    # Convert USD to CAD and add to total
    total_value_cad = total_cad + convert_usd_to_cad(total_usd)
    
    return total_value_cad

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
        # Get transactions from the database
        transactions_query = """
        SELECT * FROM transactions 
        ORDER BY transaction_date, recorded_at;
        """
        transaction_records = execute_query(transactions_query, fetchall=True)
        
        # Convert to dict format with IDs as keys
        transactions = {}
        if transaction_records:
            for tx in transaction_records:
                tx_dict = dict(tx)
                
                # Format dates for consistency
                tx_dict['date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
                tx_dict['recorded_at'] = tx_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
                transactions[str(tx_dict['id'])] = tx_dict
    
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
        # Find earliest transaction date from the database
        earliest_date_query = """
        SELECT MIN(transaction_date) as earliest_date FROM transactions;
        """
        earliest_result = execute_query(earliest_date_query, fetchone=True)
        
        earliest_date = end_date
        if earliest_result and earliest_result['earliest_date']:
            earliest_date = earliest_result['earliest_date']
        
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
    
    # Get all transactions within the period from the database
    period_transactions_query = """
    SELECT id, transaction_date as date, type, amount 
    FROM transactions 
    WHERE transaction_date BETWEEN %s AND %s
    ORDER BY transaction_date;
    """
    
    period_transactions_result = execute_query(
        period_transactions_query, 
        (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        fetchall=True
    )
    
    period_transactions = []
    if period_transactions_result:
        for tx in period_transactions_result:
            tx_dict = dict(tx)
            period_transactions.append({
                'date': tx_dict['date'],
                'type': tx_dict['type'],
                'amount': float(tx_dict['amount'])
            })
    
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
        # Get transactions from the database
        transactions_query = """
        SELECT * FROM transactions 
        WHERE transaction_date >= %s
        ORDER BY transaction_date, recorded_at;
        """
        
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
            # Find earliest transaction date from the database
            earliest_date_query = """
            SELECT MIN(transaction_date) as earliest_date FROM transactions;
            """
            earliest_result = execute_query(earliest_date_query, fetchone=True)
            
            start_date = end_date - timedelta(days=365)  # Default to 1 year
            if earliest_result and earliest_result['earliest_date']:
                start_date = earliest_result['earliest_date']
        
        transaction_records = execute_query(
            transactions_query, 
            (start_date.strftime("%Y-%m-%d"),),
            fetchall=True
        )
        
        transactions = {}
        if transaction_records:
            for tx in transaction_records:
                tx_dict = dict(tx)
                
                # Format dates for consistency
                tx_dict['date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
                tx_dict['recorded_at'] = tx_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
                transactions[str(tx_dict['id'])] = tx_dict
    
    # Get all transactions within the period
    cash_flows = []
    for transaction_id, transaction in transactions.items():
        transaction_type = transaction.get("type", "").lower()
        amount = float(transaction.get("amount", 0))
        transaction_date = datetime.strptime(transaction.get("date", ""), "%Y-%m-%d")
        
        # Buy is negative cash flow (money leaving your account)
        # Sell is positive cash flow (money coming into your account)
        if transaction_type == "buy":
            cash_flows.append((transaction_date, -amount))
        elif transaction_type == "sell":
            cash_flows.append((transaction_date, amount))
    
    # Current portfolio value is a positive cash flow (as if you were to sell everything)
    portfolio_value = sum(float(inv.get("current_value", 0)) for inv in portfolio.values())
    cash_flows.append((datetime.now(), portfolio_value))
    
    # Initial portfolio value (at start of period) is a negative cash flow
    # We'll need to estimate this if not explicitly provided
    from components.portfolio_visualizer import get_portfolio_historical_data
    historical_data = get_portfolio_historical_data(portfolio, period)
    
    if not historical_data.empty and 'Total' in historical_data.columns:
        initial_dates = historical_data.index[historical_data.index >= (end_date - timedelta(days={"1m": 30, "3m": 90, "6m": 180, "1y": 365, "all": 36500}[period]))]
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