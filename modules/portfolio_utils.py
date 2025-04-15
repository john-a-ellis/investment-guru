# modules/portfolio_utils.py
"""
Consolidated utility functions for portfolio management, using the PostgreSQL database.
Handles loading, saving, and updating portfolio holdings, tracked assets,
user profile, transactions, and target allocation.
"""
import logging
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, date # Ensure date is imported
import uuid
import json # Needed for target allocation JSON handling
from modules.db_utils import execute_query
from modules.yf_utils import get_ticker_history, download_yf_data
from modules.data_provider import data_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Portfolio Holdings ---

def load_portfolio():
    """
    Load portfolio data from database.

    Returns:
        dict: Portfolio data keyed by investment ID (string).
    """
    query = "SELECT * FROM portfolio;"
    portfolios = execute_query(query, fetchall=True)

    result = {}
    if portfolios:
        for inv in portfolios:
            inv_dict = dict(inv)
            # Format dates and convert UUID to string
            inv_dict['purchase_date'] = inv_dict['purchase_date'].strftime("%Y-%m-%d") if inv_dict.get('purchase_date') else None
            inv_dict['added_date'] = inv_dict['added_date'].strftime("%Y-%m-%d %H:%M:%S") if inv_dict.get('added_date') else None
            if inv_dict.get('last_updated'):
                inv_dict['last_updated'] = inv_dict['last_updated'].strftime("%Y-%m-%d %H:%M:%S")

            # Convert Decimal to float for consistency
            for key in ['shares', 'purchase_price', 'current_price', 'current_value', 'gain_loss', 'gain_loss_percent']:
                 if key in inv_dict and inv_dict[key] is not None:
                     try:
                         inv_dict[key] = float(inv_dict[key])
                     except (TypeError, ValueError):
                         logger.warning(f"Could not convert {key} to float for investment {inv_dict.get('id')}")
                         inv_dict[key] = 0.0 # Default to 0.0 if conversion fails

            investment_id = str(inv_dict['id']) if inv_dict.get('id') is not None else None
            if investment_id:
                result[investment_id] = inv_dict

    return result

def update_portfolio_data():
    """
    Updates portfolio data with current market prices and performance metrics
    using the DataProvider.

    Returns:
        dict: Updated portfolio data.
    """
    logger.info("update_portfolio_data in portfolio_utils.py is being called")
    query = "SELECT * FROM portfolio;"
    investments = execute_query(query, fetchall=True)

    if not investments:
        return {}

    # Get current prices for all unique symbols using DataProvider
    unique_symbols = {inv['symbol'] for inv in investments if inv.get('symbol')}
    symbol_prices = {}

    for symbol in unique_symbols:
        # Find asset type for this symbol (use the first occurrence)
        asset_type = next((inv['asset_type'] for inv in investments if inv['symbol'] == symbol), "stock")
        try:
            quote = data_provider.get_current_quote(symbol, asset_type=asset_type)
            if quote and 'price' in quote and quote['price'] is not None:
                symbol_prices[symbol] = {
                    "price": float(quote['price']),
                    "currency": quote.get('currency', 'USD') # Get currency from quote or default
                }
            else:
                logger.warning(f"No price data available for {symbol} from DataProvider. Using purchase price as fallback.")
                # Fallback to purchase price if quote fails
                purchase_price = next((float(inv['purchase_price']) for inv in investments if inv['symbol'] == symbol), 0.0)
                currency = next((inv['currency'] for inv in investments if inv['symbol'] == symbol), 'USD')
                symbol_prices[symbol] = {
                    "price": purchase_price,
                    "currency": currency
                }
        except Exception as e:
            logger.error(f"Error getting price for {symbol} via DataProvider: {e}")
            # Fallback to purchase price on error
            purchase_price = next((float(inv['purchase_price']) for inv in investments if inv['symbol'] == symbol), 0.0)
            currency = next((inv['currency'] for inv in investments if inv['symbol'] == symbol), 'USD')
            symbol_prices[symbol] = {
                "price": purchase_price,
                "currency": currency
            }

    # Now update each investment in the database
    for inv in investments:
        try:
            symbol = inv['symbol']
            shares = float(inv['shares'])
            purchase_price = float(inv['purchase_price'])
            investment_id = str(inv['id']) if inv['id'] is not None else None

            symbol_data = symbol_prices.get(symbol)

            if symbol_data and investment_id is not None:
                current_price = float(symbol_data["price"])
                currency = symbol_data["currency"]

                current_value = shares * current_price
                gain_loss = current_value - (shares * purchase_price)

                def safe_division(numerator, denominator, default=0):
                    if abs(denominator) < 1e-10: return default
                    return numerator / denominator

                gain_loss_percent = (safe_division(current_price, purchase_price, 1) - 1) * 100

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
                    current_price, current_value, gain_loss, gain_loss_percent,
                    currency, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), investment_id
                )
                execute_query(update_query, params, commit=True)
        except Exception as e:
            logger.error(f"Error processing investment update {inv.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()

    return load_portfolio() # Return the freshly updated data

def add_investment(symbol, shares, purchase_price, purchase_date, asset_type="stock", name=None):
    """
    Add a new investment to the portfolio database.
    
    Args:
        symbol (str): Investment symbol
        shares (float): Number of shares
        purchase_price (float): Purchase price per share
        purchase_date (str): Purchase date in YYYY-MM-DD format
        asset_type (str): Type of asset (stock, etf, mutual_fund, etc.)
        name (str, optional): Name of the asset
        
    Returns:
        bool: Success status
    """
    # Generate unique ID for this investment
    import uuid
    investment_id = str(uuid.uuid4())
    
    # Standardize symbol format
    symbol_upper = symbol.upper().strip()
    
    # If name isn't provided, try to look it up from tracked assets
    if name is None:
        # Check if this symbol is in tracked assets
        tracked_assets = load_tracked_assets()
        if symbol_upper in tracked_assets:
            name = tracked_assets[symbol_upper].get("name", symbol_upper)
        else:
            # Default to symbol if name not provided
            name = symbol_upper
    
    # Convert values to float for consistency
    shares_float = float(shares)
    price_float = float(purchase_price)
    
    # Calculate initial values
    initial_value = shares_float * price_float
    
    # Default current price to purchase price
    current_price = price_float
    
    # For mutual funds, try to get price from our provider
    if asset_type == "mutual_fund":
        from modules.mutual_fund_provider import MutualFundProvider
        mutual_fund_provider = MutualFundProvider()
        
        # Get the most recent price
        fund_price = mutual_fund_provider.get_current_price(symbol_upper)
        if fund_price:
            current_price = fund_price
            
        # Always treat mutual funds as CAD
        currency = "CAD"
    else:
        # Determine currency based on symbol for non-mutual fund investments
        is_canadian = symbol_upper.endswith(".TO") or symbol_upper.endswith(".V") or "-CAD" in symbol_upper
        currency = "CAD" if is_canadian else "USD"
        
        # Find if we already have this symbol in our portfolio for the current price
        select_query = "SELECT current_price FROM portfolio WHERE symbol = %s LIMIT 1;"
        existing_price = execute_query(select_query, (symbol_upper,), fetchone=True)
        
        if existing_price and existing_price['current_price']:
            current_price = float(existing_price['current_price'])
        
        # If we don't have an existing price, try to get it from the API
        if current_price == price_float and asset_type != "mutual_fund":
            try:
                # First try using DataProvider
                from modules.data_provider import data_provider
                quote = data_provider.get_current_quote(symbol_upper)
                if quote and 'price' in quote:
                    current_price = float(quote['price'])
                else:
                    # Fallback to yfinance
                    import yfinance as yf
                    ticker = yf.Ticker(symbol_upper)
                    price_data = ticker.history(period="1d")
                    
                    if not price_data.empty:
                        current_price = price_data['Close'].iloc[-1]
            except Exception as e:
                logger.error(f"Error getting initial price for {symbol_upper}: {e}")
    
    # Calculate current value and gain/loss
    current_value = shares_float * current_price
    gain_loss = current_value - initial_value
    gain_loss_percent = ((current_price / price_float) - 1) * 100 if price_float > 0 else 0
    
    # Insert into portfolio table
    insert_query = """
    INSERT INTO portfolio (
        id, symbol, name, shares, purchase_price, purchase_date, asset_type,
        current_price, current_value, gain_loss, gain_loss_percent, 
        currency, added_date, last_updated
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    );
    """
    
    params = (
        investment_id,
        symbol_upper,
        name,
        shares_float,
        price_float,
        purchase_date,
        asset_type,
        current_price,
        current_value,
        gain_loss,
        gain_loss_percent,
        currency,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    result = execute_query(insert_query, params, commit=True)
    
    if result is not None:
        logger.info(f"Investment added successfully: {symbol_upper} {shares_float} shares")
        return True
    
    logger.error(f"Failed to add investment: {symbol_upper}")
    return False

def remove_investment(investment_id):
    """
    Remove an investment from the portfolio database.

    Args:
        investment_id (str): ID of investment to remove.

    Returns:
        bool: Success status.
    """
    investment_id_str = str(investment_id) if investment_id is not None else None
    if not investment_id_str:
        logger.error("Invalid investment ID: None or empty")
        return False

    delete_query = "DELETE FROM portfolio WHERE id = %s;"
    result = execute_query(delete_query, (investment_id_str,), commit=True)

    if result is not None:
        logger.info(f"Investment removed successfully: {investment_id_str}")
        return True

    logger.error(f"Failed to remove investment: {investment_id_str}")
    return False

# --- Tracked Assets ---

def load_tracked_assets():
    """
    Load tracked assets from database.

    Returns:
        dict: Tracked assets keyed by symbol.
    """
    query = "SELECT * FROM tracked_assets;"
    assets = execute_query(query, fetchall=True)

    result = {}
    if assets:
        for asset in assets:
            asset_dict = dict(asset)
            asset_dict['added_date'] = asset_dict['added_date'].strftime("%Y-%m-%d") if asset_dict.get('added_date') else None
            result[asset_dict['symbol']] = {
                'name': asset_dict['name'],
                'type': asset_dict['type'],
                'added_date': asset_dict['added_date']
            }
    return result

def save_tracked_assets(assets):
    """
    Save tracked assets to database (overwrites existing).

    Args:
        assets (dict): Tracked assets data keyed by symbol.

    Returns:
        bool: Success status.
    """
    try:
        clear_query = "DELETE FROM tracked_assets;"
        execute_query(clear_query, commit=True)

        for symbol, details in assets.items():
            insert_query = """
            INSERT INTO tracked_assets (symbol, name, type, added_date)
            VALUES (%s, %s, %s, %s);
            """
            params = (
                symbol, details.get('name', ''), details.get('type', 'stock'),
                details.get('added_date', datetime.now().strftime("%Y-%m-%d"))
            )
            result = execute_query(insert_query, params, commit=True)
            if result is None:
                logger.error(f"Failed to save tracked asset: {symbol}")
                # Optionally rollback or handle partial failure
                return False
        return True
    except Exception as e:
        logger.error(f"Error saving tracked assets: {e}")
        return False

# --- User Profile ---

def load_user_profile():
    """
    Load user profile from database.

    Returns:
        dict: User profile data or default profile.
    """
    query = "SELECT * FROM user_profile ORDER BY id LIMIT 1;"
    profile = execute_query(query, fetchone=True)

    if profile:
        profile_dict = dict(profile)
        profile_dict['last_updated'] = profile_dict['last_updated'].strftime("%Y-%m-%d %H:%M:%S") if profile_dict.get('last_updated') else None
        # Convert Decimal to float/int
        profile_dict['initial_investment'] = float(profile_dict.get('initial_investment', 0))
        profile_dict['risk_level'] = int(profile_dict.get('risk_level', 5))
        return profile_dict

    # Return default profile if none exists
    return {
        "risk_level": 5,
        "investment_horizon": "medium",
        "initial_investment": 10000.0,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def save_user_profile(profile):
    """
    Save user profile to database (inserts if none exists, updates otherwise).

    Args:
        profile (dict): User profile data.

    Returns:
        bool: Success status.
    """
    try:
        check_query = "SELECT COUNT(*) as count FROM user_profile;"
        result = execute_query(check_query, fetchone=True)

        risk_level = int(profile.get("risk_level", 5))
        horizon = profile.get("investment_horizon", "medium")
        initial_investment = float(profile.get("initial_investment", 10000))
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if result and result['count'] == 0:
            insert_query = """
            INSERT INTO user_profile (risk_level, investment_horizon, initial_investment, last_updated)
            VALUES (%s, %s, %s, %s);
            """
            params = (risk_level, horizon, initial_investment, last_updated)
            db_result = execute_query(insert_query, params, commit=True)
        else:
            update_query = """
            UPDATE user_profile SET risk_level = %s, investment_horizon = %s,
                initial_investment = %s, last_updated = %s
            WHERE id = (SELECT id FROM user_profile ORDER BY id LIMIT 1);
            """
            params = (risk_level, horizon, initial_investment, last_updated)
            db_result = execute_query(update_query, params, commit=True)

        return db_result is not None
    except Exception as e:
        logger.error(f"Error saving user profile: {e}")
        return False

# --- Transactions ---

def _update_portfolio_after_transaction(transaction_type, symbol, price, shares, date):
    """
    PRIVATE HELPER: Update portfolio based on a transaction.
    Called internally by record_transaction.

    Args:
        transaction_type (str): "buy" or "sell".
        symbol (str): Asset symbol.
        price (float): Price per share.
        shares (float): Number of shares.
        date (str): Transaction date (YYYY-MM-DD).
    """
    select_query = "SELECT * FROM portfolio WHERE symbol = %s;"
    existing_investment = execute_query(select_query, (symbol,), fetchone=True)

    if transaction_type.lower() == "buy":
        if existing_investment:
            # Update existing investment
            inv_id = existing_investment['id']
            current_shares = float(existing_investment['shares'])
            current_purchase_price = float(existing_investment['purchase_price'])
            current_price = float(existing_investment['current_price']) if existing_investment.get('current_price') else price # Use transaction price if no current price

            new_shares = current_shares + shares
            # Calculate new average purchase price (weighted average)
            current_cost = current_shares * current_purchase_price
            new_cost = shares * price
            new_avg_price = (current_cost + new_cost) / new_shares if new_shares > 0 else 0

            current_value = new_shares * current_price
            gain_loss = current_value - (new_shares * new_avg_price)
            gain_loss_percent = ((current_value / (new_shares * new_avg_price)) - 1) * 100 if new_avg_price > 0 else 0

            update_query = """
            UPDATE portfolio SET shares = %s, purchase_price = %s, current_value = %s,
                gain_loss = %s, gain_loss_percent = %s, last_updated = %s
            WHERE id = %s;
            """
            params = (
                new_shares, new_avg_price, current_value, gain_loss, gain_loss_percent,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), inv_id
            )

            # If purchase date is earlier than current, update it
            if date < existing_investment['purchase_date'].strftime("%Y-%m-%d"):
                update_query = """
                UPDATE portfolio SET shares = %s, purchase_price = %s, purchase_date = %s,
                    current_value = %s, gain_loss = %s, gain_loss_percent = %s, last_updated = %s
                WHERE id = %s;
                """
                params = (
                    new_shares, new_avg_price, date, current_value, gain_loss, gain_loss_percent,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), inv_id
                )

            execute_query(update_query, params, commit=True)
            logger.info(f"Updated existing investment {symbol} after buy.")
        else:
            # Add new investment if it doesn't exist
            logger.info(f"Adding new investment {symbol} after buy.")
            # Need to determine asset type - default to stock or try to infer?
            # For simplicity, let's default to 'stock' here. A better approach might involve looking up the symbol.
            add_investment(symbol, shares, price, date, asset_type="stock")

    elif transaction_type.lower() == "sell":
        if existing_investment:
            inv_id = existing_investment['id']
            current_shares = float(existing_investment['shares'])
            purchase_price = float(existing_investment['purchase_price'])
            current_price = float(existing_investment['current_price']) if existing_investment.get('current_price') else price # Use transaction price if no current price

            new_shares = current_shares - shares

            if new_shares <= 0.000001: # Use tolerance for float comparison
                # If all shares sold, remove investment
                logger.info(f"Removing investment {symbol} after selling all shares.")
                remove_investment(inv_id)
            else:
                # Update shares and current value
                current_value = new_shares * current_price
                purchase_cost = new_shares * purchase_price
                gain_loss = current_value - purchase_cost
                gain_loss_percent = ((current_value / purchase_cost) - 1) * 100 if purchase_cost > 0 else 0

                update_query = """
                UPDATE portfolio SET shares = %s, current_value = %s, gain_loss = %s,
                    gain_loss_percent = %s, last_updated = %s
                WHERE id = %s;
                """
                params = (
                    new_shares, current_value, gain_loss, gain_loss_percent,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), inv_id
                )
                execute_query(update_query, params, commit=True)
                logger.info(f"Updated investment {symbol} after sell.")
        else:
            logger.warning(f"Attempted to sell {symbol} which is not in the portfolio.")


def record_transaction(transaction_type, symbol, price, shares, date=None, notes=""):
    """
    Record a buy/sell transaction in the database and update the portfolio.
    Also updates cash positions to reflect the transaction.

    Args:
        transaction_type (str): "buy" or "sell"
        symbol (str): Asset symbol
        price (float): Price per share/unit
        shares (float): Number of shares/units
        date (str): Transaction date (YYYY-MM-DD, optional, defaults to current date)
        notes (str): Transaction notes (optional)

    Returns:
        bool: Success status
    """
    # Existing code stays the same...
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        # Ensure date is in YYYY-MM-DD format
        try:
            date = datetime.strptime(str(date), '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
             logger.error(f"Invalid date format for transaction: {date}. Using today.")
             date = datetime.now().strftime("%Y-%m-%d")

    symbol_upper = symbol.upper().strip()
    shares_float = float(shares)
    price_float = float(price)
    amount = price_float * shares_float
    transaction_id = str(uuid.uuid4())

    # Insert transaction
    insert_query = """
    INSERT INTO transactions (
        id, type, symbol, price, shares, amount,
        transaction_date, notes, recorded_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    params = (
        transaction_id, transaction_type.lower(), symbol_upper, price_float, shares_float, amount,
        date, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    result = execute_query(insert_query, params, commit=True)

    if result is not None:
        # Update portfolio based on transaction
        try:
            _update_portfolio_after_transaction(transaction_type, symbol_upper, price_float, shares_float, date)
            
            # IMPORTANT NEW CODE: Update cash position based on the transaction
            try:
                # Determine currency based on symbol
                currency = "CAD" if symbol_upper.endswith(".TO") or symbol_upper.endswith(".V") or symbol_upper.startswith("MAW") else "USD"
                
                # Update cash - SUBTRACT for buy, ADD for sell
                if transaction_type.lower() == "buy":
                    track_cash_position("buy", amount, currency)
                else:  # sell
                    track_cash_position("sell", amount, currency)
                
                logger.info(f"Cash position updated for {transaction_type} of {symbol_upper}")
            except Exception as cash_err:
                logger.error(f"Failed to update cash position: {cash_err}")
                # Continue execution even if cash update fails
            
            logger.info(f"Transaction recorded and portfolio updated: {transaction_type} {shares_float} shares of {symbol_upper}")
            return True
        except Exception as update_err:
             logger.error(f"Transaction recorded ({transaction_id}), but failed to update portfolio: {update_err}")
             return False # Indicate partial failure
    else:
        logger.error(f"Failed to record transaction: {symbol_upper} {transaction_type} {shares_float} shares")
        return False

def load_transactions(symbol=None, start_date=None, end_date=None):
    """
    Load transaction records from database.

    Args:
        symbol (str): Filter by symbol (optional).
        start_date (str): Filter by start date (YYYY-MM-DD, optional).
        end_date (str): Filter by end date (YYYY-MM-DD, optional).

    Returns:
        dict: Transaction records keyed by transaction ID (string).
    """
    query = "SELECT * FROM transactions"
    params = []
    filters = []

    if symbol:
        filters.append("symbol = %s")
        params.append(symbol.upper())
    if start_date:
        filters.append("transaction_date >= %s")
        params.append(start_date)
    if end_date:
        filters.append("transaction_date <= %s")
        params.append(end_date)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY transaction_date DESC, recorded_at DESC;"

    transactions = execute_query(query, tuple(params) if params else None, fetchall=True)

    result = {}
    if transactions:
        for tx in transactions:
            try:
                tx_dict = dict(tx)
                # Convert Decimal to float
                for key in ['shares', 'price', 'amount']:
                    if key in tx_dict and tx_dict[key] is not None:
                        tx_dict[key] = float(tx_dict[key])
                # Format dates
                tx_dict['date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
                tx_dict['transaction_date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
                tx_dict['recorded_at'] = tx_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                tx_dict['id'] = str(tx_dict['id'])
                result[tx_dict['id']] = tx_dict
            except Exception as e:
                logger.error(f"Error processing transaction {tx.get('id')}: {e}")
                continue
    return result

def get_earliest_transaction_date():
    """
    Fetches the earliest transaction date recorded in the transactions table.

    Returns:
        datetime.date or None: The earliest date found, or None if no transactions exist.
    """
    query = "SELECT MIN(transaction_date) as earliest_date FROM transactions;"
    try:
        result = execute_query(query, fetchone=True)
        if result and result.get('earliest_date'):
            earliest_date = result['earliest_date']
            if isinstance(earliest_date, datetime): return earliest_date.date()
            elif isinstance(earliest_date, date): return earliest_date
            else:
                try: return datetime.strptime(str(earliest_date), '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Could not parse earliest_date: {earliest_date}")
                    return None
        else:
            logger.info("No transactions found to determine earliest date.")
            return None
    except Exception as e:
        logger.error(f"Error fetching earliest transaction date: {e}")
        return None

# --- Target Allocation ---

def load_target_allocation():
    """
    Load the target allocation from the database.

    Returns:
        dict: Target allocation by asset type or default.
    """
    query = "SELECT allocation FROM target_allocation ORDER BY last_updated DESC LIMIT 1;"
    result = execute_query(query, fetchone=True)

    if result and result.get('allocation'):
        allocation = result['allocation']
        # The allocation should already be a dict if stored as JSONB
        if isinstance(allocation, dict):
            # Convert numeric values from string/Decimal if necessary (shouldn't be needed for JSONB)
            return {k: float(v) for k, v in allocation.items()}
        else:
             logger.warning(f"Target allocation from DB is not a dict: {type(allocation)}. Returning default.")
    else:
         logger.info("No target allocation found in DB. Returning default.")


    # Default allocation if none found or error
    return {
        "stock": 40.0, "etf": 30.0, "bond": 20.0,
        "cash": 5.0, "mutual_fund": 5.0, "crypto": 0.0
    }

def save_target_allocation(allocation):
    """
    Save the target allocation to the database (overwrites previous).

    Args:
        allocation (dict): Target allocation by asset type.

    Returns:
        bool: Success status.
    """
    try:
        # Ensure values are floats
        allocation_float = {k: float(v) for k, v in allocation.items()}
        allocation_json = json.dumps(allocation_float)

        # Check if table exists (optional, execute_query handles errors)
        # Upsert logic: Delete existing and insert new, or use proper UPSERT if DB supports easily
        delete_query = "DELETE FROM target_allocation;" # Simple approach: clear old targets
        execute_query(delete_query, commit=True)

        insert_query = """
        INSERT INTO target_allocation (allocation, last_updated) VALUES (%s, %s);
        """
        params = (allocation_json, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        result = execute_query(insert_query, params, commit=True)

        if result is not None:
             logger.info("Target allocation saved successfully.")
             return True
        else:
             logger.error("Failed to save target allocation.")
             return False
    except Exception as e:
        logger.error(f"Error saving target allocation: {e}")
        return False

# --- Rebalancing Analysis (Moved from portfolio_rebalancer) ---

def get_usd_to_cad_rate():
    """Get the current USD to CAD exchange rate using DataProvider."""
    try:
        rate = data_provider.get_exchange_rate("USD", "CAD")
        if rate is not None: return float(rate)
        logger.warning("Failed to get USD to CAD rate from DataProvider, using default 1.33")
        return 1.33
    except Exception as e:
        logger.error(f"Error getting USD to CAD rate: {e}")
        return 1.33

def convert_usd_to_cad(usd_value):
    """Convert a USD value to CAD."""
    rate = get_usd_to_cad_rate()
    return usd_value * rate

def get_current_allocation(portfolio):
    """
    Calculate the current allocation of the portfolio by asset type and symbol (in CAD).

    Args:
        portfolio (dict): Portfolio data from load_portfolio().

    Returns:
        tuple: (total_value_cad, asset_type_allocation, symbol_allocation)
               Allocations contain {'value': float, 'percentage': float}.
    """
    if not portfolio: return 0, {}, {}

    asset_type_values = {}
    symbol_values = {}
    total_value_cad = 0

    for inv_id, inv in portfolio.items():
        asset_type = inv.get("asset_type", "stock")
        symbol = inv.get("symbol", "")
        current_value = float(inv.get("current_value", 0))
        currency = inv.get("currency", "USD")

        value_cad = convert_usd_to_cad(current_value) if currency == "USD" else current_value

        asset_type_values[asset_type] = asset_type_values.get(asset_type, 0) + value_cad

        if symbol not in symbol_values:
            symbol_values[symbol] = {"value_cad": 0, "currency": currency, "asset_type": asset_type, "name": inv.get("name", symbol)}
        symbol_values[symbol]["value_cad"] += value_cad

        total_value_cad += value_cad

    asset_type_allocation = {
        atype: {"value": val, "percentage": (val / total_value_cad * 100) if total_value_cad > 0 else 0}
        for atype, val in asset_type_values.items()
    }
    symbol_allocation = {
        sym: {"value_cad": data["value_cad"], "percentage": (data["value_cad"] / total_value_cad * 100) if total_value_cad > 0 else 0,
              "currency": data["currency"], "asset_type": data["asset_type"], "name": data.get("name", sym)}
        for sym, data in symbol_values.items()
    }

    return total_value_cad, asset_type_allocation, symbol_allocation

def analyze_current_vs_target(portfolio):
    """
    Compare current allocation to target allocation and calculate drift.

    Args:
        portfolio (dict): Portfolio data from load_portfolio().

    Returns:
        dict: Analysis results including drift and action items.
    """
    total_value_cad, asset_type_allocation, symbol_allocation = get_current_allocation(portfolio)
    target_allocation = load_target_allocation() # Loads from DB via this module

    # Normalize target allocation percentages
    target_sum = sum(target_allocation.values())
    normalized_target = {k: (v / target_sum * 100) if target_sum > 0 else 0 for k, v in target_allocation.items()}

    drift_analysis = {}
    all_asset_types = set(asset_type_allocation.keys()) | set(normalized_target.keys())

    for asset_type in all_asset_types:
        target_pct = normalized_target.get(asset_type, 0)
        current_data = asset_type_allocation.get(asset_type, {"value": 0, "percentage": 0})
        current_pct = current_data["percentage"]
        current_value = current_data["value"]

        drift_pct = current_pct - target_pct
        target_value = (target_pct / 100) * total_value_cad
        rebalance_amount = target_value - current_value # Positive means buy, negative means sell

        drift_analysis[asset_type] = {
            "current_percentage": current_pct,
            "target_percentage": target_pct,
            "drift_percentage": drift_pct,
            "current_value": current_value,
            "target_value": target_value,
            "rebalance_amount": rebalance_amount,
            "action": "sell" if rebalance_amount < -0.01 else "buy" if rebalance_amount > 0.01 else "hold", # Use tolerance
            "is_actionable": abs(drift_pct) >= 5 # Actionable if drift is 5% or more
        }

    actionable_items = {k: v for k, v in drift_analysis.items() if v["is_actionable"]}
    sorted_actionable = dict(sorted(actionable_items.items(), key=lambda item: abs(item[1]["drift_percentage"]), reverse=True))

    # Basic recommendations (buy/sell asset type)
    recommendations = []
    for asset_type, data in sorted_actionable.items():
        action = data["action"]
        amount = abs(data["rebalance_amount"])
        if action == "buy":
            recommendations.append({
                "asset_type": asset_type, "action": "Buy", "amount": amount,
                "description": f"Increase {asset_type} by ${amount:.2f} to reach {data['target_percentage']:.1f}% target"
            })
        elif action == "sell":
            recommendations.append({
                "asset_type": asset_type, "action": "Sell", "amount": amount,
                "description": f"Decrease {asset_type} by ${amount:.2f} to reach {data['target_percentage']:.1f}% target"
            })

    # Placeholder for specific recommendations (logic remains in portfolio_rebalancer.py)
    specific_recommendations = []

    return {
        "total_value_cad": total_value_cad,
        "current_allocation": asset_type_allocation,
        "symbol_allocation": symbol_allocation,
        "target_allocation": normalized_target,
        "drift_analysis": drift_analysis,
        "actionable_items": sorted_actionable,
        "rebalancing_recommendations": recommendations,
        "specific_recommendations": specific_recommendations # This will be populated by portfolio_rebalancer.py
    }


# --- Other Utilities (Keep existing ones like currency conversion, TWRR, MWR etc.) ---
# ... (get_historical_usd_to_cad_rates, format_currency, calculate_twrr, get_money_weighted_return remain here) ...
# Make sure calculate_twrr and get_money_weighted_return use load_transactions from this module
def calculate_twrr(portfolio, transactions=None, period="3m"):
    """Calculate Time-Weighted Rate of Return (TWRR)."""
    # Simply pass through to the simplified version
    # The simplified version handles its own transaction loading
    return calculate_twrr_simplified(portfolio, period)

    # Ensure it uses the load_transactions from this module
    if transactions is None:
        # Define date range based on period
        end_date = datetime.now()
        if period == "1m": start_date = end_date - timedelta(days=30)
        elif period == "3m": start_date = end_date - timedelta(days=90)
        elif period == "6m": start_date = end_date - timedelta(days=180)
        elif period == "1y": start_date = end_date - timedelta(days=365)
        else: # "all"
             earliest_db_date = get_earliest_transaction_date()
             start_date = earliest_db_date if earliest_db_date else end_date - timedelta(days=365*5) # Default 5 years

        transactions = load_transactions(start_date=start_date.strftime('%Y-%m-%d') if start_date else None) # Use load_transactions from portfolio_utils

    # ... (rest of TWRR calculation logic - seems okay, relies on historical data fetching which is external) ...
    # Need to ensure get_portfolio_historical_data is correctly imported/used if called internally
    from components.portfolio_visualizer import get_portfolio_historical_data # Assuming this uses DataProvider correctly
    historical_data = get_portfolio_historical_data(portfolio, period)

    if historical_data.empty or 'Total' not in historical_data.columns:
        return {'twrr': 0, 'historical_values': pd.DataFrame(), 'normalized_series': pd.Series()}

    # Filter transactions from the loaded dict based on the period
    period_transactions = []
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days={'1m': 30, '3m': 90, '6m': 180, '1y': 365}.get(period, 365*100)) # Large default for 'all'
    if period == 'all':
        earliest_db_date = get_earliest_transaction_date()
        start_dt = datetime.combine(earliest_db_date, datetime.min.time()) if earliest_db_date else start_dt

    for tx_id, tx in transactions.items():
        tx_date = datetime.strptime(tx['date'], '%Y-%m-%d')
        if start_dt <= tx_date <= end_dt:
             period_transactions.append({
                 'date': tx_date.date(), # Use date object
                 'type': tx['type'],
                 'amount': float(tx['amount'])
             })

    # ... (rest of TWRR sub-period calculation logic - seems okay) ...
    # Ensure date comparisons and indexing work correctly with timezone-naive data
    sub_period_returns = []
    portfolio_values = historical_data['Total']
    if portfolio_values.index.tz is not None:
        portfolio_values.index = portfolio_values.index.tz_localize(None)

    significant_dates = []
    for transaction in period_transactions:
        if transaction['type'].lower() in ['buy', 'sell']:
            tx_date = transaction['date']
            # Convert date to Timestamp for comparison with index
            tx_datetime = pd.Timestamp(tx_date)
            if tx_datetime.tz is not None: tx_datetime = tx_datetime.tz_localize(None)

            # Find closest date in the data >= transaction date
            closest_dates = portfolio_values.index[portfolio_values.index >= tx_datetime]
            if len(closest_dates) > 0:
                significant_dates.append({
                    'date': closest_dates[0], # Use the actual index date
                    'amount': transaction['amount'] if transaction['type'].lower() == 'buy' else -transaction['amount']
                })
            # else: logger.warning(f"No portfolio value found on or after transaction date {tx_date}")

    # ... (rest of TWRR calculation logic) ...
    # Make sure np.prod works and handle empty sub_period_returns
    if not significant_dates and len(portfolio_values) >= 2:
        first_val = portfolio_values.iloc[0]
        last_val = portfolio_values.iloc[-1]
        if first_val > 0: sub_period_returns.append(1 + (last_val / first_val - 1))
    elif significant_dates:
         # Sort significant dates
        significant_dates.sort(key=lambda x: x['date'])
        # Aggregate flows on the same date
        agg_dates = {}
        for item in significant_dates:
            date_key = item['date']
            if date_key not in agg_dates: agg_dates[date_key] = {'date': date_key, 'amount': 0}
            agg_dates[date_key]['amount'] += item['amount']
        significant_dates = list(agg_dates.values())

        prev_date = portfolio_values.index[0]
        prev_value = portfolio_values.iloc[0]

        for date_item in significant_dates:
            current_date = date_item['date']
            flow_amount = date_item['amount']

            # Get value just before flow (use loc, handle potential KeyError)
            try:
                # Find index strictly before current_date if exact match fails
                idx_before = portfolio_values.index[portfolio_values.index < current_date]
                if not idx_before.empty:
                    before_flow_value = portfolio_values.loc[idx_before[-1]]
                else: # If current_date is the first date, use the first value
                     before_flow_value = portfolio_values.iloc[0]

                # Calculate return for this sub-period
                if prev_value is not None and abs(prev_value) > 1e-9:
                    sub_period_return = (before_flow_value / prev_value) - 1
                    sub_period_returns.append((1 + sub_period_return))
                else: logger.warning(f"Skipping sub-period ending {current_date} due to zero/None prev_value")


                # Update for next sub-period (value *after* flow)
                # Get value ON the flow date
                value_on_flow_date = portfolio_values.loc[current_date]
                prev_value = value_on_flow_date + flow_amount # Value after flow adjustment
                prev_date = current_date

            except KeyError:
                logger.warning(f"Could not find portfolio value for date {current_date} or before. Skipping flow.")
                continue # Skip this flow if data is missing

        # Calculate return for the last sub-period
        last_value = portfolio_values.iloc[-1]
        if prev_value is not None and abs(prev_value) > 1e-9:
            last_sub_period_return = (last_value / prev_value) - 1
            sub_period_returns.append((1 + last_sub_period_return))
        elif not significant_dates and len(portfolio_values) >=2 : # Handle case with no flows
             pass # Already calculated above
        else: logger.warning("Skipping last sub-period due to zero/None prev_value")


    # Calculate overall TWRR
    twrr = (np.prod(sub_period_returns) - 1) * 100 if sub_period_returns else 0

    # ... (rest of TWRR normalization logic - seems complex, verify carefully) ...
    # Simplified normalization: just return the TWRR value for now
    if not portfolio_values.empty:
        # Create a series starting at 100 on the first day
        normalized_series = pd.Series(index=portfolio_values.index, dtype=float)
        normalized_series.iloc[0] = 100.0

        # Apply sub-period returns cumulatively (this is simplified, needs careful date alignment)
        # A more robust way might involve calculating daily returns from portfolio_values
        # and applying them cumulatively, adjusting for cash flows if needed for pure TWRR viz.

        # Alternative: Calculate daily returns from the 'Total' series
        daily_returns_for_norm = portfolio_values.pct_change().fillna(0)
        normalized_series = (1 + daily_returns_for_norm).cumprod() * 100
        # Ensure it starts exactly at 100
        if not normalized_series.empty:
            normalized_series = normalized_series / normalized_series.iloc[0] * 100

    else:
        normalized_series = pd.Series(dtype=float) # Keep it empty if no values

    # Ensure the return dictionary includes the calculated series
    return {'twrr': twrr, 'historical_values': portfolio_values, 'normalized_series': normalized_series}
    

def get_money_weighted_return(portfolio, transactions=None, period="3m"):
    """Calculate Money-Weighted Rate of Return (IRR)."""
    # Ensure it uses the load_transactions from this module
    if transactions is None:
        # Define date range based on period
        end_date = datetime.now()
        if period == "1m": start_date = end_date - timedelta(days=30)
        elif period == "3m": start_date = end_date - timedelta(days=90)
        elif period == "6m": start_date = end_date - timedelta(days=180)
        elif period == "1y": start_date = end_date - timedelta(days=365)
        else: # "all"
             earliest_db_date = get_earliest_transaction_date()
             start_date = earliest_db_date if earliest_db_date else end_date - timedelta(days=365*5)

        transactions = load_transactions(start_date=start_date.strftime('%Y-%m-%d') if start_date else None)

    # ... (rest of MWR calculation logic - seems okay, relies on external libraries and historical data fetching) ...
    # Ensure get_portfolio_historical_data is correctly imported/used
    try:
        from scipy import optimize
        from components.portfolio_visualizer import get_portfolio_historical_data # Assuming this uses DataProvider correctly

        cash_flows = []
        # Add initial portfolio value (negative cash flow at start_date)
        historical_data = get_portfolio_historical_data(portfolio, period)
        if not historical_data.empty and 'Total' in historical_data.columns:
             period_start_dt = start_date if isinstance(start_date, datetime) else datetime.combine(start_date, datetime.min.time())
             # Find first value on or after start_date
             initial_dates = historical_data.index[historical_data.index >= period_start_dt]
             if not initial_dates.empty:
                 initial_date = initial_dates[0]
                 initial_value = historical_data.loc[initial_date, 'Total']
                 cash_flows.append((initial_date, -initial_value)) # Start value is outflow
             else: logger.warning("Could not determine initial portfolio value for MWR.")
        else: logger.warning("No historical data for MWR initial value.")


        # Add transactions within the period
        end_dt = datetime.now()
        start_dt = end_date - timedelta(days={'1m': 30, '3m': 90, '6m': 180, '1y': 365}.get(period, 365*100))
        if period == 'all':
             earliest_db_date = get_earliest_transaction_date()
             start_dt = datetime.combine(earliest_db_date, datetime.min.time()) if earliest_db_date else start_dt

        for tx_id, tx in transactions.items():
            tx_date = datetime.strptime(tx['date'], '%Y-%m-%d')
            if start_dt <= tx_date <= end_dt:
                 amount = float(tx['amount'])
                 # Buy is outflow (-), Sell is inflow (+)
                 flow = -amount if tx['type'] == 'buy' else amount
                 cash_flows.append((tx_date, flow))

        # Add final portfolio value (positive cash flow at end_date)
        portfolio_value = sum(float(inv.get("current_value", 0)) for inv in portfolio.values()) # Use current values
        cash_flows.append((end_dt, portfolio_value)) # End value is inflow

        # Sort cash flows by date
        cash_flows.sort(key=lambda x: x[0])

        if len(cash_flows) < 2:
            logger.warning("Not enough cash flows for IRR calculation")
            return 0

        # Prepare for IRR calculation
        dates = [cf[0] for cf in cash_flows]
        values = [cf[1] for cf in cash_flows]
        first_date = dates[0]
        time_diffs = [(d - first_date).days / 365.0 for d in dates] # Time in years

        # Define NPV function for solver
        def npv(rate):
            return sum(values[i] / ((1 + rate) ** time_diffs[i]) for i in range(len(values)))

        # Solve for IRR (rate where NPV is zero)
        try:
            # Use a bounded solver like brentq for robustness
            irr = optimize.brentq(npv, -0.99, 5.0) # Search between -99% and +500% return
            return irr * 100 # Return as percentage
        except ValueError as ve:
             # Handle cases where solver fails (e.g., no sign change)
             logger.warning(f"IRR calculation failed (ValueError: {ve}). Might indicate no solution or unusual cash flows.")
             # Try Newton's method as fallback? Or just return 0?
             try:
                 irr_newton = optimize.newton(npv, 0.1) # Initial guess 10%
                 # Check if Newton result is reasonable
                 if -1 < irr_newton < 10: # Arbitrary reasonable bounds
                     return irr_newton * 100
                 else: return 0 # Return 0 if Newton result is extreme
             except RuntimeError: # Newton might fail to converge
                 logger.warning("Newton's method also failed for IRR.")
                 return 0
        except Exception as e:
            logger.error(f"Unexpected error during IRR calculation: {e}")
            return 0

    except ImportError:
         logger.error("Scipy is required for Money Weighted Return (IRR) calculation. Please install it.")
         return 0
    except Exception as e:
        logger.error(f"Error in get_money_weighted_return: {e}")
        return 0
    
def reconcile_portfolio_holdings():
    """
    Reconcile portfolio holdings against transaction history to detect discrepancies.
    Calculates expected positions based on transactions and compares to actual holdings.
    
    Returns:
        dict: Reconciliation report with discrepancies if found
    """
    # Get all transactions chronologically
    all_transactions = load_transactions()
    sorted_transactions = sorted(
        all_transactions.values(), 
        key=lambda tx: datetime.strptime(tx['date'], '%Y-%m-%d')
    )
    
    # Calculate expected positions
    expected_positions = {}
    for tx in sorted_transactions:
        symbol = tx['symbol']
        tx_type = tx['type']
        shares = float(tx['shares'])
        
        if symbol not in expected_positions:
            expected_positions[symbol] = {
                'shares': 0,
                'total_cost': 0,
                'currency': 'CAD'  # Default, will be updated by transaction
            }
        
        position = expected_positions[symbol]
        
        if tx_type == 'buy':
            # Calculate new average cost
            current_cost = position['shares'] * position['avg_price'] if 'avg_price' in position else 0
            new_cost = shares * float(tx['price'])
            position['shares'] += shares
            position['total_cost'] = current_cost + new_cost
            position['avg_price'] = position['total_cost'] / position['shares'] if position['shares'] > 0 else 0
            # Note the currency from transaction if available
            if 'currency' in tx:
                position['currency'] = tx['currency']
        elif tx_type == 'sell':
            # Reduce position
            position['shares'] -= shares
            if position['shares'] <= 0.000001:  # Using small threshold for floating point
                position['shares'] = 0
                position['total_cost'] = 0
            else:
                # Keep same average price when selling
                position['total_cost'] = position['shares'] * position['avg_price'] if 'avg_price' in position else 0
    
    # Get actual holdings
    current_portfolio = load_portfolio()
    
    # Compare expected vs. actual
    discrepancies = []
    for symbol, expected in expected_positions.items():
        if expected['shares'] <= 0.000001:
            # Position should be closed
            for inv_id, actual in current_portfolio.items():
                if actual['symbol'] == symbol and float(actual['shares']) > 0.000001:
                    discrepancies.append({
                        'symbol': symbol,
                        'issue': 'Position should be closed but is still open',
                        'expected_shares': 0,
                        'actual_shares': float(actual['shares']),
                        'investment_id': inv_id
                    })
        else:
            # Position should be open
            found = False
            total_actual_shares = 0
            for inv_id, actual in current_portfolio.items():
                if actual['symbol'] == symbol:
                    found = True
                    total_actual_shares += float(actual['shares'])
            
            if not found:
                discrepancies.append({
                    'symbol': symbol,
                    'issue': 'Position should be open but is missing',
                    'expected_shares': expected['shares'],
                    'actual_shares': 0
                })
            elif abs(total_actual_shares - expected['shares']) > 0.001:  # Allow small rounding differences
                discrepancies.append({
                    'symbol': symbol,
                    'issue': 'Share count mismatch',
                    'expected_shares': expected['shares'],
                    'actual_shares': total_actual_shares,
                    'difference': total_actual_shares - expected['shares']
                })
    
    # Check for positions in portfolio not in transaction history
    for inv_id, holding in current_portfolio.items():
        symbol = holding['symbol']
        if symbol not in expected_positions and float(holding['shares']) > 0:
            discrepancies.append({
                'symbol': symbol,
                'issue': 'Position exists but has no transaction history',
                'expected_shares': 0,
                'actual_shares': float(holding['shares']),
                'investment_id': inv_id
            })
    
    return {
        'reconciliation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'portfolio_count': len(current_portfolio),
        'expected_positions_count': len([p for p in expected_positions.values() if p['shares'] > 0]),
        'discrepancies': discrepancies,
        'is_reconciled': len(discrepancies) == 0
    }

def track_cash_position(transaction_type, amount, currency):
    """
    Update cash position when recording transactions
    
    Args:
        transaction_type (str): 'buy' or 'sell'
        amount (float): Transaction amount
        currency (str): Currency code
        
    Returns:
        bool: Success status
    """
    try:
        # Get current cash positions
        query = "SELECT * FROM cash_positions WHERE currency = %s;"
        cash_position = execute_query(query, (currency,), fetchone=True)
        
        # Calculate adjustment
        adjustment = -amount if transaction_type == 'buy' else amount
        
        if cash_position:
            # Update existing position
            current_balance = float(cash_position['balance'])
            new_balance = current_balance + adjustment
            
            update_query = """
            UPDATE cash_positions 
            SET balance = %s, last_updated = %s 
            WHERE id = %s;
            """
            params = (new_balance, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cash_position['id'])
            execute_query(update_query, params, commit=True)
        else:
            # Create new position
            insert_query = """
            INSERT INTO cash_positions (currency, balance, last_updated) 
            VALUES (%s, %s, %s);
            """
            params = (currency, adjustment, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            execute_query(insert_query, params, commit=True)
        
        return True
    except Exception as e:
        logger.error(f"Error tracking cash position: {e}")
        return False

def get_cash_positions():
    """
    Get current cash positions
    
    Returns:
        dict: Cash positions by currency
    """
    query = "SELECT * FROM cash_positions;"
    positions = execute_query(query, fetchall=True)
    
    result = {}
    if positions:
        for pos in positions:
            result[pos['currency']] = {
                'balance': float(pos['balance']),
                'last_updated': pos['last_updated'].strftime("%Y-%m-%d %H:%M:%S") if pos.get('last_updated') else None
            }
    
    return result

def snapshot_portfolio_value(comment=None):
    """
    Take a snapshot of current portfolio value and store in history table.
    Useful for tracking performance over time.
    
    Args:
        comment (str): Optional comment about this valuation
        
    Returns:
        dict: Snapshot details
    """
    try:
        # First update portfolio data
        portfolio = update_portfolio_data()
        
        # Calculate total values by currency
        totals = {'CAD': 0.0, 'USD': 0.0}
        for inv_id, details in portfolio.items():
            currency = details.get('currency', 'USD')
            if currency in totals:
                totals[currency] += float(details.get('current_value', 0))
        
        # Get exchange rate
        usd_to_cad = get_usd_to_cad_rate()
        
        # Calculate total in CAD
        total_cad = totals['CAD'] + (totals['USD'] * usd_to_cad)
        
        # Get cash positions
        cash_positions = get_cash_positions()
        cash_cad = float(cash_positions.get('CAD', {}).get('balance', 0))
        cash_usd = float(cash_positions.get('USD', {}).get('balance', 0))
        cash_total_cad = cash_cad + (cash_usd * usd_to_cad)
        
        # Total portfolio value including cash
        grand_total_cad = total_cad + cash_total_cad
        
        # Record the snapshot
        snapshot_id = str(uuid.uuid4())
        snapshot_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        insert_query = """
        INSERT INTO portfolio_snapshots (
            id, snapshot_date, value_cad, value_usd, 
            cash_cad, cash_usd, total_value_cad,
            exchange_rate_usd_cad, comment
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        params = (
            snapshot_id, snapshot_date, totals['CAD'], totals['USD'],
            cash_cad, cash_usd, grand_total_cad, 
            usd_to_cad, comment
        )
        
        execute_query(insert_query, params, commit=True)
        
        return {
            'id': snapshot_id,
            'date': snapshot_date,
            'investments_cad': totals['CAD'],
            'investments_usd': totals['USD'],
            'cash_cad': cash_cad,
            'cash_usd': cash_usd,
            'total_cad': grand_total_cad,
            'exchange_rate': usd_to_cad
        }
    except Exception as e:
        logger.error(f"Error creating portfolio snapshot: {e}")
        return None
    
# Simplified TWRR calculation for portfolio_utils.py

def calculate_twrr_simplified(portfolio, period="3m"):
    """
    Calculate Time-Weighted Rate of Return (TWRR) with simplified approach
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period ('1m', '3m', '6m', '1y', 'all')
        
    Returns:
        dict: TWRR results with normalized series
    """
    # Define date range
    end_date = datetime.now()
    
    if period == "1m": start_date = end_date - timedelta(days=30)
    elif period == "3m": start_date = end_date - timedelta(days=90)
    elif period == "6m": start_date = end_date - timedelta(days=180)
    elif period == "1y": start_date = end_date - timedelta(days=365)
    else:  # "all"
        earliest_date = get_earliest_transaction_date()
        start_date = earliest_date if earliest_date else (end_date - timedelta(days=365*5))
    
    # Get transactions in this period
    transactions = load_transactions(start_date=start_date.strftime('%Y-%m-%d'))
    
    # Get historical data
    from components.portfolio_visualizer import get_portfolio_historical_data
    historical_data = get_portfolio_historical_data(portfolio, period)
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        return {'twrr': 0, 'normalized_series': pd.Series()}
    
    # Get portfolio values series
    portfolio_values = historical_data['Total']
    
    # Sort transactions by date
    sorted_transactions = sorted(
        transactions.values(), 
        key=lambda tx: datetime.strptime(tx['date'], '%Y-%m-%d')
    )
    
    # Group transactions by date and calculate net flows
    flows_by_date = {}
    for tx in sorted_transactions:
        tx_date = datetime.strptime(tx['date'], '%Y-%m-%d')
        if tx_date < start_date or tx_date > end_date:
            continue
            
        tx_amount = float(tx['amount'])
        flow = -tx_amount if tx['type'] == 'buy' else tx_amount
        
        date_key = tx_date.strftime('%Y-%m-%d')
        if date_key not in flows_by_date:
            flows_by_date[date_key] = 0
        flows_by_date[date_key] += flow
    
    # Calculate sub-period returns
    sub_period_returns = []
    last_value = portfolio_values.iloc[0]
    last_date = portfolio_values.index[0]
    
    # Add flows to significant dates
    for date_str, flow in flows_by_date.items():
        flow_date = pd.Timestamp(date_str)
        
        # Find closest date in portfolio values on or after flow date
        try:
            # Get value just before flow
            dates_before = portfolio_values.index[portfolio_values.index < flow_date]
            if not dates_before.empty:
                before_date = dates_before[-1]
                before_value = portfolio_values.loc[before_date]
                
                # Calculate return for period before flow
                if last_value > 0:
                    period_return = before_value / last_value - 1
                    sub_period_returns.append(1 + period_return)
                
                # Adjust for flow
                last_value = before_value + flow
                last_date = before_date
            else:
                # Flow is before first portfolio value
                last_value += flow
        except Exception as e:
            logger.warning(f"Error processing flow on {date_str}: {e}")
    
    # Add final period
    if last_value > 0:
        final_value = portfolio_values.iloc[-1]
        final_return = final_value / last_value - 1
        sub_period_returns.append(1 + final_return)
    
    # Calculate overall TWRR
    if sub_period_returns:
        twrr = np.prod(sub_period_returns) - 1
        twrr_pct = twrr * 100
    else:
        # Fallback to simple return if no sub-periods
        first_value = portfolio_values.iloc[0]
        last_value = portfolio_values.iloc[-1]
        twrr = (last_value / first_value) - 1 if first_value > 0 else 0
        twrr_pct = twrr * 100
    
    # Create normalized series for visualization
    normalized_series = pd.Series(index=portfolio_values.index)
    normalized_series.iloc[0] = 100.0
    
    # Calculate daily returns and apply them to normalized series
    daily_returns = portfolio_values.pct_change().fillna(0)
    normalized_series = (1 + daily_returns).cumprod() * 100
    normalized_series = normalized_series / normalized_series.iloc[0] * 100  # Ensure it starts at exactly 100
    
    return {
        'twrr': twrr_pct, 
        'historical_values': portfolio_values, 
        'normalized_series': normalized_series
    }

def record_cash_flow(flow_type, amount, currency, date=None, description=""):
    """
    Record a cash flow (deposit or withdrawal) in the database.
    
    Args:
        flow_type (str): "deposit" or "withdrawal"
        amount (float): Amount of money
        currency (str): Currency code ("CAD" or "USD")
        date (str): Date in YYYY-MM-DD format (optional, defaults to today)
        description (str): Description/notes (optional)
        
    Returns:
        bool: Success status
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    else:
        # Ensure date is in YYYY-MM-DD format
        try:
            date = datetime.strptime(str(date), '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format for cash flow: {date}. Using today.")
            date = datetime.now().strftime("%Y-%m-%d")
    
    # Generate a unique ID for the cash flow
    import uuid
    flow_id = str(uuid.uuid4())
    
    # Insert cash flow record
    insert_query = """
    INSERT INTO cash_flows (
        id, flow_type, amount, currency, flow_date, description, recorded_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    
    params = (
        flow_id,
        flow_type.lower(),
        float(amount),
        currency.upper(),
        date,
        description,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    result = execute_query(insert_query, params, commit=True)
    
    if result is not None:
        # Update cash position
        multiplier = 1 if flow_type.lower() == "deposit" else -1
        try:
            update_cash_position(currency.upper(), amount * multiplier)
            logger.info(f"Cash flow recorded: {flow_type} of {amount} {currency}")
            return True
        except Exception as e:
            logger.error(f"Failed to update cash position after recording cash flow: {e}")
            return False
    else:
        logger.error(f"Failed to record cash flow: {flow_type} of {amount} {currency}")
        return False

def update_cash_position(currency, amount_change):
    """
    Update a cash position by a specified amount.
    
    Args:
        currency (str): Currency code
        amount_change (float): Amount to change (positive for increase, negative for decrease)
        
    Returns:
        bool: Success status
    """
    # Check if cash position exists
    query = "SELECT * FROM cash_positions WHERE currency = %s;"
    cash_position = execute_query(query, (currency,), fetchone=True)
    
    if cash_position:
        # Update existing position
        current_balance = float(cash_position['balance'])
        new_balance = current_balance + amount_change
        
        update_query = """
        UPDATE cash_positions 
        SET balance = %s, last_updated = %s 
        WHERE currency = %s;
        """
        
        params = (
            new_balance,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            currency
        )
        
        result = execute_query(update_query, params, commit=True)
        return result is not None
    else:
        # Create new position
        insert_query = """
        INSERT INTO cash_positions (currency, balance, last_updated) 
        VALUES (%s, %s, %s);
        """
        
        params = (
            currency,
            amount_change,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        result = execute_query(insert_query, params, commit=True)
        return result is not None

def load_cash_flows(start_date=None, end_date=None):
    """
    Load cash flow records from database.
    
    Args:
        start_date (str): Filter by start date (YYYY-MM-DD, optional)
        end_date (str): Filter by end date (YYYY-MM-DD, optional)
        
    Returns:
        list: Cash flow records sorted by date (newest first)
    """
    query = "SELECT * FROM cash_flows"
    params = []
    filters = []
    
    if start_date:
        filters.append("flow_date >= %s")
        params.append(start_date)
    
    if end_date:
        filters.append("flow_date <= %s")
        params.append(end_date)
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    query += " ORDER BY flow_date DESC, recorded_at DESC;"
    
    cash_flows = execute_query(query, tuple(params) if params else None, fetchall=True)
    
    result = []
    if cash_flows:
        for flow in cash_flows:
            try:
                flow_dict = dict(flow)
                
                # Convert Decimal to float
                if 'amount' in flow_dict and flow_dict['amount'] is not None:
                    flow_dict['amount'] = float(flow_dict['amount'])
                
                # Format dates
                flow_dict['date'] = flow_dict['flow_date'].strftime("%Y-%m-%d")
                flow_dict['flow_date'] = flow_dict['flow_date'].strftime("%Y-%m-%d")
                if flow_dict.get('recorded_at'):
                    flow_dict['recorded_at'] = flow_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
                # Ensure ID is string
                flow_dict['id'] = str(flow_dict['id'])
                
                result.append(flow_dict)
            except Exception as e:
                logger.error(f"Error processing cash flow {flow.get('id')}: {e}")
                continue
    
    return result

def get_cash_flow_summary(period="all"):
    """
    Get summary of cash flows for a specific period.
    
    Args:
        period (str): Time period ('1m', '3m', '6m', '1y', 'all')
        
    Returns:
        dict: Summary of deposits and withdrawals by currency
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
    else:  # all
        start_date = None
    
    # Load cash flows for the period
    cash_flows = load_cash_flows(
        start_date=start_date.strftime("%Y-%m-%d") if start_date else None
    )
    
    # Initialize summary
    summary = {
        "CAD": {"deposits": 0, "withdrawals": 0, "net": 0},
        "USD": {"deposits": 0, "withdrawals": 0, "net": 0}
    }
    
    # Calculate totals
    for flow in cash_flows:
        currency = flow.get('currency', 'CAD')
        flow_type = flow.get('type', '').lower()
        amount = flow.get('amount', 0)
        
        if currency not in summary:
            summary[currency] = {"deposits": 0, "withdrawals": 0, "net": 0}
        
        if flow_type == 'deposit':
            summary[currency]['deposits'] += amount
            summary[currency]['net'] += amount
        elif flow_type == 'withdrawal':
            summary[currency]['withdrawals'] += amount
            summary[currency]['net'] -= amount
    
    return summary