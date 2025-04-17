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
    
    # Calculate book value (important for cost basis)
    book_value = shares_float * price_float
    
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
    gain_loss = current_value - book_value
    gain_loss_percent = ((current_price / price_float) - 1) * 100 if price_float > 0 else 0
    
    # Check if this symbol already exists in portfolio
    existing_query = "SELECT * FROM portfolio WHERE symbol = %s;"
    existing_investments = execute_query(existing_query, (symbol_upper,), fetchall=True)
    
    if existing_investments:
        # When adding more of an existing investment, calculate the weighted average
        # Similar logic to _update_portfolio_after_transaction for buy
        total_shares = shares_float
        total_book_value = book_value
        first_investment = None
        
        for i, inv in enumerate(existing_investments):
            if i == 0:
                first_investment = inv
            
            existing_shares = float(inv['shares'])
            existing_price = float(inv['purchase_price'])
            total_shares += existing_shares
            total_book_value += existing_shares * existing_price
        
        # Calculate weighted average price
        weighted_price = total_book_value / total_shares if total_shares > 0 else price_float
        
        # Calculate new current value and gain/loss
        new_current_value = total_shares * current_price
        new_gain_loss = new_current_value - total_book_value
        new_gain_loss_pct = ((new_current_value / total_book_value) - 1) * 100 if total_book_value > 0 else 0
        
        # Update the first investment with the new totals
        update_query = """
        UPDATE portfolio SET
            shares = %s,
            purchase_price = %s,
            current_price = %s,
            current_value = %s,
            gain_loss = %s,
            gain_loss_percent = %s,
            last_updated = %s
        WHERE id = %s;
        """
        
        params = (
            total_shares,
            weighted_price,
            current_price,
            new_current_value,
            new_gain_loss,
            new_gain_loss_pct,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            first_investment['id']
        )
        
        # If the new purchase date is earlier, update it
        if purchase_date < first_investment['purchase_date'].strftime("%Y-%m-%d"):
            update_query = update_query.replace("last_updated = %s", "purchase_date = %s, last_updated = %s")
            params = params[:-2] + (purchase_date,) + params[-2:]
        
        # If name is provided and existing is blank or just the symbol, update it
        if name and (first_investment.get('name') is None or first_investment.get('name') == symbol_upper):
            update_query = update_query.replace("last_updated = %s", "name = %s, last_updated = %s")
            params = params[:-2] + (name,) + params[-2:]
        
        result = execute_query(update_query, params, commit=True)
        
        # Delete any other investments for this symbol to maintain one consolidated record
        if len(existing_investments) > 1:
            for i, inv in enumerate(existing_investments):
                if i > 0:  # Skip the first one
                    delete_query = "DELETE FROM portfolio WHERE id = %s;"
                    execute_query(delete_query, (inv['id'],), commit=True)
        
        logger.info(f"Updated existing investment {symbol_upper}: new total {total_shares} shares, avg price ${weighted_price:.2f}")
        return result is not None
    else:
        # Insert as a new investment
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
            logger.info(f"New investment added: {symbol_upper}, {shares_float} shares at ${price_float}, book value ${book_value:.2f}")
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

# Update to the _update_portfolio_after_transaction function in modules/portfolio_utils.py
# Replace the existing function with this enhanced version

def _update_portfolio_after_transaction(transaction_type, symbol, price, shares, date, asset_name=None, asset_type="stock"):
    """
    PRIVATE HELPER: Update portfolio based on a transaction.
    Called internally by record_transaction.

    Args:
        transaction_type (str): "buy", "sell", or "drip"
        symbol (str): Asset symbol.
        price (float): Price per share.
        shares (float): Number of shares.
        date (str): Transaction date (YYYY-MM-DD).
        asset_name (str, optional): Name of the asset.
        asset_type (str, optional): Type of asset (default: "stock").
    """
    # Convert inputs to proper types to prevent calculation errors
    price_float = float(price)
    shares_float = float(shares)
    transaction_amount = price_float * shares_float
    
    # Normalize transaction type
    transaction_type = transaction_type.lower()
    
    # Get existing investments for this symbol
    select_query = "SELECT * FROM portfolio WHERE symbol = %s;"
    existing_investments = execute_query(select_query, (symbol,), fetchall=True)
    
    logger.info(f"Processing portfolio update for {transaction_type} of {shares_float} shares of {symbol} at ${price_float}")
    
    # Handle buy and drip transactions similarly - both add shares
    if transaction_type in ["buy", "drip"]:
        if existing_investments:
            # Get the total of all existing positions for this symbol
            total_current_shares = 0
            total_book_value = 0
            first_investment = None
            
            for i, inv in enumerate(existing_investments):
                if i == 0:
                    first_investment = inv  # Save the first one for the update
                
                current_shares = float(inv['shares'])
                purchase_price = float(inv['purchase_price'])
                total_current_shares += current_shares
                total_book_value += current_shares * purchase_price
            
            # Calculate new totals after the transaction
            new_total_shares = total_current_shares + shares_float
            
            # For DRIP, we don't change the average purchase price since it's reinvested dividends
            # For buy, we recalculate the average purchase price
            if transaction_type == "buy":
                # Calculate new average purchase price (weighted average)
                new_total_book_value = total_book_value + transaction_amount
                new_avg_price = new_total_book_value / new_total_shares if new_total_shares > 0 else 0
            else:  # drip
                # For DRIP, average cost basis is reduced since we're adding shares "for free"
                new_total_book_value = total_book_value  # Book value stays the same
                new_avg_price = new_total_book_value / new_total_shares if new_total_shares > 0 else 0
            
            # Get current market price
            current_price = float(first_investment['current_price']) if first_investment.get('current_price') else price_float
            
            # Calculate new current value and gain/loss
            new_current_value = new_total_shares * current_price
            new_gain_loss = new_current_value - new_total_book_value
            new_gain_loss_pct = ((new_current_value / new_total_book_value) - 1) * 100 if new_total_book_value > 0 else 0
            
            # We'll update just the first investment record with the new totals
            # and remove any other records for this symbol
            inv_id = first_investment['id']
            
            # Update the first investment with new totals
            update_query = """
            UPDATE portfolio SET 
                shares = %s, 
                purchase_price = %s, 
                current_value = %s,
                gain_loss = %s, 
                gain_loss_percent = %s, 
                last_updated = %s
            WHERE id = %s;
            """
            
            params = (
                new_total_shares,
                new_avg_price,
                new_current_value,
                new_gain_loss,
                new_gain_loss_pct,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inv_id
            )
            
            # If purchase date is earlier than current, update it (for buy only, not for DRIP)
            if transaction_type == "buy" and first_investment['purchase_date'] and date < first_investment['purchase_date'].strftime("%Y-%m-%d"):
                update_query = """
                UPDATE portfolio SET 
                    shares = %s, 
                    purchase_price = %s, 
                    purchase_date = %s,
                    current_value = %s,
                    gain_loss = %s, 
                    gain_loss_percent = %s, 
                    last_updated = %s
                WHERE id = %s;
                """
                
                params = (
                    new_total_shares,
                    new_avg_price,
                    date,
                    new_current_value,
                    new_gain_loss,
                    new_gain_loss_pct,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    inv_id
                )
            
            # Update name if provided and not already set
            if asset_name and (first_investment.get('name') is None or first_investment.get('name') == symbol):
                update_query = update_query.replace("last_updated = %s", "name = %s, last_updated = %s")
                params = params[:-2] + (asset_name,) + params[-2:]
            
            # Execute the update
            execute_query(update_query, params, commit=True)
            
            # If there were multiple investments for this symbol, delete the others
            if len(existing_investments) > 1:
                for i, inv in enumerate(existing_investments):
                    if i > 0:  # Skip the first one which we updated above
                        delete_query = "DELETE FROM portfolio WHERE id = %s;"
                        execute_query(delete_query, (inv['id'],), commit=True)
            
            logger.info(f"Updated {symbol} portfolio: {total_current_shares} + {shares_float} = {new_total_shares} shares, " +
                       f"new avg price: ${new_avg_price:.2f}, book value: ${new_total_book_value:.2f}")
        else:
            # Add new investment if it doesn't exist
            logger.info(f"Adding new investment {symbol} after {transaction_type}")
            # Use provided asset_name and asset_type, or default
            name_to_use = asset_name if asset_name else symbol
            add_investment(symbol, shares_float, price_float, date, asset_type=asset_type, name=name_to_use)
    
    elif transaction_type == "sell":
        if existing_investments:
            # Get the total of all existing positions for this symbol
            total_current_shares = 0
            total_book_value = 0
            first_investment = None
            
            for i, inv in enumerate(existing_investments):
                if i == 0:
                    first_investment = inv  # Save the first one for the update
                
                current_shares = float(inv['shares'])
                purchase_price = float(inv['purchase_price'])
                total_current_shares += current_shares
                total_book_value += current_shares * purchase_price
            
            # Calculate new totals after the sell
            new_total_shares = total_current_shares - shares_float
            
            # Handle case where selling more shares than owned
            if new_total_shares < 0:
                logger.warning(f"Cannot sell more shares than owned: attempting to sell {shares_float} but only have {total_current_shares}")
                return False
            
            # Handle case where all shares are sold
            if new_total_shares <= 0.000001:  # Use small threshold for float comparison
                logger.info(f"Removing all investments for {symbol} after selling all shares")
                
                # Delete all existing investments for this symbol
                for inv in existing_investments:
                    delete_query = "DELETE FROM portfolio WHERE id = %s;"
                    execute_query(delete_query, (inv['id'],), commit=True)
                
                return True
            
            # Calculate proportion of book value being sold
            proportion_sold = shares_float / total_current_shares
            book_value_sold = total_book_value * proportion_sold
            new_total_book_value = total_book_value - book_value_sold
            
            # Keep the same average purchase price
            current_purchase_price = float(first_investment['purchase_price'])
            
            # Get current market price
            current_price = float(first_investment['current_price']) if first_investment.get('current_price') else price_float
            
            # Calculate new current value and gain/loss
            new_current_value = new_total_shares * current_price
            new_gain_loss = new_current_value - new_total_book_value
            new_gain_loss_pct = ((new_current_value / new_total_book_value) - 1) * 100 if new_total_book_value > 0 else 0
            
            # Update the first investment with new totals
            inv_id = first_investment['id']
            
            update_query = """
            UPDATE portfolio SET 
                shares = %s, 
                current_value = %s,
                gain_loss = %s, 
                gain_loss_percent = %s, 
                last_updated = %s
            WHERE id = %s;
            """
            
            params = (
                new_total_shares,
                new_current_value,
                new_gain_loss,
                new_gain_loss_pct,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inv_id
            )
            
            # Execute the update
            execute_query(update_query, params, commit=True)
            
            # If there were multiple investments for this symbol, delete the others
            if len(existing_investments) > 1:
                for i, inv in enumerate(existing_investments):
                    if i > 0:  # Skip the first one which we updated above
                        delete_query = "DELETE FROM portfolio WHERE id = %s;"
                        execute_query(delete_query, (inv['id'],), commit=True)
            
            logger.info(f"Updated {symbol} portfolio after sell: {total_current_shares} - {shares_float} = {new_total_shares} shares, " +
                       f"Book value reduced from ${total_book_value:.2f} to ${new_total_book_value:.2f}")
        else:
            logger.warning(f"Cannot sell {symbol} because it is not in the portfolio")
            return False
    
    return True




def record_transaction(transaction_type, symbol, price, shares, date=None, notes="", asset_name=None, asset_type=None):
    """
    Record a buy/sell/drip transaction in the database and update the portfolio.
    Also updates cash positions to reflect the transaction.

    Args:
        transaction_type (str): "buy", "sell", or "drip"
        symbol (str): Asset symbol
        price (float): Price per share/unit
        shares (float): Number of shares/units
        date (str): Transaction date (YYYY-MM-DD, optional, defaults to current date)
        notes (str): Transaction notes (optional)
        asset_name (str): Name of the asset (optional, used for new investments)
        asset_type (str): Type of asset (optional, used for new investments)

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
             logger.error(f"Invalid date format for transaction: {date}. Using today.")
             date = datetime.now().strftime("%Y-%m-%d")

    symbol_upper = symbol.upper().strip()
    shares_float = float(shares)
    price_float = float(price)
    amount = price_float * shares_float
    transaction_id = str(uuid.uuid4())

    # Standardize transaction type
    transaction_type = transaction_type.lower()
    
    # Validate transaction type
    valid_types = ["buy", "sell", "drip"]
    if transaction_type not in valid_types:
        logger.error(f"Invalid transaction type: {transaction_type}. Must be one of: {valid_types}")
        return False

    # Determine currency based on symbol or from existing investment
    # Look up the currency from the portfolio first for accuracy
    currency = None
    
    # Check portfolio for this symbol to determine currency
    select_query = "SELECT currency FROM portfolio WHERE symbol = %s LIMIT 1;"
    currency_result = execute_query(select_query, (symbol_upper,), fetchone=True)
    
    if currency_result and currency_result.get('currency'):
        currency = currency_result.get('currency')
    else:
        # Default currency determination based on symbol
        currency = "CAD" if symbol_upper.endswith(".TO") or symbol_upper.endswith(".V") or symbol_upper.startswith("MAW") else "USD"
    
    logger.info(f"Recording {transaction_type} transaction for {symbol_upper}: {shares_float} shares at {price_float} {currency}")
    
    # Insert transaction first
    insert_query = """
    INSERT INTO transactions (
        id, type, symbol, price, shares, amount,
        transaction_date, notes, recorded_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    params = (
        transaction_id, transaction_type, symbol_upper, price_float, shares_float, amount,
        date, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    result = execute_query(insert_query, params, commit=True)

    if result is None:
        logger.error(f"Failed to record transaction in database: {symbol_upper} {transaction_type} {shares_float} shares")
        return False
        
    # Now update cash positions BEFORE updating portfolio
    # Skip cash updates for DRIP transactions
    cash_updated = True
    if transaction_type != "drip":
        cash_updated = track_cash_position(transaction_type, amount, currency)
        if not cash_updated:
            logger.warning(f"Failed to update cash position for {transaction_type} of {symbol_upper} in {currency}")
            # Continue with portfolio update even if cash update fails
    
    # Now update the portfolio
    try:
        # For buy transactions of new assets, use the specified name and asset type
        if transaction_type == "buy" or transaction_type == "drip":
            # Check if this asset exists in portfolio
            select_query = "SELECT * FROM portfolio WHERE symbol = %s;"
            existing_investment = execute_query(select_query, (symbol_upper,), fetchone=True)
            
            if not existing_investment:
                # This is a new investment, add it with the provided name and type
                asset_type_to_use = asset_type if asset_type else "stock"
                add_investment(symbol_upper, shares_float, price_float, date, asset_type_to_use, asset_name)
            else:
                # Existing investment, update with standard function but pass name and type
                _update_portfolio_after_transaction(
                    transaction_type, symbol_upper, price_float, shares_float, 
                    date, asset_name=asset_name, asset_type=asset_type
                )
        else:
            # For sell transactions, use standard update function
            _update_portfolio_after_transaction(transaction_type, symbol_upper, price_float, shares_float, date)
        
        logger.info(f"Transaction recorded and portfolio updated: {transaction_type} {shares_float} shares of {symbol_upper}")
        return True
    except Exception as update_err:
        logger.error(f"Transaction recorded ({transaction_id}), but failed to update portfolio: {update_err}")
        import traceback
        traceback.print_exc()
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
        transaction_type (str): 'buy', 'sell', 'drip', etc.
        amount (float): Transaction amount
        currency (str): Currency code
        
    Returns:
        bool: Success status
    """
    try:
        # Normalize transaction type to lowercase
        transaction_type = transaction_type.lower()
        
        # Skip cash tracking for DRIP transactions
        if transaction_type == 'drip':
            logger.info(f"Skipping cash tracking for DRIP transaction (no cash involved)")
            return True
            
        # Get current cash positions
        query = "SELECT * FROM cash_positions WHERE currency = %s;"
        cash_position = execute_query(query, (currency,), fetchone=True)
        
        # Calculate adjustment:
        # - Buy transactions DECREASE cash (-amount)
        # - Sell transactions INCREASE cash (+amount)
        adjustment = -float(amount) if transaction_type == 'buy' else float(amount)
        
        logger.info(f"Adjusting {currency} cash position by {adjustment} for {transaction_type} transaction")
        
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
            result = execute_query(update_query, params, commit=True)
            
            if result is None:
                logger.error(f"Database error updating cash position: {currency} balance to {new_balance}")
                return False
                
            logger.info(f"Updated {currency} cash position from {current_balance} to {new_balance}")
            return True
        else:
            # Create new position
            insert_query = """
            INSERT INTO cash_positions (currency, balance, last_updated) 
            VALUES (%s, %s, %s);
            """
            params = (currency, adjustment, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            result = execute_query(insert_query, params, commit=True)
            
            if result is None:
                logger.error(f"Database error creating new cash position for {currency} with balance {adjustment}")
                return False
                
            logger.info(f"Created new cash position for {currency} with initial balance {adjustment}")
            return True
    except Exception as e:
        logger.error(f"Error tracking cash position: {e}")
        import traceback
        traceback.print_exc()
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

def record_currency_exchange(from_currency, from_amount, to_currency, to_amount, date=None, description=""):
    """
    Record a currency exchange transaction in the database.
    
    Args:
        from_currency (str): Source currency code ("CAD" or "USD")
        from_amount (float): Amount in source currency
        to_currency (str): Target currency code ("CAD" or "USD")
        to_amount (float): Amount in target currency
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
            logger.error(f"Invalid date format for currency exchange: {date}. Using today.")
            date = datetime.now().strftime("%Y-%m-%d")
    
    # Calculate the exchange rate
    rate = from_amount / to_amount if to_amount > 0 else 0
    
    # Generate a unique ID for the exchange
    exchange_id = str(uuid.uuid4())
    
    # Insert exchange record
    insert_query = """
    INSERT INTO currency_exchanges (
        id, from_currency, from_amount, to_currency, to_amount, rate,
        exchange_date, description, recorded_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    params = (
        exchange_id,
        from_currency.upper(),
        float(from_amount),
        to_currency.upper(),
        float(to_amount),
        float(rate),
        date,
        description,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    result = execute_query(insert_query, params, commit=True)
    
    if result is not None:
        # Update cash positions for both currencies
        try:
            # Subtract from source currency
            update_cash_position(from_currency.upper(), -float(from_amount))
            
            # Add to target currency
            update_cash_position(to_currency.upper(), float(to_amount))
            
            logger.info(f"Currency exchange recorded: {from_amount} {from_currency} to {to_amount} {to_currency}")
            return True
        except Exception as e:
            logger.error(f"Failed to update cash positions after recording currency exchange: {e}")
            return False
    else:
        logger.error(f"Failed to record currency exchange: {from_amount} {from_currency} to {to_amount} {to_currency}")
        return False

def load_currency_exchanges(start_date=None, end_date=None):
    """
    Load currency exchange records from database.
    
    Args:
        start_date (str): Filter by start date (YYYY-MM-DD, optional)
        end_date (str): Filter by end date (YYYY-MM-DD, optional)
        
    Returns:
        list: Currency exchange records sorted by date (newest first)
    """
    query = "SELECT * FROM currency_exchanges"
    params = []
    filters = []
    
    if start_date:
        filters.append("exchange_date >= %s")
        params.append(start_date)
    
    if end_date:
        filters.append("exchange_date <= %s")
        params.append(end_date)
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    query += " ORDER BY exchange_date DESC, recorded_at DESC;"
    
    exchanges = execute_query(query, tuple(params) if params else None, fetchall=True)
    
    result = []
    if exchanges:
        for exchange in exchanges:
            try:
                exchange_dict = dict(exchange)
                
                # Convert Decimal to float
                for key in ['from_amount', 'to_amount', 'rate']:
                    if key in exchange_dict and exchange_dict[key] is not None:
                        exchange_dict[key] = float(exchange_dict[key])
                
                # Format dates
                exchange_dict['date'] = exchange_dict['exchange_date'].strftime("%Y-%m-%d")
                exchange_dict['exchange_date'] = exchange_dict['exchange_date'].strftime("%Y-%m-%d")
                if exchange_dict.get('recorded_at'):
                    exchange_dict['recorded_at'] = exchange_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
                # Ensure ID is string
                exchange_dict['id'] = str(exchange_dict['id'])
                
                result.append(exchange_dict)
            except Exception as e:
                logger.error(f"Error processing currency exchange {exchange.get('id')}: {e}")
                continue
    
    return result

def get_weighted_exchange_rate(from_currency="CAD", to_currency="USD", lookback_days=365):
    """
    Calculate the weighted average exchange rate from historical exchanges.
    
    Args:
        from_currency (str): Source currency code
        to_currency (str): Target currency code
        lookback_days (int): Number of days to look back
        
    Returns:
        float: Weighted average exchange rate or current rate if no data
    """
    # Calculate start date based on lookback days
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Load relevant exchanges
    exchanges = load_currency_exchanges(start_date=start_date)
    
    # Filter exchanges for the specified currency pair
    relevant_exchanges = [
        ex for ex in exchanges 
        if ex.get('from_currency') == from_currency.upper() and ex.get('to_currency') == to_currency.upper()
    ]
    
    if relevant_exchanges:
        # Calculate weighted average
        total_from_amount = sum(ex.get('from_amount', 0) for ex in relevant_exchanges)
        total_to_amount = sum(ex.get('to_amount', 0) for ex in relevant_exchanges)
        
        if total_to_amount > 0:
            weighted_rate = total_from_amount / total_to_amount
            return weighted_rate
    
    # Fallback to current rate if no data
    if from_currency.upper() == "CAD" and to_currency.upper() == "USD":
        # Invert for CAD to USD
        return 1.0 / get_usd_to_cad_rate()
    elif from_currency.upper() == "USD" and to_currency.upper() == "CAD":
        return get_usd_to_cad_rate()
    else:
        # Default fallback
        return 1.0
    
def calculate_total_return(portfolio, period="1y", include_dividends=True):
    """
    Calculate total return for the portfolio including price appreciation and dividends.
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period ('1m', '3m', '6m', '1y', 'all')
        include_dividends (bool): Whether to include dividend income in return calculation
        
    Returns:
        dict: Total return information
    """
    # Import twrr calculation
    from modules.portfolio_utils import calculate_twrr_simplified
    
    # Get time-weighted return (price appreciation only)
    twrr_data = calculate_twrr_simplified(portfolio, period)
    price_return_pct = twrr_data.get('twrr', 0)
    
    if not include_dividends:
        return {
            "price_return_pct": price_return_pct,
            "dividend_return_pct": 0,
            "total_return_pct": price_return_pct,
            "includes_dividends": False
        }
    
    # Define date range based on period
    end_date = datetime.now()
    
    if period == "1m":
        start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")
    elif period == "3m":
        start_date = (end_date - timedelta(days=90)).strftime("%Y-%m-%d")
    elif period == "6m":
        start_date = (end_date - timedelta(days=180)).strftime("%Y-%m-%d")
    elif period == "1y":
        start_date = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")
    else:  # "all"
        earliest_date = get_earliest_transaction_date()
        start_date = earliest_date.strftime("%Y-%m-%d") if earliest_date else (end_date - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    end_date = end_date.strftime("%Y-%m-%d")
    
    # Calculate portfolio value at start of period
    # This is a simplification - ideally would use actual portfolio value at start date
    portfolio_value_start = sum((float(inv.get("shares", 0)) * float(inv.get("purchase_price", 0))) for inv in portfolio.values())
    if portfolio_value_start <= 0:
        return {
            "price_return_pct": price_return_pct,
            "dividend_return_pct": 0,
            "total_return_pct": price_return_pct,
            "includes_dividends": True
        }
    
    # Get dividend data for period
    from modules.dividend_utils import load_dividends
    dividends = load_dividends(start_date=start_date, end_date=end_date)
    
    # Calculate dividend return
    total_cad_dividends = sum(div["total_amount"] for div in dividends if div["currency"] == "CAD")
    total_usd_dividends = sum(div["total_amount"] for div in dividends if div["currency"] == "USD")
    
    # Convert USD dividends to CAD
    usd_to_cad_rate = get_usd_to_cad_rate()
    total_dividends_cad = total_cad_dividends + (total_usd_dividends * usd_to_cad_rate)
    
    # Calculate dividend return as percentage of starting portfolio value
    dividend_return_pct = (total_dividends_cad / portfolio_value_start) * 100
    
    # Calculate total return (price + dividends)
    total_return_pct = price_return_pct + dividend_return_pct
    
    return {
        "price_return_pct": price_return_pct,
        "dividend_return_pct": dividend_return_pct,
        "total_return_pct": total_return_pct,
        "includes_dividends": True,
        "total_dividends_cad": total_dividends_cad,
        "portfolio_value_start": portfolio_value_start
    }

def rebuild_portfolio_from_transactions():
    """
    Rebuilds the entire portfolio from transaction history to ensure
    accurate book values and share counts.
    
    This function:
    1. Gets all transactions in chronological order
    2. Rebuilds portfolio positions transaction by transaction
    3. Updates the portfolio database with corrected values
    4. Returns a report of changes made
    
    Returns:
        dict: Report of the rebuild process
    """
    logger.info("Starting portfolio rebuild from transaction history")
    
    # Load all transactions
    all_transactions = load_transactions()
    
    # Sort transactions chronologically 
    sorted_transactions = sorted(
        all_transactions.values(), 
        key=lambda tx: datetime.strptime(tx['date'], '%Y-%m-%d')
    )
    
    logger.info(f"Found {len(sorted_transactions)} transactions to process")
    
    # Track positions as we build them
    rebuilt_positions = {}
    rebuild_log = []
    
    # Process each transaction chronologically
    for tx in sorted_transactions:
        symbol = tx['symbol'].upper().strip()
        tx_type = tx['type'].lower()
        tx_date = tx['date']
        tx_shares = float(tx['shares'])
        tx_price = float(tx['price'])
        tx_amount = tx_shares * tx_price
        
        # Initialize position if it doesn't exist
        if symbol not in rebuilt_positions:
            rebuilt_positions[symbol] = {
                'shares': 0.0,
                'book_value': 0.0,
                'last_date': None,
                'asset_type': None,
                'currency': None,
                'name': None,
                'transactions': []
            }
        
        position = rebuilt_positions[symbol]
        
        # Process based on transaction type
        if tx_type == 'buy':
            # For buy: add shares and increase book value
            prev_shares = position['shares']
            prev_book_value = position['book_value']
            
            position['shares'] += tx_shares
            position['book_value'] += tx_amount
            
            # Update other details if this is the first transaction
            if not position['last_date'] or tx_date < position['last_date']:
                position['last_date'] = tx_date
            
            # Get currency from transaction or determine from symbol
            if tx.get('currency'):
                position['currency'] = tx['currency']
            elif position['currency'] is None:
                position['currency'] = "CAD" if symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol else "USD"
                
            # Store transaction in log
            rebuild_log.append({
                'symbol': symbol,
                'date': tx_date,
                'action': 'buy',
                'shares_change': tx_shares,
                'book_value_change': tx_amount,
                'new_shares': position['shares'],
                'new_book_value': position['book_value']
            })
            
        elif tx_type == 'sell':
            # For sell: remove shares and reduce book value proportionally
            if position['shares'] <= 0:
                logger.warning(f"Cannot sell {tx_shares} shares of {symbol} on {tx_date} - position shows zero shares")
                rebuild_log.append({
                    'symbol': symbol,
                    'date': tx_date,
                    'action': 'sell_error',
                    'error': 'Attempting to sell from zero position',
                    'shares_change': 0,
                    'book_value_change': 0,
                    'new_shares': position['shares'],
                    'new_book_value': position['book_value']
                })
                continue
                
            if tx_shares > position['shares']:
                logger.warning(f"Cannot sell {tx_shares} shares of {symbol} on {tx_date} - only {position['shares']} shares available")
                rebuild_log.append({
                    'symbol': symbol,
                    'date': tx_date,
                    'action': 'sell_error',
                    'error': 'Selling more shares than owned',
                    'shares_requested': tx_shares,
                    'shares_available': position['shares'],
                    'new_shares': position['shares'],
                    'new_book_value': position['book_value']
                })
                continue
            
            # Calculate proportion of position being sold
            proportion_sold = tx_shares / position['shares']
            book_value_sold = position['book_value'] * proportion_sold
            
            # Update position
            prev_shares = position['shares']
            prev_book_value = position['book_value']
            
            position['shares'] -= tx_shares
            position['book_value'] -= book_value_sold
            
            # Handle small floating point errors - if shares are very close to zero, set to exactly zero
            if abs(position['shares']) < 0.000001:
                position['shares'] = 0.0
                position['book_value'] = 0.0
            
            # Store transaction in log
            rebuild_log.append({
                'symbol': symbol,
                'date': tx_date,
                'action': 'sell',
                'shares_change': -tx_shares,
                'book_value_change': -book_value_sold,
                'proportion_sold': proportion_sold,
                'new_shares': position['shares'],
                'new_book_value': position['book_value']
            })
            
        elif tx_type == 'drip':
            # For DRIP: add shares but don't change book value (reinvested dividends)
            prev_shares = position['shares']
            prev_book_value = position['book_value']
            
            position['shares'] += tx_shares
            # Book value stays the same for DRIP transactions
            
            # Store transaction in log
            rebuild_log.append({
                'symbol': symbol,
                'date': tx_date,
                'action': 'drip',
                'shares_change': tx_shares,
                'book_value_change': 0,
                'new_shares': position['shares'],
                'new_book_value': position['book_value']
            })
            
        else:
            # Unrecognized transaction type
            logger.warning(f"Unrecognized transaction type: {tx_type} for {symbol} on {tx_date}")
            rebuild_log.append({
                'symbol': symbol,
                'date': tx_date,
                'action': 'unknown',
                'error': f'Unrecognized transaction type: {tx_type}',
                'shares_change': 0,
                'book_value_change': 0
            })
            continue
            
        # Save the transaction in the position's history
        position['transactions'].append({
            'date': tx_date,
            'type': tx_type,
            'shares': tx_shares,
            'price': tx_price,
            'amount': tx_amount,
            'running_shares': position['shares'],
            'running_book_value': position['book_value'],
            'running_avg_cost': position['book_value'] / position['shares'] if position['shares'] > 0 else 0
        })
    
    # Get current portfolio to compare
    current_portfolio = load_portfolio()
    
    # Now update the portfolio database with rebuilt positions
    changes_made = 0
    positions_added = 0
    positions_removed = 0
    positions_updated = 0
    
    # Remove positions with zero shares
    for symbol in list(rebuilt_positions.keys()):
        if rebuilt_positions[symbol]['shares'] <= 0:
            del rebuilt_positions[symbol]
            
    # Track all changes to be made
    positions_to_update = []  # existing positions to update
    positions_to_add = []     # new positions to add
    positions_to_remove = []  # current positions to remove
    
    # Find positions to update and missing positions to add
    for symbol, position in rebuilt_positions.items():
        if position['shares'] <= 0:
            continue  # Skip positions with no shares
            
        # Look for this symbol in current portfolio
        existing_id = None
        
        for inv_id, details in current_portfolio.items():
            if details['symbol'].upper() == symbol:
                existing_id = inv_id
                break
        
        # Get additional details for this position if needed
        if not position.get('asset_type'):
            # Look up in existing portfolio first
            if existing_id:
                position['asset_type'] = current_portfolio[existing_id].get('asset_type', 'stock')
            else:
                # Try looking up in tracked assets
                tracked_assets = load_tracked_assets()
                if symbol in tracked_assets:
                    position['asset_type'] = tracked_assets[symbol].get('type', 'stock')
                    position['name'] = tracked_assets[symbol].get('name', symbol)
                else:
                    position['asset_type'] = 'stock'
        
        if not position.get('name'):
            # Look up in existing portfolio first
            if existing_id:
                position['name'] = current_portfolio[existing_id].get('name', symbol)
            else:
                position['name'] = symbol
                
        # Get current price for calculating gain/loss
        current_price = 0
        
        if existing_id:
            # Use current price from existing position
            current_price = float(current_portfolio[existing_id].get('current_price', 0))
        else:
            # Try to get current price from API
            try:
                # First try using DataProvider
                from modules.data_provider import data_provider
                quote = data_provider.get_current_quote(symbol)
                if quote and 'price' in quote:
                    current_price = float(quote['price'])
                else:
                    # Fallback to purchase price from last transaction
                    if position['transactions']:
                        current_price = position['transactions'][-1]['price']
                    else:
                        current_price = 0
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                if position['transactions']:
                    current_price = position['transactions'][-1]['price']
                else:
                    current_price = 0
        
        # Calculate current value and gain/loss
        current_value = position['shares'] * current_price
        gain_loss = current_value - position['book_value']
        gain_loss_percent = ((current_price / (position['book_value'] / position['shares'])) - 1) * 100 if position['shares'] > 0 and position['book_value'] > 0 else 0
        
        # If position exists, update it
        if existing_id:
            positions_to_update.append({
                'id': existing_id,
                'shares': position['shares'],
                'purchase_price': position['book_value'] / position['shares'] if position['shares'] > 0 else 0,
                'current_price': current_price,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent,
                'purchase_date': position['last_date'],
                'asset_type': position['asset_type'],
                'currency': position['currency'],
                'name': position['name'],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # Add new position
            positions_to_add.append({
                'symbol': symbol,
                'shares': position['shares'],
                'purchase_price': position['book_value'] / position['shares'] if position['shares'] > 0 else 0,
                'current_price': current_price,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent,
                'purchase_date': position['last_date'],
                'asset_type': position['asset_type'],
                'currency': position['currency'],
                'name': position['name'],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Find positions to remove (those in current portfolio but not in rebuilt positions)
    current_symbols = {details['symbol'].upper() for _, details in current_portfolio.items()}
    rebuilt_symbols = set(rebuilt_positions.keys())
    
    for inv_id, details in current_portfolio.items():
        if details['symbol'].upper() not in rebuilt_symbols:
            positions_to_remove.append(inv_id)
    
    # Now execute all the changes in a transaction if database supports it
    # First remove positions
    for inv_id in positions_to_remove:
        remove_query = "DELETE FROM portfolio WHERE id = %s;"
        result = execute_query(remove_query, (inv_id,), commit=True)
        
        if result is not None:
            positions_removed += 1
            logger.info(f"Removed position {current_portfolio[inv_id]['symbol']} (ID: {inv_id})")
    
    # Then update existing positions
    for position in positions_to_update:
        update_query = """
        UPDATE portfolio SET
            shares = %s,
            purchase_price = %s,
            current_price = %s,
            current_value = %s,
            gain_loss = %s,
            gain_loss_percent = %s,
            purchase_date = %s,
            asset_type = %s,
            currency = %s,
            name = %s,
            last_updated = %s
        WHERE id = %s;
        """
        
        params = (
            position['shares'],
            position['purchase_price'],
            position['current_price'],
            position['current_value'],
            position['gain_loss'],
            position['gain_loss_percent'],
            position['purchase_date'],
            position['asset_type'],
            position['currency'],
            position['name'],
            position['last_updated'],
            position['id']
        )
        
        result = execute_query(update_query, params, commit=True)
        
        if result is not None:
            positions_updated += 1
            changes_made += 1
            logger.info(f"Updated position for {current_portfolio[position['id']]['symbol']} (ID: {position['id']})")
    
    # Finally add new positions
    for position in positions_to_add:
        # Generate new ID
        new_id = str(uuid.uuid4())
        
        insert_query = """
        INSERT INTO portfolio (
            id, symbol, shares, purchase_price, purchase_date, asset_type,
            current_price, current_value, gain_loss, gain_loss_percent, 
            currency, name, added_date, last_updated
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        params = (
            new_id,
            position['symbol'],
            position['shares'],
            position['purchase_price'],
            position['purchase_date'],
            position['asset_type'],
            position['current_price'],
            position['current_value'],
            position['gain_loss'],
            position['gain_loss_percent'],
            position['currency'],
            position['name'],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            position['last_updated']
        )
        
        result = execute_query(insert_query, params, commit=True)
        
        if result is not None:
            positions_added += 1
            changes_made += 1
            logger.info(f"Added new position for {position['symbol']} (ID: {new_id})")
    
    # Return a report of the rebuild process
    return {
        'status': 'success',
        'changes_made': changes_made,
        'positions_updated': positions_updated,
        'positions_added': positions_added,
        'positions_removed': positions_removed,
        'rebuilt_positions': {symbol: {k: v for k, v in pos.items() if k != 'transactions'} 
                             for symbol, pos in rebuilt_positions.items() if pos['shares'] > 0},
        'transaction_log': rebuild_log,
        'rebuild_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def debug_cash_positions():
    """
    Get detailed information about cash positions for debugging purposes.
    
    Returns:
        dict: Debug information about cash positions
    """
    try:
        # Get raw cash positions data
        query = "SELECT * FROM cash_positions;"
        positions = execute_query(query, fetchall=True)
        
        # Get recent cash-related transactions
        tx_query = """
        SELECT * FROM transactions 
        ORDER BY transaction_date DESC, recorded_at DESC
        LIMIT 10;
        """
        recent_transactions = execute_query(tx_query, fetchall=True)
        
        # Get recent cash flows
        flow_query = """
        SELECT * FROM cash_flows 
        ORDER BY flow_date DESC, recorded_at DESC
        LIMIT 10;
        """
        recent_flows = execute_query(flow_query, fetchall=True)
        
        # Get recent currency exchanges
        exchange_query = """
        SELECT * FROM currency_exchanges 
        ORDER BY exchange_date DESC, recorded_at DESC
        LIMIT 10;
        """
        recent_exchanges = execute_query(exchange_query, fetchall=True)
        
        # Format the data for easier viewing
        formatted_positions = []
        if positions:
            for pos in positions:
                formatted_positions.append({
                    'id': pos['id'],
                    'currency': pos['currency'],
                    'balance': float(pos['balance']),
                    'last_updated': pos['last_updated'].strftime("%Y-%m-%d %H:%M:%S") if pos.get('last_updated') else None
                })
        
        formatted_transactions = []
        if recent_transactions:
            for tx in recent_transactions:
                formatted_transactions.append({
                    'id': tx['id'],
                    'type': tx['type'],
                    'symbol': tx['symbol'],
                    'shares': float(tx['shares']),
                    'price': float(tx['price']),
                    'amount': float(tx['amount']),
                    'date': tx['transaction_date'].strftime("%Y-%m-%d") if tx.get('transaction_date') else None,
                })
        
        formatted_flows = []
        if recent_flows:
            for flow in recent_flows:
                formatted_flows.append({
                    'id': flow['id'],
                    'type': flow['flow_type'],
                    'amount': float(flow['amount']),
                    'currency': flow['currency'],
                    'date': flow['flow_date'].strftime("%Y-%m-%d") if flow.get('flow_date') else None,
                })
        
        formatted_exchanges = []
        if recent_exchanges:
            for ex in recent_exchanges:
                formatted_exchanges.append({
                    'id': ex['id'],
                    'from_currency': ex['from_currency'],
                    'from_amount': float(ex['from_amount']),
                    'to_currency': ex['to_currency'], 
                    'to_amount': float(ex['to_amount']),
                    'rate': float(ex['rate']),
                    'date': ex['exchange_date'].strftime("%Y-%m-%d") if ex.get('exchange_date') else None,
                })
        
        return {
            'cash_positions': formatted_positions,
            'recent_transactions': formatted_transactions,
            'recent_cash_flows': formatted_flows,
            'recent_exchanges': formatted_exchanges,
            'debug_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error in debug_cash_positions: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
    
def debug_book_value():
    """
    Get detailed information about book value calculations for debugging purposes.
    
    Returns:
        dict: Debug information about book values and transactions
    """
    try:
        # Get portfolio data
        portfolio = load_portfolio()
        
        # Get transaction history
        transactions = load_transactions()
        
        # Calculate expected book values based on transactions
        expected_book_values = {}
        
        for tx_id, tx in transactions.items():
            symbol = tx['symbol'].upper().strip()
            tx_type = tx['type'].lower()
            shares = float(tx['shares'])
            price = float(tx['price'])
            
            if symbol not in expected_book_values:
                expected_book_values[symbol] = {
                    'shares': 0,
                    'book_value': 0,
                    'transactions': []
                }
            
            if tx_type == 'buy':
                # Add to position
                expected_book_values[symbol]['shares'] += shares
                expected_book_values[symbol]['book_value'] += shares * price
                expected_book_values[symbol]['transactions'].append({
                    'date': tx['date'],
                    'type': tx_type,
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'running_shares': expected_book_values[symbol]['shares'],
                    'running_book_value': expected_book_values[symbol]['book_value']
                })
            elif tx_type == 'sell':
                # Calculate proportion of book value being sold
                if expected_book_values[symbol]['shares'] > 0:
                    proportion_sold = shares / expected_book_values[symbol]['shares']
                    book_value_sold = expected_book_values[symbol]['book_value'] * proportion_sold
                    expected_book_values[symbol]['shares'] -= shares
                    expected_book_values[symbol]['book_value'] -= book_value_sold
                    
                    # If shares become very small or negative, reset to 0 to avoid floating point issues
                    if expected_book_values[symbol]['shares'] <= 0.000001:
                        expected_book_values[symbol]['shares'] = 0
                        expected_book_values[symbol]['book_value'] = 0
                        
                    expected_book_values[symbol]['transactions'].append({
                        'date': tx['date'],
                        'type': tx_type,
                        'shares': shares,
                        'price': price,
                        'amount': shares * price,
                        'book_value_sold': book_value_sold,
                        'running_shares': expected_book_values[symbol]['shares'],
                        'running_book_value': expected_book_values[symbol]['book_value']
                    })
            elif tx_type == 'drip':
                # Add shares but no additional book value for DRIP
                expected_book_values[symbol]['shares'] += shares
                expected_book_values[symbol]['transactions'].append({
                    'date': tx['date'],
                    'type': tx_type,
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'running_shares': expected_book_values[symbol]['shares'],
                    'running_book_value': expected_book_values[symbol]['book_value']
                })
                
        # Compare with current portfolio
        for symbol, expected in expected_book_values.items():
            # Skip symbols with zero shares
            if expected['shares'] <= 0.000001:
                continue
                
            # Calculate expected average cost
            expected_avg_cost = expected['book_value'] / expected['shares'] if expected['shares'] > 0 else 0
            
            # Find in portfolio
            actual_in_portfolio = False
            actual_shares = 0
            actual_avg_cost = 0
            actual_book_value = 0
            
            for inv_id, details in portfolio.items():
                if details['symbol'].upper().strip() == symbol:
                    actual_in_portfolio = True
                    actual_shares = float(details['shares'])
                    actual_avg_cost = float(details['purchase_price'])
                    actual_book_value = actual_shares * actual_avg_cost
                    break
            
            # Set comparison results
            expected['in_portfolio'] = actual_in_portfolio
            expected['actual_shares'] = actual_shares
            expected['actual_avg_cost'] = actual_avg_cost
            expected['actual_book_value'] = actual_book_value
            expected['expected_avg_cost'] = expected_avg_cost
            expected['shares_diff'] = actual_shares - expected['shares']
            expected['book_value_diff'] = actual_book_value - expected['book_value']
            expected['avg_cost_diff'] = actual_avg_cost - expected_avg_cost
            
            # Calculate percentage differences
            if expected['shares'] > 0:
                expected['shares_diff_pct'] = (actual_shares / expected['shares'] - 1) * 100
            else:
                expected['shares_diff_pct'] = 0
                
            if expected['book_value'] > 0:
                expected['book_value_diff_pct'] = (actual_book_value / expected['book_value'] - 1) * 100
            else:
                expected['book_value_diff_pct'] = 0
                
            if expected_avg_cost > 0:
                expected['avg_cost_diff_pct'] = (actual_avg_cost / expected_avg_cost - 1) * 100
            else:
                expected['avg_cost_diff_pct'] = 0
                
            # Classify the discrepancy
            if abs(expected['book_value_diff']) < 0.01 and abs(expected['shares_diff']) < 0.0001:
                expected['status'] = 'ok'
            elif abs(expected['book_value_diff_pct']) < 1 and abs(expected['shares_diff_pct']) < 1:
                expected['status'] = 'minor_discrepancy'
            else:
                expected['status'] = 'major_discrepancy'
        
        # Find investments in portfolio but not in transaction history
        extra_investments = []
        for inv_id, details in portfolio.items():
            symbol = details['symbol'].upper().strip()
            if symbol not in expected_book_values or expected_book_values[symbol]['shares'] <= 0:
                extra_investments.append({
                    'id': inv_id,
                    'symbol': symbol,
                    'shares': float(details['shares']),
                    'purchase_price': float(details['purchase_price']),
                    'book_value': float(details['shares']) * float(details['purchase_price']),
                    'status': 'no_transactions'
                })
        
        return {
            'book_values': expected_book_values,
            'extra_investments': extra_investments,
            'debug_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error in debug_book_value: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}