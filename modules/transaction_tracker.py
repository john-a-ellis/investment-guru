# modules/transaction_tracker.py
"""
This module is now deprecated. All transaction recording and loading logic
has been moved to modules/portfolio_utils.py.
"""
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.warning("modules.transaction_tracker is deprecated. Use functions from modules.portfolio_utils instead.")

# You can optionally leave stub functions that raise an error or log a warning
# def record_transaction(*args, **kwargs):
#     raise DeprecationWarning("record_transaction moved to portfolio_utils.py")

# def load_transactions(*args, **kwargs):
#     raise DeprecationWarning("load_transactions moved to portfolio_utils.py")

# It's generally better to remove the file entirely if it's truly empty and unused.
# For now, leaving the deprecation warning.

# # modules/transaction_tracker.py
# """
# Transaction tracker module for the Investment Recommendation System.
# Handles recording and retrieving transaction data from PostgreSQL database.
# """
# from datetime import datetime
# import uuid
# import logging
# from modules.db_utils import execute_query

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def record_transaction(symbol, transaction_type, shares, price, date=None, notes=""):
#     """
#     Record a buy/sell transaction in the database
    
#     Args:
#         symbol (str): Asset symbol
#         transaction_type (str): "buy" or "sell"
#         shares (float): Number of shares/units
#         price (float): Price per share/unit
#         date (str): Transaction date (optional, defaults to current date)
#         notes (str): Transaction notes (optional)
        
#     Returns:
#         bool: Success status
#     """
#     if date is None:
#         date = datetime.now().strftime("%Y-%m-%d")
    
#     # Standardize symbol format
#     symbol = symbol.upper().strip()
        
#     # Convert values to Python float to ensure type consistency
#     shares_float = float(shares)
#     price_float = float(price)
    
#     # Calculate total amount
#     amount = price_float * shares_float
    
#     # Generate a UUID for the transaction
#     transaction_id = str(uuid.uuid4())
    
#     # Insert transaction into database
#     insert_query = """
#     INSERT INTO transactions (
#         id, type, symbol, price, shares, amount, 
#         transaction_date, notes, recorded_at
#     ) VALUES (
#         %s, %s, %s, %s, %s, %s, %s, %s, %s
#     );
#     """
    
#     params = (
#         transaction_id,
#         transaction_type.lower(),
#         symbol,
#         price_float,
#         shares_float,
#         amount,
#         date,
#         notes,
#         datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     )
    
#     result = execute_query(insert_query, params, commit=True)
    
#     if result is not None:
#         # Update portfolio based on transaction
#         update_portfolio_for_transaction(transaction_type, symbol, price_float, shares_float, date)
#         logger.info(f"Transaction recorded successfully: {transaction_type} {shares_float} shares of {symbol}")
#         return True
    
#     logger.error(f"Failed to record transaction: {symbol} {transaction_type} {shares_float} shares")
#     return False

# def update_portfolio_for_transaction(transaction_type, symbol, price, shares, date):
#     """
#     Update portfolio based on a transaction
    
#     Args:
#         transaction_type (str): "buy" or "sell"
#         symbol (str): Asset symbol
#         price (float): Price per share
#         shares (float): Number of shares
#         date (str): Transaction date
#     """
#     # Get current portfolio entry for this symbol
#     select_query = "SELECT * FROM portfolio WHERE symbol = %s;"
#     existing_investment = execute_query(select_query, (symbol,), fetchone=True)
    
#     if transaction_type.lower() == "buy":
#         # Buy transaction
#         if existing_investment:
#             # Update existing investment
#             current_shares = float(existing_investment['shares'])
#             current_price = float(existing_investment['current_price']) if existing_investment['current_price'] else float(price)
            
#             # Add new shares
#             new_shares = current_shares + float(shares)
            
#             # Calculate new average purchase price (weighted average)
#             current_cost = current_shares * float(existing_investment['purchase_price'])
#             new_cost = float(shares) * float(price)
#             new_avg_price = (current_cost + new_cost) / new_shares if new_shares > 0 else 0
            
#             # Calculate new current value
#             current_value = new_shares * current_price
            
#             # Calculate gain/loss
#             gain_loss = current_value - (new_shares * new_avg_price)
#             gain_loss_percent = (current_value / (new_shares * new_avg_price) - 1) * 100 if new_avg_price > 0 else 0
            
#             # Update investment
#             update_query = """
#             UPDATE portfolio SET
#                 shares = %s,
#                 purchase_price = %s,
#                 current_value = %s,
#                 gain_loss = %s,
#                 gain_loss_percent = %s,
#                 last_updated = %s
#             WHERE id = %s;
#             """
            
#             params = (
#                 new_shares,
#                 new_avg_price,
#                 current_value,
#                 gain_loss,
#                 gain_loss_percent,
#                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 existing_investment['id']
#             )
            
#             # If purchase date is earlier than current, update it
#             if date < existing_investment['purchase_date'].strftime("%Y-%m-%d"):
#                 update_query = """
#                 UPDATE portfolio SET
#                     shares = %s,
#                     purchase_price = %s,
#                     purchase_date = %s,
#                     current_value = %s,
#                     gain_loss = %s,
#                     gain_loss_percent = %s,
#                     last_updated = %s
#                 WHERE id = %s;
#                 """
                
#                 params = (
#                     new_shares,
#                     new_avg_price,
#                     date,
#                     current_value,
#                     gain_loss,
#                     gain_loss_percent,
#                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     existing_investment['id']
#                 )
            
#             execute_query(update_query, params, commit=True)
#         else:
#             # Add new investment
#             add_investment(symbol, shares, price, date)
    
#     elif transaction_type.lower() == "sell":
#         # Sell transaction
#         if existing_investment:
#             # Update existing investment
#             current_shares = float(existing_investment['shares'])
            
#             # Remove sold shares
#             new_shares = current_shares - float(shares)
            
#             if new_shares <= 0:
#                 # If all shares sold, remove investment
#                 delete_query = "DELETE FROM portfolio WHERE id = %s;"
#                 execute_query(delete_query, (existing_investment['id'],), commit=True)
#             else:
#                 # Update shares and current value
#                 current_price = float(existing_investment['current_price']) if existing_investment['current_price'] else float(price)
#                 current_value = new_shares * current_price
                
#                 # Recalculate gain/loss
#                 purchase_cost = new_shares * float(existing_investment['purchase_price'])
#                 gain_loss = current_value - purchase_cost
#                 gain_loss_percent = (current_value / purchase_cost - 1) * 100 if purchase_cost > 0 else 0
                
#                 update_query = """
#                 UPDATE portfolio SET
#                     shares = %s,
#                     current_value = %s,
#                     gain_loss = %s,
#                     gain_loss_percent = %s,
#                     last_updated = %s
#                 WHERE id = %s;
#                 """
                
#                 params = (
#                     new_shares,
#                     current_value,
#                     gain_loss,
#                     gain_loss_percent,
#                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     existing_investment['id']
#                 )
                
#                 execute_query(update_query, params, commit=True)

# def add_investment(symbol, shares, purchase_price, purchase_date, asset_type="stock"):
#     """
#     Add a new investment to the portfolio database
    
#     Args:
#         symbol (str): Investment symbol
#         shares (float): Number of shares
#         purchase_price (float): Purchase price per share
#         purchase_date (str): Purchase date in YYYY-MM-DD format
#         asset_type (str): Type of asset (stock, etf, mutual_fund, etc.)
        
#     Returns:
#         bool: Success status
#     """
#     import yfinance as yf
#     from modules.mutual_fund_provider import MutualFundProvider
    
#     # Generate unique ID for this investment
#     investment_id = str(uuid.uuid4())
    
#     # Calculate initial values
#     initial_value = float(shares) * float(purchase_price)
    
#     # Default current price to purchase price
#     current_price = float(purchase_price)
    
#     # For mutual funds, try to get price from our provider
#     if asset_type == "mutual_fund":
#         # Import here to avoid circular imports
#         mutual_fund_provider = MutualFundProvider()
        
#         # Get the most recent price
#         fund_price = mutual_fund_provider.get_current_price(symbol)
#         if fund_price:
#             current_price = fund_price
            
#         # Always treat mutual funds as CAD
#         currency = "CAD"
#     else:
#         # Determine currency based on symbol for non-mutual fund investments
#         is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
#         currency = "CAD" if is_canadian else "USD"
        
#         # Find if we already have this symbol in our portfolio for the current price
#         select_query = "SELECT current_price FROM portfolio WHERE symbol = %s LIMIT 1;"
#         existing_price = execute_query(select_query, (symbol,), fetchone=True)
        
#         if existing_price and existing_price['current_price']:
#             current_price = float(existing_price['current_price'])
        
#         # If we don't have an existing price, try to get it from the API
#         if current_price == float(purchase_price) and asset_type != "mutual_fund":
#             try:
#                 ticker = yf.Ticker(symbol)
#                 price_data = ticker.history(period="1d")
                
#                 if not price_data.empty:
#                     current_price = price_data['Close'].iloc[-1]
#             except Exception as e:
#                 logger.error(f"Error getting initial price for {symbol}: {e}")
    
#     # Calculate current value and gain/loss
#     current_value = float(shares) * current_price
#     gain_loss = current_value - initial_value
#     gain_loss_percent = ((current_price / float(purchase_price)) - 1) * 100 if float(purchase_price) > 0 else 0
    
#     # Insert into portfolio table
#     insert_query = """
#     INSERT INTO portfolio (
#         id, symbol, shares, purchase_price, purchase_date, asset_type,
#         current_price, current_value, gain_loss, gain_loss_percent, 
#         currency, added_date, last_updated
#     ) VALUES (
#         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
#     );
#     """
    
#     params = (
#         investment_id,
#         symbol,
#         float(shares),
#         float(purchase_price),
#         purchase_date,
#         asset_type,
#         current_price,
#         current_value,
#         gain_loss,
#         gain_loss_percent,
#         currency,
#         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     )
    
#     result = execute_query(insert_query, params, commit=True)
    
#     if result is not None:
#         logger.info(f"Investment added successfully: {symbol} {shares} shares")
#         return True
    
#     logger.error(f"Failed to add investment: {symbol}")
#     return False

# def remove_investment(investment_id):
#     """
#     Remove an investment from the portfolio
    
#     Args:
#         investment_id (str): ID of investment to remove
        
#     Returns:
#         bool: Success status
#     """
#     # Delete from portfolio
#     delete_query = "DELETE FROM portfolio WHERE id = %s;"
#     result = execute_query(delete_query, (investment_id,), commit=True)
    
#     if result is not None:
#         logger.info(f"Investment removed successfully: {investment_id}")
#         return True
    
#     logger.error(f"Failed to remove investment: {investment_id}")
#     return False

# def load_transactions(symbol=None, start_date=None, end_date=None):
#     """
#     Load transaction records from database
    
#     Args:
#         symbol (str): Filter by symbol (optional)
#         start_date (str): Filter by start date (optional)
#         end_date (str): Filter by end date (optional)
        
#     Returns:
#         dict: Transaction records
#     """
#     query = "SELECT * FROM transactions"
#     params = []
    
#     # Add filters
#     filters = []
#     if symbol:
#         filters.append("symbol = %s")
#         params.append(symbol)
    
#     if start_date:
#         filters.append("transaction_date >= %s")
#         params.append(start_date)
    
#     if end_date:
#         filters.append("transaction_date <= %s")
#         params.append(end_date)
    
#     # Add WHERE clause if there are filters
#     if filters:
#         query += " WHERE " + " AND ".join(filters)
    
#     # Order by date
#     query += " ORDER BY transaction_date DESC, recorded_at DESC;"
    
#     transactions = execute_query(query, tuple(params) if params else None, fetchall=True)
    
#     # Convert to dictionary keyed by transaction ID
#     result = {}
#     if transactions:
#         for tx in transactions:
#             try:
#                 # Convert RealDictRow to regular dict
#                 tx_dict = dict(tx)
                
#                 # Convert Decimal values to float
#                 for key in ['shares', 'price', 'amount']:
#                     if key in tx_dict and tx_dict[key] is not None:
#                         tx_dict[key] = float(tx_dict[key])
                
#                 # Format dates as strings
#                 tx_dict['date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
#                 tx_dict['transaction_date'] = tx_dict['transaction_date'].strftime("%Y-%m-%d")
#                 tx_dict['recorded_at'] = tx_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
#                 # Ensure ID is converted to string
#                 tx_dict['id'] = str(tx_dict['id'])
                
#                 # Add to result dictionary
#                 result[tx_dict['id']] = tx_dict
#             except Exception as e:
#                 logger.error(f"Error processing transaction: {e}")
#                 continue
    
#     return result