# modules/transaction_tracker.py - Fixed imports
from datetime import datetime
import uuid

from modules.portfolio_utils import load_transactions, save_transactions, update_portfolio_for_transaction
# Fix imports - import directly from components or define the functions here
# try:
#     # Try to import directly from portfolio_management
#     from components.portfolio_management import load_portfolio, save_portfolio
# except ImportError:
#     # Fallback - define the functions here to avoid circular imports
#     def load_portfolio():
#         """
#         Load portfolio data from storage file
#         """
#         try:
#             if os.path.exists('data/portfolio.json'):
#                 with open('data/portfolio.json', 'r') as f:
#                     return json.load(f)
#             else:
#                 # Default empty portfolio if no file exists
#                 return {}
#         except Exception as e:
#             print(f"Error loading portfolio: {e}")
#             return {}

#     def save_portfolio(portfolio):
#         """
#         Save portfolio data to storage file
#         """
#         try:
#             os.makedirs('data', exist_ok=True)
#             with open('data/portfolio.json', 'w') as f:
#                 json.dump(portfolio, f, indent=4)
#             return True
#         except Exception as e:
#             print(f"Error saving portfolio: {e}")
#             return False

def record_transaction(transaction_type, symbol, price, shares, amount, date=None, notes=""):
    """
    Record a buy/sell transaction
    
    Args:
        transaction_type (str): "buy" or "sell"
        symbol (str): Asset symbol
        price (float): Price per share/unit
        shares (float): Number of shares/units
        amount (float): Total transaction amount
        date (str): Transaction date (optional, defaults to current date)
        notes (str): Transaction notes (optional)
        
    Returns:
        bool: Success status
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
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
        "amount": float(amount),
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

# def load_transactions():
#     """
#     Load transaction records from storage file
    
#     Returns:
#         dict: Transaction records
#     """
#     try:
#         if os.path.exists('data/transactions.json'):
#             with open('data/transactions.json', 'r') as f:
#                 return json.load(f)
#         else:
#             # Default empty transactions if no file exists
#             return {}
#     except Exception as e:
#         print(f"Error loading transactions: {e}")
#         return {}

# def save_transactions(transactions):
#     """
#     Save transaction records to storage file
    
#     Args:
#         transactions (dict): Transaction records
        
#     Returns:
#         bool: Success status
#     """
#     try:
#         os.makedirs('data', exist_ok=True)
#         with open('data/transactions.json', 'w') as f:
#             json.dump(transactions, f, indent=4)
#         return True
#     except Exception as e:
#         print(f"Error saving transactions: {e}")
#         return False

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
#     # Load current portfolio
#     portfolio = load_portfolio()
    
#     # Find if we already have this symbol in our portfolio
#     existing_investment = None
#     for inv_id, inv in portfolio.items():
#         if inv.get("symbol") == symbol:
#             existing_investment = (inv_id, inv)
#             break
    
#     if transaction_type.lower() == "buy":
#         # Buy transaction
#         if existing_investment:
#             # Update existing investment
#             inv_id, inv = existing_investment
            
#             current_shares = inv.get("shares", 0)
#             current_value = current_shares * inv.get("current_price", price)
            
#             # Add new shares
#             new_shares = current_shares + float(shares)
            
#             # Calculate new average purchase price (weighted average)
#             current_cost = current_shares * inv.get("purchase_price", 0)
#             new_cost = float(shares) * float(price)
#             new_avg_price = (current_cost + new_cost) / new_shares if new_shares > 0 else 0
            
#             # Update investment
#             inv["shares"] = new_shares
#             inv["purchase_price"] = new_avg_price
            
#             # If purchase date is earlier than current, update it
#             if date < inv.get("purchase_date", date):
#                 inv["purchase_date"] = date
                
#             # Update current value
#             inv["current_value"] = new_shares * inv.get("current_price", price)
            
#             # Recalculate gain/loss
#             inv["gain_loss"] = inv["current_value"] - (new_shares * new_avg_price)
#             inv["gain_loss_percent"] = (inv["current_value"] / (new_shares * new_avg_price) - 1) * 100
#         else:
#             # Add new investment
#             from modules.portfolio_data_updater import add_investment
#             add_investment(symbol, shares, price, date)
    
#     elif transaction_type.lower() == "sell":
#         # Sell transaction
#         if existing_investment:
#             # Update existing investment
#             inv_id, inv = existing_investment
            
#             current_shares = inv.get("shares", 0)
            
#             # Remove sold shares
#             new_shares = current_shares - float(shares)
            
#             if new_shares <= 0:
#                 # If all shares sold, remove investment
#                 del portfolio[inv_id]
#             else:
#                 # Update shares and current value
#                 inv["shares"] = new_shares
#                 inv["current_value"] = new_shares * inv.get("current_price", price)
                
#                 # Recalculate gain/loss
#                 purchase_cost = new_shares * inv.get("purchase_price", 0)
#                 inv["gain_loss"] = inv["current_value"] - purchase_cost
#                 inv["gain_loss_percent"] = (inv["current_value"] / purchase_cost - 1) * 100
    
#     # Save updated portfolio
#     save_portfolio(portfolio)