# modules/transaction_tracker.py - Fixed imports
from datetime import datetime
import uuid

from modules.portfolio_utils import load_transactions, save_transactions, update_portfolio_for_transaction

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

