# modules/dividend_utils.py
"""
Utility functions for tracking and analyzing dividend income.
Handles recording, retrieving, and analyzing dividend payments.
"""
import logging
from datetime import datetime
import uuid
from modules.db_utils import execute_query
from modules.portfolio_utils import load_portfolio, update_portfolio_data, get_usd_to_cad_rate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def record_dividend(symbol, dividend_date, amount_per_share, shares_held=None, 
                    record_date=None, currency=None, is_drip=False, 
                    drip_shares=0, drip_price=0, notes=None):
    """
    Record a dividend payment in the database.
    
    Args:
        symbol (str): Stock/ETF symbol
        dividend_date (str): Date dividend was paid (YYYY-MM-DD)
        amount_per_share (float): Dividend amount per share
        shares_held (float, optional): Number of shares held. If None, will use current portfolio
        record_date (str, optional): Record date for dividend eligibility (YYYY-MM-DD)
        currency (str, optional): Currency of dividend (USD or CAD). If None, will determine from symbol
        is_drip (bool): Whether dividend was reinvested via DRIP
        drip_shares (float): Number of shares acquired through DRIP
        drip_price (float): Price per share for DRIP
        notes (str, optional): Additional notes about the dividend
        
    Returns:
        bool: Success status
    """
    try:
        # Standardize symbol and dates
        symbol = symbol.upper().strip()
        
        # Ensure dividend_date is in YYYY-MM-DD format
        try:
            if dividend_date:
                dividend_date = datetime.strptime(dividend_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid dividend date format: {dividend_date}")
            return False
        
        # Ensure record_date is in YYYY-MM-DD format if provided
        if record_date:
            try:
                record_date = datetime.strptime(record_date, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid record date format: {record_date}")
                return False
        
        # If shares_held not provided, check portfolio
        if shares_held is None:
            portfolio = load_portfolio()
            total_shares = 0
            
            for inv_id, inv in portfolio.items():
                if inv.get("symbol", "").upper() == symbol:
                    total_shares += float(inv.get("shares", 0))
            
            if total_shares <= 0:
                logger.warning(f"No shares of {symbol} found in portfolio. Dividend will be recorded with 0 shares.")
                shares_held = 0
            else:
                shares_held = total_shares
        
        # Convert to float for calculations
        amount_per_share = float(amount_per_share)
        shares_held = float(shares_held)
        
        # Calculate total dividend amount
        total_amount = amount_per_share * shares_held
        
        # If currency not provided, determine from symbol
        if currency is None:
            is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
            currency = "CAD" if is_canadian else "USD"
        
        # Generate unique ID for this dividend record
        dividend_id = str(uuid.uuid4())
        
        # Insert into dividends table
        insert_query = """
        INSERT INTO dividends (
            id, symbol, dividend_date, record_date, amount_per_share, 
            shares_held, total_amount, currency, is_drip, 
            drip_shares, drip_price, notes, recorded_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        params = (
            dividend_id,
            symbol,
            dividend_date,
            record_date,
            amount_per_share,
            shares_held,
            total_amount,
            currency.upper(),
            is_drip,
            float(drip_shares) if drip_shares else 0,
            float(drip_price) if drip_price else 0,
            notes,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        result = execute_query(insert_query, params, commit=True)
        
        if result is not None:
            logger.info(f"Dividend recorded for {symbol}: {total_amount} {currency}")
            
            # If this is a DRIP, update the portfolio with new shares
            if is_drip and drip_shares > 0:
                _update_portfolio_for_drip(symbol, drip_shares, drip_price, dividend_date)
            
            return True
        else:
            logger.error(f"Failed to record dividend for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error recording dividend: {e}")
        return False

def _update_portfolio_for_drip(symbol, drip_shares, drip_price, date):
    """
    Internal helper to update portfolio holdings when dividends are reinvested.
    
    Args:
        symbol (str): Stock symbol
        drip_shares (float): Number of shares acquired through DRIP
        drip_price (float): Price per share for DRIP
        date (str): Date of DRIP transaction (YYYY-MM-DD)
        
    Returns:
        bool: Success status
    """
    try:
        # Import here to avoid circular imports
        from modules.portfolio_utils import record_transaction
        
        # Record as a special "drip" transaction type
        transaction_success = record_transaction(
            transaction_type="drip",  # This will need to be handled in record_transaction
            symbol=symbol,
            price=drip_price,
            shares=drip_shares,
            date=date,
            notes="Dividend Reinvestment Plan (DRIP)"
        )
        
        if transaction_success:
            logger.info(f"Portfolio updated with DRIP shares: {drip_shares} shares of {symbol}")
            return True
        else:
            logger.error(f"Failed to update portfolio with DRIP shares for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating portfolio for DRIP: {e}")
        return False

def load_dividends(symbol=None, start_date=None, end_date=None):
    """
    Load dividend records from database.
    
    Args:
        symbol (str, optional): Filter by symbol
        start_date (str, optional): Filter by start date (YYYY-MM-DD)
        end_date (str, optional): Filter by end date (YYYY-MM-DD)
        
    Returns:
        list: Dividend records
    """
    query = "SELECT * FROM dividends"
    params = []
    filters = []
    
    if symbol:
        filters.append("symbol = %s")
        params.append(symbol.upper())
    
    if start_date:
        filters.append("dividend_date >= %s")
        params.append(start_date)
    
    if end_date:
        filters.append("dividend_date <= %s")
        params.append(end_date)
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    query += " ORDER BY dividend_date DESC, recorded_at DESC;"
    
    dividends = execute_query(query, tuple(params) if params else None, fetchall=True)
    
    result = []
    if dividends:
        for div in dividends:
            try:
                div_dict = dict(div)
                
                # Convert Decimal to float
                for key in ['amount_per_share', 'shares_held', 'total_amount', 'drip_shares', 'drip_price']:
                    if key in div_dict and div_dict[key] is not None:
                        div_dict[key] = float(div_dict[key])
                
                # Format dates
                div_dict['date'] = div_dict['dividend_date'].strftime("%Y-%m-%d")
                div_dict['dividend_date'] = div_dict['dividend_date'].strftime("%Y-%m-%d")
                if div_dict.get('record_date'):
                    div_dict['record_date'] = div_dict['record_date'].strftime("%Y-%m-%d")
                div_dict['recorded_at'] = div_dict['recorded_at'].strftime("%Y-%m-%d %H:%M:%S")
                
                # Ensure boolean is correct
                div_dict['is_drip'] = bool(div_dict.get('is_drip', False))
                
                # Ensure ID is string
                div_dict['id'] = str(div_dict['id'])
                
                result.append(div_dict)
            except Exception as e:
                logger.error(f"Error processing dividend record {div.get('id')}: {e}")
                continue
    
    return result

def get_dividend_yield(symbol=None, period="1y"):
    """
    Calculate current dividend yield for a symbol or the entire portfolio.
    
    Args:
        symbol (str, optional): Calculate for specific symbol. If None, calculates for whole portfolio
        period (str): Time period for yield calculation ('1m', '3m', '6m', '1y', 'all')
        
    Returns:
        dict: Dividend yield information
    """
    # Define date range based on period
    end_date = datetime.now()
    
    if period == "1m":
        start_date = (end_date - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    elif period == "3m":
        start_date = (end_date - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    elif period == "6m":
        start_date = (end_date - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    elif period == "1y":
        start_date = (end_date - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    else:  # "all" - no start date filter
        start_date = None
    
    end_date = end_date.strftime("%Y-%m-%d")
    
    # Load dividends for the period
    dividends = load_dividends(symbol=symbol, start_date=start_date, end_date=end_date)
    
    # Initialize results
    result = {
        "total_dividends_cad": 0,
        "total_dividends_usd": 0,
        "total_dividends_combined_cad": 0,
        "count": len(dividends),
        "period": period,
        "symbols": set(),
        "dividend_yield": 0,
        "annualized_yield": 0
    }
    
    # Get current exchange rate for USD dividends
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Sum dividend amounts by currency
    for div in dividends:
        result["symbols"].add(div["symbol"])
        
        if div["currency"] == "CAD":
            result["total_dividends_cad"] += div["total_amount"]
        else:  # USD
            result["total_dividends_usd"] += div["total_amount"]
    
    # Convert USD dividends to CAD for combined total
    result["total_dividends_combined_cad"] = result["total_dividends_cad"] + (result["total_dividends_usd"] * usd_to_cad_rate)
    
    # Get current portfolio value to calculate yield
    portfolio = update_portfolio_data()
    
    # Calculate total portfolio value in CAD
    portfolio_value_cad = 0
    portfolio_value_usd = 0
    
    if symbol:
        # Calculate value for specific symbol
        for inv_id, inv in portfolio.items():
            if inv.get("symbol", "").upper() == symbol.upper():
                if inv.get("currency", "USD") == "CAD":
                    portfolio_value_cad += float(inv.get("current_value", 0))
                else:  # USD
                    portfolio_value_usd += float(inv.get("current_value", 0))
    else:
        # Calculate value for entire portfolio
        for inv_id, inv in portfolio.items():
            if inv.get("currency", "USD") == "CAD":
                portfolio_value_cad += float(inv.get("current_value", 0))
            else:  # USD
                portfolio_value_usd += float(inv.get("current_value", 0))
    
    # Convert USD value to CAD for total
    portfolio_value_combined_cad = portfolio_value_cad + (portfolio_value_usd * usd_to_cad_rate)
    
    # Calculate yield if portfolio value exists
    if portfolio_value_combined_cad > 0:
        # Calculate yield based on period
        if period == "1m":
            annualization_factor = 12
        elif period == "3m":
            annualization_factor = 4
        elif period == "6m":
            annualization_factor = 2
        elif period == "1y":
            annualization_factor = 1
        else:  # "all" - need to calculate based on timespan
            # Get earliest dividend date
            if dividends:
                earliest_date = min(datetime.strptime(div["dividend_date"], "%Y-%m-%d") for div in dividends)
                days_span = (datetime.now() - earliest_date).days
                if days_span > 0:
                    annualization_factor = 365 / days_span
                else:
                    annualization_factor = 1
            else:
                annualization_factor = 1
        
        # Calculate actual yield for the period
        result["dividend_yield"] = (result["total_dividends_combined_cad"] / portfolio_value_combined_cad) * 100
        
        # Calculate annualized yield
        result["annualized_yield"] = result["dividend_yield"] * annualization_factor
    
    # Add portfolio value information
    result["portfolio_value_cad"] = portfolio_value_cad
    result["portfolio_value_usd"] = portfolio_value_usd
    result["portfolio_value_combined_cad"] = portfolio_value_combined_cad
    
    return result

def get_dividend_summary_by_symbol(start_date=None, end_date=None):
    """
    Get dividend summary grouped by symbol.
    
    Args:
        start_date (str, optional): Filter by start date (YYYY-MM-DD)
        end_date (str, optional): Filter by end date (YYYY-MM-DD)
        
    Returns:
        dict: Dividend summary by symbol
    """
    # Load all dividends for the period
    dividends = load_dividends(start_date=start_date, end_date=end_date)
    
    summary = {}
    
    for div in dividends:
        symbol = div["symbol"]
        
        if symbol not in summary:
            summary[symbol] = {
                "total_cad": 0,
                "total_usd": 0,
                "count": 0,
                "total_shares": 0,
                "currencies": set(),
                "latest_dividend_date": None,
                "latest_amount_per_share": 0
            }
        
        # Update summary
        if div["currency"] == "CAD":
            summary[symbol]["total_cad"] += div["total_amount"]
        else:  # USD
            summary[symbol]["total_usd"] += div["total_amount"]
            
        summary[symbol]["count"] += 1
        summary[symbol]["currencies"].add(div["currency"])
        
        # Track latest dividend
        if not summary[symbol]["latest_dividend_date"] or div["dividend_date"] > summary[symbol]["latest_dividend_date"]:
            summary[symbol]["latest_dividend_date"] = div["dividend_date"]
            summary[symbol]["latest_amount_per_share"] = div["amount_per_share"]
        
        # If this is the first dividend, use its shares as the total
        if summary[symbol]["total_shares"] == 0:
            summary[symbol]["total_shares"] = div["shares_held"]
    
    # Get current USD to CAD rate for conversion
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Calculate combined totals and yield
    for symbol, data in summary.items():
        # Convert currencies set to list
        data["currencies"] = list(data["currencies"])
        
        # Calculate combined total in CAD
        data["total_combined_cad"] = data["total_cad"] + (data["total_usd"] * usd_to_cad_rate)
        
        # Get current share price and calculate yield
        portfolio = load_portfolio()
        current_price = 0
        current_shares = 0
        
        for inv_id, inv in portfolio.items():
            if inv.get("symbol", "") == symbol:
                current_price = float(inv.get("current_price", 0))
                current_shares += float(inv.get("shares", 0))
        
        data["current_price"] = current_price
        data["current_shares"] = current_shares
        
        # Calculate current yield
        if current_price > 0 and data["latest_amount_per_share"] > 0:
            # Assume quarterly dividends for estimation
            data["estimated_annual_dividend"] = data["latest_amount_per_share"] * 4
            data["current_yield"] = (data["estimated_annual_dividend"] / current_price) * 100
        else:
            data["estimated_annual_dividend"] = 0
            data["current_yield"] = 0
    
    return summary