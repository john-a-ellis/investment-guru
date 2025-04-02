# modules/mutual_fund_db.py
"""
Database integration for mutual fund price data.
Replaces the JSON file storage in MutualFundProvider.
"""
import logging
from datetime import datetime
from modules.db_utils import execute_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_mutual_fund_price(fund_code, date, price):
    """
    Add a manually entered price point for a mutual fund
    
    Args:
        fund_code (str): Fund code/symbol
        date (datetime or str): Date of the price point
        price (float): NAV price
    
    Returns:
        bool: Success status
    """
    try:
        # Convert string date to datetime if needed
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # Insert the price point
        insert_query = """
        INSERT INTO mutual_fund_prices (
            fund_code, price_date, price, added_at
        ) VALUES (
            %s, %s, %s, %s
        ) ON CONFLICT (fund_code, price_date) 
        DO UPDATE SET price = EXCLUDED.price, added_at = EXCLUDED.added_at;
        """
        
        params = (
            fund_code.upper(),
            date,
            float(price),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        result = execute_query(insert_query, params, commit=True)
        
        if result is not None:
            logger.info(f"Mutual fund price added successfully: {fund_code} {date} {price}")
            return True
        
        logger.error(f"Failed to add mutual fund price: {fund_code} {date} {price}")
        return False
    except Exception as e:
        logger.error(f"Error adding mutual fund price for {fund_code}: {e}")
        return False

def get_mutual_fund_prices(fund_code, start_date=None, end_date=None):
    """
    Get historical price data for a mutual fund
    
    Args:
        fund_code (str): Fund code/symbol
        start_date (datetime or str): Start date (optional)
        end_date (datetime or str): End date (optional)
    
    Returns:
        list: List of price data dictionaries
    """
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Build query
        query = "SELECT * FROM mutual_fund_prices WHERE fund_code = %s"
        params = [fund_code.upper()]
        
        if start_date:
            query += " AND price_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND price_date <= %s"
            params.append(end_date)
        
        query += " ORDER BY price_date ASC;"
        
        # Execute query
        results = execute_query(query, params, fetchall=True)
        
        if results:
            # Format the results
            price_data = []
            for row in results:
                price_data.append({
                    'fund_code': row['fund_code'],
                    'date': row['price_date'].strftime("%Y-%m-%d"),
                    'price': float(row['price']),
                    'added_at': row['added_at'].strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return price_data
        
        return []
    except Exception as e:
        logger.error(f"Error getting mutual fund prices for {fund_code}: {e}")
        return []

def get_latest_mutual_fund_price(fund_code):
    """
    Get the latest price for a mutual fund
    
    Args:
        fund_code (str): Fund code/symbol
    
    Returns:
        float: Latest price or None if not available
    """
    try:
        query = """
        SELECT price FROM mutual_fund_prices 
        WHERE fund_code = %s 
        ORDER BY price_date DESC 
        LIMIT 1;
        """
        
        result = execute_query(query, (fund_code.upper(),), fetchone=True)
        
        if result:
            return float(result['price'])
        
        return None
    except Exception as e:
        logger.error(f"Error getting latest mutual fund price for {fund_code}: {e}")
        return None

def migrate_mutual_fund_data_to_db():
    """
    Migrate mutual fund price data from JSON cache to PostgreSQL database.
    This should be called once during the transition.
    """
    import os
    import json
    
    # Path to the mutual fund cache file
    cache_file = 'data/mutual_fund_cache.json'
    
    if not os.path.exists(cache_file):
        logger.info("No mutual fund cache file found. Nothing to migrate.")
        return True
    
    try:
        # Load the cache file
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Migrate each fund's data
        for fund_code, entries in cache_data.items():
            for date_str, price in entries.items():
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    add_mutual_fund_price(fund_code, date, float(price))
                except Exception as e:
                    logger.error(f"Error migrating data for {fund_code} on {date_str}: {e}")
        
        logger.info("Mutual fund data migration completed successfully.")
        
        # Rename the original file as backup
        backup_file = f"{cache_file}.bak"
        os.rename(cache_file, backup_file)
        logger.info(f"Original cache file renamed to {backup_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error during mutual fund data migration: {e}")
        return False