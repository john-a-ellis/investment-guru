# setup/migrate_mutual_funds.py
"""
Script to migrate mutual fund price data from JSON to PostgreSQL.
Run this once during initial setup or when migrating from the old storage format.
"""
import os
import json
import logging
from datetime import datetime
from modules.db_utils import execute_query
from modules.mutual_fund_db import add_mutual_fund_price

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_mutual_fund_data_to_db():
    """
    Migrate mutual fund price data from JSON cache to PostgreSQL database.
    This should be called once during the transition.
    """
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

if __name__ == "__main__":
    print("Starting mutual fund data migration to PostgreSQL...")
    success = migrate_mutual_fund_data_to_db()
    
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed! Check the logs for details.")