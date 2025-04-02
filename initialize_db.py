#!/usr/bin/env python
"""
Database initialization script for the Investment Recommendation System.
This script creates the necessary database tables and migrates data from JSON files.
"""
import os
import sys
import logging
import json
from datetime import datetime
import uuid

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database utilities
from modules.db_utils import initialize_database, execute_query, initialize_pool

def load_json_file(filename):
    """Load data from a JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return {}

def migrate_transactions():
    """Migrate transactions from JSON to database"""
    logger.info("Migrating transactions to database...")
    transactions = load_json_file('data/transactions.json')
    
    for trans_id, trans in transactions.items():
        # Format transaction for insertion
        insert_query = """
        INSERT INTO transactions (
            id, type, symbol, price, shares, amount, transaction_date, notes, recorded_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (id) DO NOTHING;
        """
        
        params = (
            trans_id,
            trans.get("type", ""),
            trans.get("symbol", "").upper(),
            float(trans.get("price", 0)),
            float(trans.get("shares", 0)),
            float(trans.get("amount", 0)),
            trans.get("date", datetime.now().strftime("%Y-%m-%d")),
            trans.get("notes", ""),
            trans.get("recorded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        result = execute_query(insert_query, params, commit=True)
        if result is not None:
            logger.info(f"Migrated transaction {trans_id}")
    
    logger.info(f"Migrated {len(transactions)} transactions")
    return True

def migrate_portfolio():
    """Migrate portfolio from JSON to database"""
    logger.info("Migrating portfolio to database...")
    portfolio = load_json_file('data/portfolio.json')
    
    for inv_id, inv in portfolio.items():
        # Format investment for insertion
        insert_query = """
        INSERT INTO portfolio (
            id, symbol, shares, purchase_price, purchase_date, asset_type,
            current_price, current_value, gain_loss, gain_loss_percent, currency,
            added_date, last_updated
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (id) DO NOTHING;
        """
        
        params = (
            inv_id,
            inv.get("symbol", "").upper(),
            float(inv.get("shares", 0)),
            float(inv.get("purchase_price", 0)),
            inv.get("purchase_date", datetime.now().strftime("%Y-%m-%d")),
            inv.get("asset_type", "stock"),
            float(inv.get("current_price", 0)) if inv.get("current_price") else None,
            float(inv.get("current_value", 0)) if inv.get("current_value") else None,
            float(inv.get("gain_loss", 0)) if inv.get("gain_loss") else None,
            float(inv.get("gain_loss_percent", 0)) if inv.get("gain_loss_percent") else None,
            inv.get("currency", "USD"),
            inv.get("added_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            inv.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        result = execute_query(insert_query, params, commit=True)
        if result is not None:
            logger.info(f"Migrated investment {inv_id}")
    
    logger.info(f"Migrated {len(portfolio)} investments")
    return True

def migrate_tracked_assets():
    """Migrate tracked assets from JSON to database"""
    logger.info("Migrating tracked assets to database...")
    assets = load_json_file('data/tracked_assets.json')
    
    for symbol, details in assets.items():
        # Format asset for insertion
        insert_query = """
        INSERT INTO tracked_assets (
            symbol, name, type, added_date
        ) VALUES (
            %s, %s, %s, %s
        ) ON CONFLICT (symbol) DO NOTHING;
        """
        
        params = (
            symbol.upper(),
            details.get("name", ""),
            details.get("type", "stock"),
            details.get("added_date", datetime.now().strftime("%Y-%m-%d"))
        )
        
        result = execute_query(insert_query, params, commit=True)
        if result is not None:
            logger.info(f"Migrated tracked asset {symbol}")
    
    logger.info(f"Migrated {len(assets)} tracked assets")
    return True

def migrate_user_profile():
    """Migrate user profile from JSON to database"""
    logger.info("Migrating user profile to database...")
    profile = load_json_file('data/user_profile.json')
    
    if profile:
        # First, check if a profile already exists
        check_query = "SELECT COUNT(*) as count FROM user_profile;"
        result = execute_query(check_query, fetchone=True)
        
        if result and result['count'] == 0:
            # Insert the profile if none exists
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
                profile.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            
            result = execute_query(insert_query, params, commit=True)
            if result is not None:
                logger.info("Migrated user profile")
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
                profile.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            
            result = execute_query(update_query, params, commit=True)
            if result is not None:
                logger.info("Updated existing user profile")
    
    return True

def migrate_mutual_fund_data():
    """Migrate mutual fund data from JSON to database"""
    logger.info("Migrating mutual fund data to database...")
    # Path to the mutual fund cache file
    cache_file = 'data/mutual_fund_cache.json'
    
    if not os.path.exists(cache_file):
        logger.info("No mutual fund cache file found. Nothing to migrate.")
        return True
    
    try:
        # Load the cache file
        cache_data = load_json_file(cache_file)
        total_entries = 0
        
        # Migrate each fund's data
        for fund_code, entries in cache_data.items():
            for date_str, price in entries.items():
                try:
                    # Format for insertion
                    insert_query = """
                    INSERT INTO mutual_fund_prices (
                        fund_code, price_date, price, added_at
                    ) VALUES (
                        %s, %s, %s, %s
                    ) ON CONFLICT (fund_code, price_date) 
                    DO UPDATE SET price = EXCLUDED.price, added_at = EXCLUDED.added_at;
                    """
                    
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    params = (
                        fund_code.upper(),
                        date,
                        float(price),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    result = execute_query(insert_query, params, commit=True)
                    if result is not None:
                        total_entries += 1
                except Exception as e:
                    logger.error(f"Error migrating data for {fund_code} on {date_str}: {e}")
        
        logger.info(f"Migrated {total_entries} mutual fund price entries")
        
        # Rename the original file as backup
        backup_file = f"{cache_file}.bak"
        if os.path.exists(cache_file):
            os.rename(cache_file, backup_file)
            logger.info(f"Original cache file renamed to {backup_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error during mutual fund data migration: {e}")
        return False

def migrate_json_to_db():
    """
    Migrate all data from JSON files to PostgreSQL database.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Migrate each data type
        if not migrate_transactions():
            logger.error("Failed to migrate transactions")
            return False
        
        if not migrate_portfolio():
            logger.error("Failed to migrate portfolio")
            return False
        
        if not migrate_tracked_assets():
            logger.error("Failed to migrate tracked assets")
            return False
        
        if not migrate_user_profile():
            logger.error("Failed to migrate user profile")
            return False
        
        if not migrate_mutual_fund_data():
            logger.error("Failed to migrate mutual fund data")
            return False
        
        logger.info("Migration from JSON to database completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

def main():
    """Main function to initialize database and migrate data"""
    logger.info("Initializing database connection...")
    
    # Initialize connection pool
    if not initialize_pool():
        logger.error("Failed to initialize connection pool")
        return False
    
    logger.info("Initializing database schema...")
    # Create database tables if they don't exist
    if not initialize_database():
        logger.error("Failed to initialize database schema")
        return False
    
    logger.info("Migrating data from JSON files to database...")
    # Migrate data from JSON files to database
    if not migrate_json_to_db():
        logger.error("Failed to migrate data from JSON files to database")
        return False
    
    logger.info("Database initialization completed successfully!")
    return True

if __name__ == "__main__":
    if main():
        print("Database initialization completed successfully!")
        sys.exit(0)
    else:
        print("Database initialization failed!")
        sys.exit(1)