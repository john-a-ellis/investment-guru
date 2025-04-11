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
# Make sure initialize_database and execute_query are imported correctly
from modules.db_utils import initialize_database, execute_query, initialize_pool, save_model_metadata # Added save_model_metadata for potential testing

# --- (Keep load_json_file and all migration functions as they are) ---
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
    migrated_count = 0
    
    for trans_id, trans in transactions.items():
        # Check if transaction already exists
        check_query = "SELECT id FROM transactions WHERE id = %s;"
        existing = execute_query(check_query, (trans_id,), fetchone=True)
        
        if existing:
            logger.debug(f"Transaction {trans_id} already exists, skipping migration.")
            continue

        # Format transaction for insertion
        insert_query = """
        INSERT INTO transactions (
            id, type, symbol, price, shares, amount, transaction_date, notes, recorded_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        # Ensure date is valid, default if not
        trans_date = trans.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            datetime.strptime(trans_date, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format '{trans_date}' for transaction {trans_id}. Defaulting to today.")
            trans_date = datetime.now().strftime("%Y-%m-%d")

        params = (
            trans_id,
            trans.get("type", ""),
            trans.get("symbol", "").upper(),
            float(trans.get("price", 0)),
            float(trans.get("shares", 0)),
            float(trans.get("amount", 0)),
            trans_date,
            trans.get("notes", ""),
            trans.get("recorded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        result = execute_query(insert_query, params, commit=True)
        if result:
            logger.info(f"Migrated transaction {trans_id}")
            migrated_count += 1
        else:
             logger.error(f"Failed to migrate transaction {trans_id}")
    
    logger.info(f"Migration check complete. Migrated {migrated_count} new transactions.")
    return True

def migrate_portfolio():
    """Migrate portfolio from JSON to database"""
    logger.info("Migrating portfolio to database...")
    portfolio = load_json_file('data/portfolio.json')
    migrated_count = 0
    
    for inv_id, inv in portfolio.items():
        symbol_upper = inv.get("symbol", "").upper()
        if not symbol_upper:
            logger.warning(f"Skipping portfolio item with missing symbol (ID: {inv_id})")
            continue

        # Check if portfolio item for this symbol already exists
        check_query = "SELECT symbol FROM portfolio WHERE symbol = %s;"
        existing = execute_query(check_query, (symbol_upper,), fetchone=True)
        
        if existing:
            logger.debug(f"Portfolio item for symbol {symbol_upper} already exists, skipping migration.")
            continue

        # Format investment for insertion
        insert_query = """
        INSERT INTO portfolio (
            id, symbol, shares, purchase_price, purchase_date, asset_type,
            current_price, current_value, gain_loss, gain_loss_percent, currency,
            added_date, last_updated
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        # Ensure date is valid, default if not
        purchase_date = inv.get("purchase_date", datetime.now().strftime("%Y-%m-%d"))
        try:
             if purchase_date: datetime.strptime(purchase_date, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format '{purchase_date}' for portfolio item {symbol_upper}. Defaulting to today.")
            purchase_date = datetime.now().strftime("%Y-%m-%d")
        except TypeError: # Handle if purchase_date is not a string
             logger.warning(f"Invalid date type for portfolio item {symbol_upper}. Defaulting to today.")
             purchase_date = datetime.now().strftime("%Y-%m-%d")


        params = (
            inv_id, # Use the ID from JSON
            symbol_upper,
            float(inv.get("shares", 0)),
            float(inv.get("purchase_price", 0)),
            purchase_date,
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
        if result:
            logger.info(f"Migrated investment {symbol_upper} (ID: {inv_id})")
            migrated_count += 1
        else:
            logger.error(f"Failed to migrate investment {symbol_upper} (ID: {inv_id})")

    logger.info(f"Migration check complete. Migrated {migrated_count} new portfolio items.")
    return True

def migrate_tracked_assets():
    """Migrate tracked assets from JSON to database"""
    logger.info("Migrating tracked assets to database...")
    assets = load_json_file('data/tracked_assets.json')
    migrated_count = 0
    
    for symbol, details in assets.items():
        symbol_upper = symbol.upper()
        # Check if asset already exists
        check_query = "SELECT symbol FROM tracked_assets WHERE symbol = %s;"
        existing = execute_query(check_query, (symbol_upper,), fetchone=True)
        
        if existing:
            logger.debug(f"Tracked asset {symbol_upper} already exists, skipping migration.")
            continue

        # Format asset for insertion
        insert_query = """
        INSERT INTO tracked_assets (
            symbol, name, type, added_date
        ) VALUES (
            %s, %s, %s, %s
        );
        """
         # Ensure date is valid, default if not
        added_date = details.get("added_date", datetime.now().strftime("%Y-%m-%d"))
        try:
             if added_date: datetime.strptime(added_date, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format '{added_date}' for tracked asset {symbol_upper}. Defaulting to today.")
            added_date = datetime.now().strftime("%Y-%m-%d")
        except TypeError:
             logger.warning(f"Invalid date type for tracked asset {symbol_upper}. Defaulting to today.")
             added_date = datetime.now().strftime("%Y-%m-%d")

        params = (
            symbol_upper,
            details.get("name", ""),
            details.get("type", "stock"),
            added_date
        )
        
        result = execute_query(insert_query, params, commit=True)
        if result:
            logger.info(f"Migrated tracked asset {symbol_upper}")
            migrated_count += 1
        else:
            logger.error(f"Failed to migrate tracked asset {symbol_upper}")
    
    logger.info(f"Migration check complete. Migrated {migrated_count} new tracked assets.")
    return True

def migrate_user_profile():
    """Migrate user profile from JSON to database"""
    logger.info("Migrating user profile to database...")
    profile = load_json_file('data/user_profile.json')
    
    if profile:
        # Check if a profile already exists
        check_query = "SELECT COUNT(*) as count FROM user_profile;"
        result = execute_query(check_query, fetchone=True)
        
        if result and result['count'] == 0:
            # Insert the profile if none exists
            logger.info("No existing user profile found, inserting from JSON...")
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
            
            insert_success = execute_query(insert_query, params, commit=True)
            if insert_success:
                logger.info("Migrated user profile from JSON.")
            else:
                logger.error("Failed to migrate user profile from JSON.")
                return False
        else:
            logger.info("Existing user profile found in database. JSON data will not overwrite it.")
            # Optionally, you could implement an update strategy here if desired
            # update_query = """ UPDATE user_profile SET ... WHERE id = ... """
            # execute_query(update_query, params, commit=True)
            # logger.info("Updated existing user profile from JSON (if update logic implemented).")
    else:
        logger.info("No user profile JSON file found or it's empty.")

    return True


def migrate_mutual_fund_data():
    """Migrate mutual fund data from JSON to database"""
    logger.info("Migrating mutual fund data to database...")
    cache_file = 'data/mutual_fund_cache.json'
    
    if not os.path.exists(cache_file):
        logger.info("No mutual fund cache file found. Nothing to migrate.")
        return True
    
    try:
        cache_data = load_json_file(cache_file)
        total_entries_migrated = 0
        
        for fund_code, entries in cache_data.items():
            fund_code_upper = fund_code.upper()
            for date_str, price in entries.items():
                try:
                    # Check if this specific entry already exists
                    check_query = """
                    SELECT id FROM mutual_fund_prices 
                    WHERE fund_code = %s AND price_date = %s;
                    """
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    existing = execute_query(check_query, (fund_code_upper, date_obj), fetchone=True)

                    if existing:
                        logger.debug(f"Price for {fund_code_upper} on {date_str} already exists, skipping.")
                        continue

                    # Format for insertion
                    insert_query = """
                    INSERT INTO mutual_fund_prices (
                        fund_code, price_date, price, added_at
                    ) VALUES (
                        %s, %s, %s, %s
                    );
                    """
                    
                    params = (
                        fund_code_upper,
                        date_obj,
                        float(price),
                        datetime.now() # Use current timestamp for added_at during migration
                    )
                    
                    result = execute_query(insert_query, params, commit=True)
                    if result:
                        total_entries_migrated += 1
                    else:
                         logger.error(f"Failed migrating data for {fund_code_upper} on {date_str}")

                except ValueError:
                     logger.error(f"Invalid date format '{date_str}' for fund {fund_code_upper}, skipping entry.")
                except Exception as e:
                    logger.error(f"Error migrating data for {fund_code_upper} on {date_str}: {e}")
        
        logger.info(f"Migration check complete. Migrated {total_entries_migrated} new mutual fund price entries.")
        
        # Optional: Rename the original file as backup after successful migration check
        # backup_file = f"{cache_file}.migrated_bak_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # if os.path.exists(cache_file):
        #     try:
        #         os.rename(cache_file, backup_file)
        #         logger.info(f"Original cache file renamed to {backup_file}")
        #     except OSError as e:
        #          logger.error(f"Could not rename cache file {cache_file}: {e}")

        return True
    except Exception as e:
        logger.error(f"Error during mutual fund data migration: {e}")
        return False

def migrate_json_to_db():
    """
    Migrate all data from JSON files to PostgreSQL database.
    Only migrates data that doesn't already exist based on primary/unique keys.
    """
    logger.info("Starting data migration check from JSON files to database...")
    all_success = True
    try:
        os.makedirs('data', exist_ok=True)
        
        if not migrate_transactions(): all_success = False
        if not migrate_portfolio(): all_success = False
        if not migrate_tracked_assets(): all_success = False
        if not migrate_user_profile(): all_success = False
        if not migrate_mutual_fund_data(): all_success = False
        
        if all_success:
            logger.info("Migration check from JSON to database completed.")
        else:
            logger.warning("Some migrations encountered issues (see logs above).")
            
        return all_success # Return overall success status
        
    except Exception as e:
        logger.error(f"Critical error during migration process: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to initialize database and migrate data"""
    logger.info("--- Starting Database Initialization ---")
    
    logger.info("Initializing database connection pool...")
    if not initialize_pool():
        logger.critical("Failed to initialize connection pool. Aborting.")
        return False
    
    logger.info("Initializing/Verifying database schema...")
    if not initialize_database(): # This now creates the trained_models table too
        logger.critical("Failed to initialize database schema. Aborting.")
        return False
    
    logger.info("Checking for data migration from JSON files...")
    if not migrate_json_to_db():
        logger.warning("Data migration check from JSON encountered issues or failed.")
        # Decide if this should be a fatal error or just a warning
        # return False # Uncomment this line if migration failure should stop the process
    
    # --- Optional: Add a dummy model entry for testing ---
    # logger.info("Adding a dummy trained model entry for testing purposes...")
    # dummy_metrics = {"accuracy": 0.85, "mse": 0.015, "r2_score": 0.75}
    # save_model_metadata(
    #     filename="DUMMY_LSTM_TEST.pkl",
    #     symbol="TEST",
    #     model_type="LSTM",
    #     metrics=dummy_metrics,
    #     notes="This is a test entry added during initialization."
    # )
    # ----------------------------------------------------

    logger.info("--- Database Initialization Process Completed ---")
    return True

if __name__ == "__main__":
    if main():
        print("\nDatabase initialization and migration check completed successfully!")
        sys.exit(0)
    else:
        print("\nDatabase initialization or migration check failed! Check logs for details.")
        sys.exit(1)

# --- END OF initialize_db.py modifications ---
