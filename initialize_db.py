#!/usr/bin/env python3
# initialize_db.py
"""
Initialize the PostgreSQL database for the Investment Recommendation System.
This script:
1. Creates the database tables if they don't exist
2. Migrates data from existing JSON files to PostgreSQL
3. Validates the migration
"""
import sys
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_initialization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def initialize_database(skip_migration=False):
    """
    Initialize the database schema and perform data migration.
    
    Args:
        skip_migration (bool): Skip data migration if True
    
    Returns:
        bool: Success status
    """
    try:
        # Import the database utilities module
        from modules.db_utils import initialize_pool, initialize_database, migrate_json_to_db
        
        # Initialize the database connection pool
        logger.info("Initializing database connection pool...")
        if not initialize_pool():
            logger.error("Failed to initialize database connection pool.")
            return False
        
        # Create database schema
        logger.info("Creating database schema...")
        if not initialize_database():
            logger.error("Failed to create database schema.")
            return False
        
        # Migrate data from JSON files to PostgreSQL
        if not skip_migration:
            logger.info("Migrating data from JSON files to PostgreSQL...")
            if not migrate_json_to_db():
                logger.error("Failed to migrate data from JSON files to PostgreSQL.")
                return False
            
            # Migrate mutual fund data
            logger.info("Migrating mutual fund data...")
            from modules.mutual_fund_db import migrate_mutual_fund_data_to_db
            if not migrate_mutual_fund_data_to_db():
                logger.error("Failed to migrate mutual fund data to PostgreSQL.")
                return False
        
        logger.info("Database initialization completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        return False

def validate_migration():
    """
    Validate the migration from JSON files to PostgreSQL.
    Compares record counts and key data points.
    
    Returns:
        bool: Success status
    """
    try:
        logger.info("Validating data migration...")
        
        # Import the necessary modules
        from modules.db_utils import execute_query
        import json
        import os
        
        validation_results = []
        
        # Validate transactions
        if os.path.exists('data/transactions.json'):
            try:
                with open('data/transactions.json', 'r') as f:
                    json_transactions = json.load(f)
                
                db_count_query = "SELECT COUNT(*) as count FROM transactions;"
                db_result = execute_query(db_count_query, fetchone=True)
                
                if db_result:
                    json_count = len(json_transactions)
                    db_count = db_result['count']
                    
                    logger.info(f"Transactions - JSON count: {json_count}, DB count: {db_count}")
                    validation_results.append(json_count == db_count)
            except Exception as e:
                logger.error(f"Error validating transactions: {e}")
                validation_results.append(False)
        
        # Validate portfolio
        if os.path.exists('data/portfolio.json'):
            try:
                with open('data/portfolio.json', 'r') as f:
                    json_portfolio = json.load(f)
                
                db_count_query = "SELECT COUNT(*) as count FROM portfolio;"
                db_result = execute_query(db_count_query, fetchone=True)
                
                if db_result:
                    json_count = len(json_portfolio)
                    db_count = db_result['count']
                    
                    logger.info(f"Portfolio - JSON count: {json_count}, DB count: {db_count}")
                    validation_results.append(json_count == db_count)
            except Exception as e:
                logger.error(f"Error validating portfolio: {e}")
                validation_results.append(False)
        
        # Validate tracked assets
        if os.path.exists('data/tracked_assets.json'):
            try:
                with open('data/tracked_assets.json', 'r') as f:
                    json_assets = json.load(f)
                
                db_count_query = "SELECT COUNT(*) as count FROM tracked_assets;"
                db_result = execute_query(db_count_query, fetchone=True)
                
                if db_result:
                    json_count = len(json_assets)
                    db_count = db_result['count']
                    
                    logger.info(f"Tracked assets - JSON count: {json_count}, DB count: {db_count}")
                    validation_results.append(json_count == db_count)
            except Exception as e:
                logger.error(f"Error validating tracked assets: {e}")
                validation_results.append(False)
        
        # Validate mutual fund prices
        if os.path.exists('data/mutual_fund_cache.json'):
            try:
                with open('data/mutual_fund_cache.json', 'r') as f:
                    json_fund_cache = json.load(f)
                
                # Count total price points in JSON
                json_count = 0
                for fund_code, prices in json_fund_cache.items():
                    json_count += len(prices)
                
                db_count_query = "SELECT COUNT(*) as count FROM mutual_fund_prices;"
                db_result = execute_query(db_count_query, fetchone=True)
                
                if db_result:
                    db_count = db_result['count']
                    
                    logger.info(f"Mutual fund prices - JSON count: {json_count}, DB count: {db_count}")
                    validation_results.append(json_count == db_count)
            except Exception as e:
                logger.error(f"Error validating mutual fund prices: {e}")
                validation_results.append(False)
        
        # Overall validation result
        if not validation_results:
            logger.warning("No validation checks were performed. No JSON files found.")
            return True
        
        if all(validation_results):
            logger.info("Data migration validation successful!")
            return True
        else:
            logger.warning("Data migration validation failed. Some counts don't match.")
            return False
    
    except Exception as e:
        logger.error(f"Error during migration validation: {e}", exc_info=True)
        return False

def create_backups():
    """
    Create backups of existing JSON files before migration.
    
    Returns:
        bool: Success status
    """
    try:
        logger.info("Creating backups of JSON files...")
        
        backup_dir = 'data/backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        import shutil
        import time
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Files to backup
        json_files = [
            'data/transactions.json',
            'data/portfolio.json',
            'data/tracked_assets.json',
            'data/user_profile.json',
            'data/mutual_fund_cache.json'
        ]
        
        for file_path in json_files:
            if os.path.exists(file_path):
                backup_path = f"{backup_dir}/{os.path.basename(file_path)}.{timestamp}"
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
        
        logger.info("Backup process completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error creating backups: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize PostgreSQL database for the Investment Recommendation System')
    parser.add_argument('--skip-migration', action='store_true', help='Skip data migration from JSON files')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of migrated data')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup of JSON files')
    args = parser.parse_args()
    
    # Create backups
    if not args.skip_backup:
        if not create_backups():
            logger.error("Backup creation failed. Exiting.")
            sys.exit(1)
    
    # Initialize database and migrate data
    if not initialize_database(skip_migration=args.skip_migration):
        logger.error("Database initialization failed. Exiting.")
        sys.exit(1)
    
    # Validate migration
    if not args.skip_migration and not args.skip_validation:
        if not validate_migration():
            logger.warning("Migration validation failed, but will continue anyway.")
    
    logger.info("Database setup completed successfully!")