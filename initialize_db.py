#!/usr/bin/env python3
"""
Database Initialization Script for Investment Recommendation System

This script:
1. Initializes the PostgreSQL database with required tables
2. Migrates data from JSON files to PostgreSQL database
3. Creates backups of original JSON files

Usage:
    python initialize_db.py
"""

import os
import sys
import shutil
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.db_utils import initialize_pool, initialize_database, migrate_json_to_db
from modules.mutual_fund_db import migrate_mutual_fund_data_to_db
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_backups():
    """Create backups of existing JSON data files"""
    backup_dir = 'data/backups'
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Files to back up
    files_to_backup = [
        'data/transactions.json',
        'data/portfolio.json',
        'data/tracked_assets.json',
        'data/user_profile.json',
        'data/mutual_fund_cache.json'
    ]
    
    logger.info("Creating backups of JSON files...")
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = f"{backup_dir}/{os.path.basename(file_path)}.{timestamp}"
            try:
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup {file_path}: {e}")
    
    logger.info("Backup process completed successfully!")

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Step 1: Create backups of JSON files
    create_backups()
    
    # Step 2: Initialize database connection pool
    logger.info("Initializing database connection pool...")
    if not initialize_pool():
        logger.error("Failed to initialize database connection pool.")
        return False
    
    # Step 3: Create database schema
    logger.info("Creating database schema...")
    if not initialize_database():
        logger.error("Failed to create database schema.")
        return False
    
    # Step 4: Migrate data from JSON files to database
    logger.info("Migrating data from JSON files to database...")
    if not migrate_json_to_db():
        logger.error("Failed to migrate data from JSON files to database.")
        return False
    
    # Step 5: Migrate mutual fund data to database
    logger.info("Migrating mutual fund data to database...")
    if not migrate_mutual_fund_data_to_db():
        logger.error("Failed to migrate mutual fund data to database.")
        return False
    
    logger.info("Database initialization and data migration completed successfully!")
    return True

if __name__ == "__main__":
    try:
        if main():
            logger.info("Database setup completed successfully.")
        else:
            logger.error("Database initialization failed. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled exception during database initialization: {e}")
        sys.exit(1)