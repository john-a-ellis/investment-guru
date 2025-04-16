# scripts/migrate_dividends.py
"""
Migration script to ensure the dividends table exists in the database.
Run this script after updating the application with dividend functionality.
"""
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database utilities
from modules.db_utils import execute_query, initialize_pool

def ensure_dividends_table():
    """Create the dividends table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS dividends (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        symbol VARCHAR(50) NOT NULL,
        dividend_date DATE NOT NULL,
        record_date DATE,
        amount_per_share NUMERIC(15, 6) NOT NULL,
        shares_held NUMERIC(15, 6) NOT NULL,
        total_amount NUMERIC(15, 4) NOT NULL,
        currency VARCHAR(10) DEFAULT 'CAD',
        is_drip BOOLEAN DEFAULT FALSE,
        drip_shares NUMERIC(15, 6) DEFAULT 0,
        drip_price NUMERIC(15, 4) DEFAULT 0,
        notes TEXT,
        recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_indexes_query = """
    CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends (symbol);
    CREATE INDEX IF NOT EXISTS idx_dividends_date ON dividends (dividend_date);
    """
    
    try:
        # Create the table
        table_result = execute_query(create_table_query, commit=True)
        if table_result:
            logger.info("Dividends table created or already exists.")
        else:
            logger.error("Failed to create dividends table.")
            return False
        
        # Create indexes
        index_result = execute_query(create_indexes_query, commit=True)
        if index_result:
            logger.info("Dividend indexes created or already exist.")
        else:
            logger.warning("Failed to create dividend indexes.")
        
        return True
    except Exception as e:
        logger.error(f"Error creating dividends table: {e}")
        return False

def update_transactions_for_drip():
    """Update the transactions table to support DRIP transactions if needed."""
    # Check if there are any DRIP transactions in the table already
    check_query = "SELECT COUNT(*) as count FROM transactions WHERE type = 'drip';"
    result = execute_query(check_query, fetchone=True)
    
    if result and result['count'] > 0:
        logger.info("DRIP transactions already exist in the transactions table.")
        return True
    
    logger.info("No DRIP transactions found. The table is ready for DRIP transactions.")
    return True

def main():
    """Run the migration."""
    logger.info("Starting dividends migration...")
    
    # Initialize database connection
    if not initialize_pool():
        logger.error("Failed to initialize database connection. Aborting.")
        return False
    
    # Ensure dividends table exists
    if not ensure_dividends_table():
        logger.error("Failed to ensure dividends table. Aborting.")
        return False
    
    # Update transactions table for DRIP support if needed
    if not update_transactions_for_drip():
        logger.warning("Issue updating transactions table for DRIP support.")
    
    logger.info("Dividends migration completed successfully.")
    return True

if __name__ == "__main__":
    if main():
        print("Dividends database migration completed successfully!")
        sys.exit(0)
    else:
        print("Dividends database migration failed!")
        sys.exit(1)