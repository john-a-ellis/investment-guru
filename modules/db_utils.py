# modules/db_utils.py
"""
Database utility functions for the Investment Recommendation System.
Handles PostgreSQL database connections and operations.
"""
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PostgreSQL connection parameters - should be moved to environment variables in production
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'investment_system'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432')
}

# Create a connection pool
connection_pool = None

def initialize_pool(min_connections=1, max_connections=10):
    """
    Initialize the connection pool.
    Should be called at application startup.
    """
    global connection_pool
    try:
        connection_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            **DB_CONFIG
        )
        logger.info("Database connection pool initialized successfully.")
        
        # Test connection
        conn = get_connection()
        if conn:
            put_connection(conn)
            logger.info("Successfully connected to the database.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error initializing database connection pool: {e}")
        return False

def get_connection():
    """
    Get a connection from the pool.
    Remember to return the connection using put_connection()
    """
    global connection_pool
    if not connection_pool:
        initialize_pool()
    try:
        return connection_pool.getconn()
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        return None

def put_connection(conn):
    """
    Return a connection to the pool.
    """
    global connection_pool
    if connection_pool:
        connection_pool.putconn(conn)

def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    """
    Execute a SQL query and return the results.
    
    Args:
        query (str): SQL query to execute
        params (tuple or dict): Parameters for the query
        fetchone (bool): Whether to fetch one result
        fetchall (bool): Whether to fetch all results
        commit (bool): Whether to commit after executing
        
    Returns:
        The query results or None if there was an error
    """
    conn = get_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            
            result = None
            if fetchone:
                result = cur.fetchone()
            elif fetchall:
                result = cur.fetchall()
                
            if commit:
                conn.commit()
                
            return result
    except Exception as e:
        logger.error(f"Database error: {e}")
        if commit:
            conn.rollback()
        return None
    finally:
        put_connection(conn)

def initialize_database():
    """
    Initialize the database schema if it doesn't exist.
    Creates the necessary tables for the investment system.
    """
    # Create transactions table
    create_transactions_table_query = """
    CREATE TABLE IF NOT EXISTS transactions (
        id UUID PRIMARY KEY,
        type VARCHAR(10) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        price DECIMAL(16, 6) NOT NULL,
        shares DECIMAL(16, 6) NOT NULL,
        amount DECIMAL(16, 6) NOT NULL,
        transaction_date DATE NOT NULL,
        notes TEXT,
        recorded_at TIMESTAMP NOT NULL
    );
    """
    
    # Create portfolio table for tracking investments
    create_portfolio_table_query = """
    CREATE TABLE IF NOT EXISTS portfolio (
        id UUID PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        shares DECIMAL(16, 6) NOT NULL,
        purchase_price DECIMAL(16, 6) NOT NULL,
        purchase_date DATE NOT NULL,
        asset_type VARCHAR(20) NOT NULL,
        current_price DECIMAL(16, 6),
        current_value DECIMAL(16, 6),
        gain_loss DECIMAL(16, 6),
        gain_loss_percent DECIMAL(16, 6),
        currency VARCHAR(5) NOT NULL,
        added_date TIMESTAMP NOT NULL,
        last_updated TIMESTAMP
    );
    """
    
    # Create tracked assets table
    create_tracked_assets_table_query = """
    CREATE TABLE IF NOT EXISTS tracked_assets (
        symbol VARCHAR(20) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        type VARCHAR(20) NOT NULL,
        added_date DATE NOT NULL
    );
    """
    
    # Create user profile table
    create_user_profile_table_query = """
    CREATE TABLE IF NOT EXISTS user_profile (
        id SERIAL PRIMARY KEY,
        risk_level INTEGER NOT NULL,
        investment_horizon VARCHAR(20) NOT NULL,
        initial_investment DECIMAL(16, 2) NOT NULL,
        last_updated TIMESTAMP NOT NULL
    );
    """
    
    # Create mutual fund prices table
    create_mutual_fund_prices_table_query = """
    CREATE TABLE IF NOT EXISTS mutual_fund_prices (
        fund_code VARCHAR(20) NOT NULL,
        price_date DATE NOT NULL,
        price DECIMAL(16, 6) NOT NULL,
        added_at TIMESTAMP NOT NULL,
        PRIMARY KEY (fund_code, price_date)
    );
    """
    
    # Execute all creation queries
    for query in [
        create_transactions_table_query,
        create_portfolio_table_query,
        create_tracked_assets_table_query,
        create_user_profile_table_query,
        create_mutual_fund_prices_table_query
    ]:
        success = execute_query(query, commit=True)
        if success is None:
            logger.error(f"Failed to execute initialization query: {query}")
            return False
    
    # Add indexes for performance
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions(symbol);",
        "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);",
        "CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol);",
        "CREATE INDEX IF NOT EXISTS idx_mutual_fund_prices_fund_code ON mutual_fund_prices(fund_code);"
    ]
    
    for query in index_queries:
        execute_query(query, commit=True)
    
    return True

def migrate_json_to_db():
    """
    Migrate data from JSON files to PostgreSQL database.
    This function should be called once during the transition.
    """
    from modules.portfolio_utils import (
        load_portfolio, load_transactions, load_tracked_assets, load_user_profile
    )
    import json
    import uuid
    from datetime import datetime
    
    # Migrate transactions
    transactions = load_transactions()
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
        
        execute_query(insert_query, params, commit=True)
    
    # Migrate portfolio
    portfolio = load_portfolio()
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
            float(inv.get("current_price", 0)),
            float(inv.get("current_value", 0)),
            float(inv.get("gain_loss", 0)),
            float(inv.get("gain_loss_percent", 0)),
            inv.get("currency", "USD"),
            inv.get("added_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            inv.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        execute_query(insert_query, params, commit=True)
    
    # Migrate tracked assets
    assets = load_tracked_assets()
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
        
        execute_query(insert_query, params, commit=True)
    
    # Migrate user profile
    profile = load_user_profile()
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
            
            execute_query(insert_query, params, commit=True)
        else:
            # Update existing profile
            update_query = """
            UPDATE user_profile SET
                risk_level = %s,
                investment_horizon = %s,
                initial_investment = %s,
                last_updated = %s
            WHERE id = 1;
            """
            
            params = (
                profile.get("risk_level", 5),
                profile.get("investment_horizon", "medium"),
                float(profile.get("initial_investment", 10000)),
                profile.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            
            execute_query(update_query, params, commit=True)
    
    logger.info("Migration from JSON to database completed successfully.")
    return True