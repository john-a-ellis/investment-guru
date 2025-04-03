# modules/db_utils.py
"""
Database utility functions for the Investment Recommendation System.
Handles PostgreSQL database connections and operations.
"""
import os
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

# PostgreSQL connection parameters from environment variables
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'investment_system'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': 5432  # Hardcoded default port
}

# If DB_PORT is in environment variables, try to use it
if 'DB_PORT' in os.environ:
    try:
        DB_CONFIG['port'] = int(os.environ.get('DB_PORT'))
    except ValueError:
        # If conversion fails, log error and keep default
        logger.error(f"Invalid DB_PORT value: {os.environ.get('DB_PORT')}. Using default: 5432")

# Create a connection pool
connection_pool = None

def initialize_pool(min_connections=1, max_connections=20):
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
    Execute a SQL query and return the results with better error handling.
    
    Args:
        query (str): SQL query to execute
        params (tuple or dict): Parameters for the query
        fetchone (bool): Whether to fetch one result
        fetchall (bool): Whether to fetch all results
        commit (bool): Whether to commit after executing
        
    Returns:
        The query results or True/None if there was an error
    """
    conn = get_connection()
    if not conn:
        logger.error("Could not get database connection")
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            logger.debug(f"Executing query: {query}")
            if params:
                logger.debug(f"With parameters: {params}")
            
            cur.execute(query, params)
            
            result = None
            if fetchone:
                result = cur.fetchone()
                logger.debug(f"Fetched one result: {result}")
            elif fetchall:
                result = cur.fetchall()
                logger.debug(f"Fetched all results: {len(result) if result else 0} rows")
                
            if commit:
                logger.debug("Committing transaction")
                conn.commit()
            
            # Return True for successful non-fetch operations that require commit
            if commit and not fetchone and not fetchall:
                return True
                
            return result
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        logger.error(f"Query was: {query}")
        if params:
            logger.error(f"Parameters were: {params}")
        
        if commit:
            logger.error("Rolling back transaction")
            conn.rollback()
        return None
    finally:
        put_connection(conn)

def initialize_database():
    """
    Initialize the database schema if it doesn't exist.
    Creates the necessary tables for the investment system.
    """
    conn = None
    try:
        # Get a direct connection for more control
        conn = get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        cur = conn.cursor()
        
        # Create transactions table
        logger.info("Creating transactions table...")
        cur.execute("""
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
        """)
        
        # Create portfolio table for tracking investments
        logger.info("Creating portfolio table...")
        cur.execute("""
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
        """)
        
        # Create tracked assets table
        logger.info("Creating tracked assets table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tracked_assets (
            symbol VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(20) NOT NULL,
            added_date DATE NOT NULL
        );
        """)
        
        # Create user profile table
        logger.info("Creating user profile table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id SERIAL PRIMARY KEY,
            risk_level INTEGER NOT NULL,
            investment_horizon VARCHAR(20) NOT NULL,
            initial_investment DECIMAL(16, 2) NOT NULL,
            last_updated TIMESTAMP NOT NULL
        );
        """)
        
        # Create mutual fund prices table
        logger.info("Creating mutual fund prices table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mutual_fund_prices (
            fund_code VARCHAR(20) NOT NULL,
            price_date DATE NOT NULL,
            price DECIMAL(16, 6) NOT NULL,
            added_at TIMESTAMP NOT NULL,
            PRIMARY KEY (fund_code, price_date)
        );
        """)
        
        # Add indexes for performance
        logger.info("Creating indexes...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions(symbol);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mutual_fund_prices_fund_code ON mutual_fund_prices(fund_code);")
        
        # Commit all changes
        conn.commit()
        logger.info("Database schema created successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if conn:
            put_connection(conn)

# Remove the problematic migrate_json_to_db function since we're replacing it
# with a better implementation in initialize_db.py