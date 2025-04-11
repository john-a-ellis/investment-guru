# modules/db_utils.py
import os
import psycopg2
import psycopg2.pool
import psycopg2.extras # Needed for DictCursor
from dotenv import load_dotenv
import logging
import json # Added
import pandas as pd # Added
from datetime import datetime # Added
import traceback # Added for logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose output during testing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global connection pool variable
pool = None

def initialize_pool():
    """Initialize the PostgreSQL connection pool."""
    global pool
    if pool is None:
        try:
            db_config = {
                'dbname': os.environ.get('DB_NAME', 'investment_system'),
                'user': os.environ.get('DB_USER', 'postgres'),
                'password': os.environ.get('DB_PASSWORD', 'postgres'),
                'host': os.environ.get('DB_HOST', 'localhost'),
                'port': os.environ.get('DB_PORT', '5432')
            }
            logger.info(f"Initializing connection pool with config: {db_config}")
            pool = psycopg2.pool.SimpleConnectionPool(1, 10, **db_config)
            logger.info("Connection pool initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}")
            return False
    return True

def get_connection():
    """Get a connection from the pool."""
    global pool
    if pool is None:
        if not initialize_pool():
            raise Exception("Failed to initialize connection pool")
    try:
        return pool.getconn()
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        raise

def release_connection(conn):
    """Release a connection back to the pool."""
    global pool
    if pool and conn:
        pool.putconn(conn)

def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    """Execute a SQL query using the connection pool."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- Log the query being executed ---
            logger.debug(f"Executing SQL: {cur.mogrify(query, params).decode('utf-8') if params else query}")
            cur.execute(query, params)

            if commit:
                conn.commit()
                logger.debug("Commit successful.")
                return True # Indicate success for commit

            if fetchone:
                result = cur.fetchone()
                logger.debug(f"Fetchone result: {result}")
                return result
            elif fetchall:
                results = cur.fetchall()
                logger.debug(f"Fetchall results count: {len(results)}")
                return results
            else:
                # Successful execution without fetch/commit
                return None

    except (psycopg2.Error, Exception) as e: # Catch specific DB errors too
        logger.error(f"Database query error: {e}\nQuery: {query}\nParams: {params}")
        logger.error(traceback.format_exc()) # Log full traceback
        if conn:
             try:
                 conn.rollback() # Rollback on error
                 logger.info("Transaction rolled back due to error.")
             except Exception as rb_e:
                 logger.error(f"Error during rollback: {rb_e}")
        if commit:
            return False # Indicate failure for commit
        return None # Return None for fetch errors or other non-commit errors

    finally:
        if conn:
            release_connection(conn)

def initialize_database():
    """Create necessary database tables if they don't exist."""
    logger.info("Initializing database schema...")
    
    # List of table creation queries
    create_table_queries = [
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            type VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
            symbol VARCHAR(50) NOT NULL,
            price NUMERIC(15, 4) NOT NULL,
            shares NUMERIC(15, 6) NOT NULL,
            amount NUMERIC(15, 4), -- Calculated: price * shares
            transaction_date DATE NOT NULL,
            notes TEXT,
            recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(50) UNIQUE NOT NULL, -- Assuming one entry per symbol
            shares NUMERIC(15, 6) NOT NULL,
            purchase_price NUMERIC(15, 4), -- Average purchase price might be better here
            purchase_date DATE, -- Date of first purchase or weighted average date
            asset_type VARCHAR(50) DEFAULT 'stock',
            current_price NUMERIC(15, 4),
            current_value NUMERIC(15, 4),
            gain_loss NUMERIC(15, 4),
            gain_loss_percent NUMERIC(10, 4),
            currency VARCHAR(10) DEFAULT 'USD',
            added_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP WITH TIME ZONE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tracked_assets (
            symbol VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255),
            type VARCHAR(50) DEFAULT 'stock',
            added_date DATE DEFAULT CURRENT_DATE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            id SERIAL PRIMARY KEY, -- Use SERIAL for auto-incrementing integer PK
            risk_level INTEGER DEFAULT 5,
            investment_horizon VARCHAR(50) DEFAULT 'medium',
            initial_investment NUMERIC(15, 2) DEFAULT 10000.00,
            last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS mutual_fund_prices (
            id SERIAL PRIMARY KEY,
            fund_code VARCHAR(50) NOT NULL,
            price_date DATE NOT NULL,
            price NUMERIC(15, 4) NOT NULL,
            added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (fund_code, price_date) -- Ensure only one price per fund per day
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS target_allocation (
            asset_type VARCHAR(50) PRIMARY KEY,
            percentage NUMERIC(5, 2) NOT NULL CHECK (percentage >= 0 AND percentage <= 100)
        );
        """,
        # --- NEW TABLE FOR TRAINED MODELS ---
        """
        CREATE TABLE IF NOT EXISTS trained_models (
            id SERIAL PRIMARY KEY,
            model_filename VARCHAR(255) UNIQUE NOT NULL, -- e.g., 'LSTM_AAPL.pkl'
            symbol VARCHAR(50) NOT NULL,                 -- e.g., 'AAPL'
            model_type VARCHAR(50),                      -- e.g., 'LSTM', 'ARIMA'
            training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metrics JSONB,                               -- Store performance metrics as JSON
            notes TEXT                                   -- Optional notes
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS trained_models (
            id SERIAL PRIMARY KEY,
            model_filename VARCHAR(255) UNIQUE NOT NULL,
            symbol VARCHAR(50) NOT NULL,
            model_type VARCHAR(50),
            training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metrics JSONB,
            notes TEXT
        );
        """,
        # --- INDEXES (Optional but recommended) ---
        """CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions (symbol);""",
        """CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions (transaction_date);""",
        """CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio (symbol);""",
        """CREATE INDEX IF NOT EXISTS idx_mutual_fund_prices_code_date ON mutual_fund_prices (fund_code, price_date);""",
        """CREATE INDEX IF NOT EXISTS idx_trained_models_symbol ON trained_models (symbol);""", # New Index
        """CREATE INDEX IF NOT EXISTS idx_trained_models_training_date ON trained_models (training_date);""" # New Index
        """CREATE INDEX IF NOT EXISTS idx_trained_models_symbol ON trained_models (symbol);""",
        """CREATE INDEX IF NOT EXISTS idx_trained_models_training_date ON trained_models (training_date);"""
        ]
    
    # Ensure UUID extension is enabled
    enable_uuid_query = "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
    conn = None
    success = True
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            logger.info("Ensuring pgcrypto extension is enabled...")
            cur.execute(enable_uuid_query)
            conn.commit()
            logger.info("Executing table creation queries...")
            for query in create_table_queries:
                try:
                    cur.execute(query)
                    logger.debug(f"Successfully executed: {query.splitlines()[1].strip()}...")
                except Exception as table_e:
                    logger.error(f"Error executing query: {query}\nError: {table_e}")
                    conn.rollback()
                    success = False
                    break
            if success:
                conn.commit()
                logger.info("Database schema initialized/verified successfully.")
            else:
                 logger.error("Database schema initialization failed due to errors.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        if conn: conn.rollback()
        success = False
    finally:
        if conn: release_connection(conn)
    return success

# --- NEW FUNCTION TO GET MODEL DATA ---
def get_trained_models_data():
    """Retrieves metadata for all trained models from the database."""
    logger.info("Attempting to fetch trained models data...")
    query = """
    SELECT model_filename, symbol, model_type, training_date, metrics, notes
    FROM trained_models ORDER BY training_date DESC;
    """
    try:
        results = execute_query(query, fetchall=True)
        logger.debug(f"Raw results from DB for trained models: {results}")
        if results:
            data = [dict(row) for row in results]
            df = pd.DataFrame(data)
            logger.info(f"DataFrame created. Head:\n{df.head().to_string()}")
            def safe_json_loads(x):
                if x is None: return {}
                try:
                    return json.loads(x) if isinstance(x, str) else x # Handle if already dict
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse metrics JSON: {x}. Error: {e}")
                    return {'error': 'invalid format'}
            if 'metrics' in df.columns:
                 logger.debug("Parsing metrics JSON...")
                 df['metrics'] = df['metrics'].apply(safe_json_loads)
                 logger.debug("Metrics JSON parsing complete.")
            if 'training_date' in df.columns:
                logger.debug("Formatting training_date...")
                df['training_date'] = pd.to_datetime(df['training_date'])
                df['training_date'] = df['training_date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('N/A')
                logger.debug("Date formatting complete.")
            logger.info(f"Returning DataFrame with {len(df)} trained models.")
            return df
        else:
            logger.info("No trained models found in the database (execute_query returned no results).")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching or processing trained models data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# --- Refactored save_model_metadata with Enhanced Logging ---
def save_model_metadata(filename, symbol, model_type, metrics, notes=None):
    """
    Saves metadata about a trained model to the database.
    Includes enhanced logging for debugging.
    """
    logger.info(f"Attempting to save metadata for: {filename}")
    logger.debug(f"Received params: symbol={symbol}, type={model_type}, metrics={metrics}, notes={notes}")

    insert_query = """
    INSERT INTO trained_models (model_filename, symbol, model_type, training_date, metrics, notes)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (model_filename) DO UPDATE SET
        symbol = EXCLUDED.symbol,
        model_type = EXCLUDED.model_type,
        training_date = EXCLUDED.training_date,
        metrics = EXCLUDED.metrics,
        notes = EXCLUDED.notes;
    """
    # Ensure metrics are stored as a JSON string
    try:
        # --- Ensure metrics is a dict before dumping ---
        if not isinstance(metrics, dict):
             logger.error(f"Metrics data is not a dictionary for {filename}. Type: {type(metrics)}. Data: {metrics}")
             # Optionally try to convert if it's a string representation, or just fail
             return False
        metrics_json = json.dumps(metrics)
        logger.debug(f"Metrics JSON for {filename}: {metrics_json}")
    except TypeError as e:
        logger.error(f"Could not serialize metrics dictionary to JSON for {filename}: {metrics}. Error: {e}")
        return False
    except Exception as json_e: # Catch broader JSON errors
         logger.error(f"Unexpected error serializing metrics for {filename}: {metrics}. Error: {json_e}")
         return False

    params = (
        filename,
        symbol.upper(),
        model_type,
        datetime.now(), # Use current time for training_date
        metrics_json,
        notes
    )
    logger.debug(f"Executing save_model_metadata query for {filename} with params: {params}")

    try:
        # --- Call execute_query with commit=True ---
        success = execute_query(insert_query, params, commit=True)
        logger.info(f"execute_query result for saving metadata {filename}: {success}")

        if success: # execute_query returns True on successful commit
            logger.info(f"Metadata for model {filename} saved to database.")
            return True
        else:
            # execute_query logs the specific DB error if commit fails or exception occurs
            logger.error(f"Failed to save metadata for model {filename} to database (execute_query returned False/None). Check previous DB error logs.")
            return False
    except Exception as e:
        # This catches errors if execute_query itself raises an exception before execution
        logger.error(f"Exception during execute_query call for {filename} metadata: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        return False


# --- END OF db_utils.py modifications ---
