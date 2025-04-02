#!/usr/bin/env python3
"""
Database Diagnostic Tool for Investment Recommendation System

This script tests direct PostgreSQL connections and operations to help
diagnose issues with database initialization.

Usage:
    python db_diagnostic.py
"""

import os
import psycopg2
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_direct_connection():
    """Test a direct connection to PostgreSQL and table creation"""
    # Get connection parameters from environment or use defaults
    db_config = {
        'dbname': os.environ.get('DB_NAME', 'investment_system'),
        'user': os.environ.get('DB_USER', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', 'postgres'),
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': 5432
    }
    
    logger.info(f"Testing connection with parameters: {db_config}")
    
    try:
        # Try to connect
        conn = psycopg2.connect(**db_config)
        logger.info("Database connection successful!")
        
        # Check if we can create a test table
        with conn.cursor() as cur:
            logger.info("Attempting to create a test table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL
                );
            """)
            conn.commit()
            logger.info("Test table created successfully!")
            
            # Test if we can insert data
            logger.info("Testing data insertion...")
            cur.execute("INSERT INTO test_table (name) VALUES (%s)", ("Test Name",))
            conn.commit()
            logger.info("Data inserted successfully!")
            
            # Test if we can query data
            logger.info("Testing data retrieval...")
            cur.execute("SELECT * FROM test_table")
            result = cur.fetchone()
            logger.info(f"Retrieved data: {result}")
            
            # Drop the test table to clean up
            logger.info("Cleaning up test table...")
            cur.execute("DROP TABLE test_table")
            conn.commit()
            logger.info("Test table dropped successfully!")
        
        conn.close()
        logger.info("All database operations successful!")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

# Also, let's examine the execute_query function in your modules/db_utils.py
def analyze_execute_query():
    """Analyze the execute_query function from db_utils.py"""
    try:
        # Import the function - assuming it's in the expected path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from modules.db_utils import execute_query
        
        logger.info("Successfully imported execute_query function")
        
        # Test a simple query
        logger.info("Testing execute_query with a simple SELECT...")
        result = execute_query("SELECT 1 as test", fetchone=True)
        logger.info(f"Result from execute_query: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing execute_query: {e}")
        return False

if __name__ == "__main__":
    direct_test_result = test_direct_connection()
    logger.info(f"Direct connection test {'PASSED' if direct_test_result else 'FAILED'}")
    
    execute_query_test = analyze_execute_query()
    logger.info(f"execute_query analysis {'PASSED' if execute_query_test else 'FAILED'}")