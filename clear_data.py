# Create a clear_data.py script:

import os
from modules.db_utils import execute_query, initialize_pool

def clear_all_tables():
    """Clear all data from tables while preserving the schema."""
    
    # Initialize the connection pool
    initialize_pool()
    
    # List of tables to clear (in order to respect foreign keys)
    tables = [
        'cash_flows', 
        'cash_positions',
        'mutual_fund_prices',
        'portfolio_snapshots',
        'trained_models',
        'transactions',
        'portfolio',
        'tracked_assets',
        'target_allocation',
        'user_profile'
    ]
    
    for table in tables:
        try:
            # Truncate clears the table faster than DELETE but resets all sequences
            query = f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"
            execute_query(query, commit=True)
            print(f"Cleared table: {table}")
        except Exception as e:
            print(f"Error clearing table {table}: {e}")
    
    print("All tables cleared successfully.")

if __name__ == "__main__":
    confirm = input("This will DELETE ALL DATA in all tables. Type 'yes' to confirm: ")
    if confirm.lower() == 'yes':
        clear_all_tables()
        print("All data has been cleared. Database structure has been preserved.")
    else:
        print("Operation cancelled.")