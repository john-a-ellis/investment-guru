# components/book_value_debug.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

def create_book_value_debug_component():
    """
    Creates a debug component for investigating book value/cost basis issues
    """
    return dbc.Card([
        dbc.CardHeader([
            "Book Value Debugging",
            dbc.Button(
                "Refresh", 
                id="refresh-book-value-debug", 
                color="primary", 
                size="sm", 
                className="float-end"
            )
        ]),
        dbc.CardBody([
            html.Div(id="book-value-debug-content"),
            dcc.Interval(
                id="book-value-debug-interval",
                interval=10000,  # 10 seconds
                n_intervals=0
            )
        ])
    ])

# Fixed display_book_value_debug_info function for components/book_value_debug.py

def display_book_value_debug_info(debug_data):
    """
    Display book value debug information
    
    Args:
        debug_data (dict): Debug data from debug_book_value()
        
    Returns:
        Component: Dash component with debug information
    """
    if not debug_data:
        return html.P("No debug data available.")
    
    if 'error' in debug_data:
        return dbc.Alert(
            f"Error retrieving debug data: {debug_data.get('error', 'Unknown error')}", 
            color="danger"
        )
    
    # Create the summary table of discrepancies
    discrepancy_rows = []
    
    for symbol, data in debug_data.get('book_values', {}).items():
        # Skip zero-share positions
        if data.get('shares', 0) <= 0:
            continue
            
        # Format the status with appropriate color
        status_style = {}
        badge_color = "success"
        
        # Check if 'status' key exists and handle properly
        status = data.get('status', 'unknown')
        
        if status == 'minor_discrepancy':
            badge_color = "warning"
            status_style = {"color": "orange"}
        elif status == 'major_discrepancy' or status == 'unknown':
            badge_color = "danger"
            status_style = {"color": "red"}
        
        # Calculate the average cost
        avg_cost = data.get('book_value', 0) / data.get('shares', 1) if data.get('shares', 0) > 0 else 0
        actual_avg_cost = data.get('actual_avg_cost', 0)
        
        discrepancy_rows.append(html.Tr([
            html.Td(symbol),
            html.Td(f"{data.get('shares', 0):.4f}"),
            html.Td(f"${avg_cost:.4f}"),
            html.Td(f"${data.get('book_value', 0):.2f}"),
            html.Td(f"{data.get('actual_shares', 0):.4f}"),
            html.Td(f"${actual_avg_cost:.4f}"),
            html.Td(f"${data.get('actual_book_value', 0):.2f}"),
            html.Td(f"{data.get('shares_diff', 0):.4f}", style=status_style),
            html.Td(f"${data.get('avg_cost_diff', 0):.4f}", style=status_style),
            html.Td(f"${data.get('book_value_diff', 0):.2f}", style=status_style),
            html.Td(dbc.Badge(status.replace('_', ' ').title(), color=badge_color))
        ]))
    
    # Add any investments that exist in portfolio but have no transactions
    for item in debug_data.get('extra_investments', []):
        discrepancy_rows.append(html.Tr([
            html.Td(item.get('symbol', '')),
            html.Td("0.0000"),  # Expected shares
            html.Td("$0.0000"),  # Expected avg cost
            html.Td("$0.00"),  # Expected book value
            html.Td(f"{item.get('shares', 0):.4f}"),  # Actual shares
            html.Td(f"${item.get('purchase_price', 0):.4f}"),  # Actual avg cost
            html.Td(f"${item.get('book_value', 0):.2f}"),  # Actual book value
            html.Td(f"{item.get('shares', 0):.4f}", style={"color": "red"}),  # Shares diff
            html.Td(f"${item.get('purchase_price', 0):.4f}", style={"color": "red"}),  # Avg cost diff
            html.Td(f"${item.get('book_value', 0):.2f}", style={"color": "red"}),  # Book value diff
            html.Td(dbc.Badge("No Transactions", color="danger"))
        ]))
        
    # Create the main summary table
    discrepancy_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbol"),
                html.Th("Expected Shares"),
                html.Th("Expected Avg Cost"),
                html.Th("Expected Book Value"),
                html.Th("Actual Shares"),
                html.Th("Actual Avg Cost"),
                html.Th("Actual Book Value"),
                html.Th("Shares Diff"),
                html.Th("Avg Cost Diff"),
                html.Th("Book Value Diff"),
                html.Th("Status")
            ])
        ),
        html.Tbody(discrepancy_rows)
    ], bordered=True, striped=True, size="sm", className="mb-4")
    
    # Create transaction details for each symbol
    transaction_details = []
    
    for symbol, data in debug_data.get('book_values', {}).items():
        # Only show transaction details for positions with issues
        # Check for both status and transactions
        if data.get('transactions') and (
            data.get('status', '') != 'ok' or 
            abs(data.get('shares_diff', 0)) > 0.0001 or 
            abs(data.get('book_value_diff', 0)) > 0.01
        ):
            # Create transaction table
            tx_rows = []
            
            for tx in data.get('transactions', []):
                tx_rows.append(html.Tr([
                    html.Td(tx.get('date', '')),
                    html.Td(tx.get('type', '').capitalize(), 
                           style={"color": "green" if tx.get('type') == 'buy' else "red"}),
                    html.Td(f"{tx.get('shares', 0):.4f}"),
                    html.Td(f"${tx.get('price', 0):.4f}"),
                    html.Td(f"${tx.get('amount', 0):.2f}"),
                    html.Td(f"${tx.get('book_value_sold', 0):.2f}" if tx.get('type') == 'sell' else ""),
                    html.Td(f"{tx.get('running_shares', 0):.4f}"),
                    html.Td(f"${tx.get('running_book_value', 0):.2f}"),
                    html.Td(f"${tx.get('running_book_value', 0) / tx.get('running_shares', 1):.4f}" 
                           if tx.get('running_shares', 0) > 0 else "$0.0000")
                ]))
            
            if tx_rows:
                tx_table = dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Date"),
                            html.Th("Type"),
                            html.Th("Shares"),
                            html.Th("Price"),
                            html.Th("Amount"),
                            html.Th("Book Value Sold"),
                            html.Th("Running Shares"),
                            html.Th("Running Book Value"),
                            html.Th("Running Avg Cost")
                        ])
                    ),
                    html.Tbody(tx_rows)
                ], bordered=True, size="sm")
                
                transaction_details.append(html.Div([
                    html.H5(f"Transaction History for {symbol}", className="mt-3"),
                    tx_table
                ]))
    
    return html.Div([
        html.H4("Book Value Discrepancy Analysis"),
        discrepancy_table if discrepancy_rows else 
            dbc.Alert("No book value discrepancies found.", color="success"),
        
        html.Hr(),
        
        html.H4("Transaction Details for Problematic Symbols"),
        html.Div(transaction_details) if transaction_details else 
            html.P("No transaction details to display."),
        
        html.Small(f"Debug info as of: {debug_data.get('debug_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", className="text-muted")
    ])