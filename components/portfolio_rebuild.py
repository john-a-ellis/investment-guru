# components/portfolio_rebuild.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

def create_portfolio_rebuild_component():
    """
    Creates a component for rebuilding portfolio from transaction history
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Rebuild Utility"),
        dbc.CardBody([
            html.P("This utility will rebuild your entire portfolio from transaction history. "
                  "It will recalculate share counts and book values based on the actual "
                  "sequence of transactions, fixing any discrepancies in the current portfolio data."),
            
            dbc.Alert(
                "Warning: This operation will modify your portfolio data. "
                "Consider backing up your database before proceeding.",
                color="warning",
                className="mb-3"
            ),
            
            dbc.Button(
                "Rebuild Portfolio from Transactions", 
                id="rebuild-portfolio-button",
                color="danger",
                className="me-2"
            ),
            
            dbc.Button(
                "Just Show Calculation (No Changes)", 
                id="calculate-portfolio-button",
                color="secondary",
                outline=True
            ),
            
            html.Div(id="rebuild-status", className="mt-3"),
            
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        html.Div(id="rebuild-details")
                    )
                ),
                id="rebuild-details-collapse",
                is_open=False
            )
        ])
    ])

def display_rebuild_results(results, made_changes=True):
    """
    Display the results of a portfolio rebuild operation
    
    Args:
        results (dict): Results from rebuild_portfolio_from_transactions
        made_changes (bool): Whether changes were actually made to the database
        
    Returns:
        Component: Dash component with rebuild results
    """
    if not results:
        return html.P("No results to display.")
    
    if 'error' in results:
        return dbc.Alert(f"Error during rebuild: {results['error']}", color="danger")
    
    # Create summary information
    operation_text = "Rebuild" if made_changes else "Calculation only"
    
    summary = dbc.Alert([
        html.H4(f"Portfolio {operation_text} Completed", className="alert-heading"),
        html.P([
            f"Changes {'made' if made_changes else 'identified'}: ",
            html.Strong(f"{results['changes_made']}")
        ]),
        html.Hr(),
        html.P([
            f"Positions updated: {results['positions_updated']}, ",
            f"Positions added: {results['positions_added']}, ",
            f"Positions removed: {results['positions_removed']}"
        ]),
        html.P(f"Operation completed at: {results['rebuild_time']}", className="mb-0")
    ], color="success" if made_changes else "info")
    
    # Create table of rebuilt positions
    position_rows = []
    for symbol, pos in results.get('rebuilt_positions', {}).items():
        avg_cost = pos['book_value'] / pos['shares'] if pos['shares'] > 0 else 0
        position_rows.append(html.Tr([
            html.Td(symbol),
            html.Td(f"{pos['shares']:.4f}"),
            html.Td(f"${avg_cost:.4f}"),
            html.Td(f"${pos['book_value']:.2f}"),
            html.Td(pos.get('currency', 'USD')),
            html.Td(pos.get('asset_type', 'stock'))
        ]))
    
    positions_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Avg Cost"),
                html.Th("Book Value"),
                html.Th("Currency"),
                html.Th("Asset Type")
            ])
        ),
        html.Tbody(position_rows)
    ], bordered=True, striped=True, size="sm", className="mb-4")
    
    # Create transaction log display
    log_rows = []
    for entry in results.get('transaction_log', [])[:50]:  # Limit to first 50 entries to avoid huge displays
        if 'error' in entry:
            action_style = {"color": "red", "font-weight": "bold"}
            action_text = f"{entry['action']}: {entry['error']}"
        else:
            action_style = {}
            action_text = entry['action']
            
            if entry['action'] == 'buy':
                action_style = {"color": "green"}
            elif entry['action'] == 'sell':
                action_style = {"color": "red"}
            elif entry['action'] == 'drip':
                action_style = {"color": "blue"}
        
        log_rows.append(html.Tr([
            html.Td(entry['date']),
            html.Td(entry['symbol']),
            html.Td(action_text, style=action_style),
            html.Td(f"{entry.get('shares_change', 0):.4f}"),
            html.Td(f"${entry.get('book_value_change', 0):.2f}"),
            html.Td(f"{entry.get('new_shares', 0):.4f}"),
            html.Td(f"${entry.get('new_book_value', 0):.2f}")
        ]))
    
    transaction_log = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("Symbol"),
                html.Th("Action"),
                html.Th("Shares Change"),
                html.Th("Book Value Change"),
                html.Th("New Shares"),
                html.Th("New Book Value")
            ])
        ),
        html.Tbody(log_rows)
    ], bordered=True, striped=True, size="sm", className="mb-4")
    
    # Show note if log was truncated
    log_note = None
    if len(results.get('transaction_log', [])) > 50:
        log_note = html.P(f"Note: Displaying first 50 of {len(results['transaction_log'])} log entries.", className="text-muted")
    
    return html.Div([
        summary,
        html.H5("Rebuilt Positions"),
        positions_table,
        html.H5("Transaction Processing Log"),
        log_note,
        transaction_log
    ])