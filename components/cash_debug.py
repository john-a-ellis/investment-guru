# components/cash_debug.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

def create_cash_debug_component():
    """
    Creates a debug component for investigating cash position issues
    """
    return dbc.Card([
        dbc.CardHeader([
            "Cash Position Debugging",
            dbc.Button(
                "Refresh", 
                id="refresh-cash-debug", 
                color="primary", 
                size="sm", 
                className="float-end"
            )
        ]),
        dbc.CardBody([
            html.Div(id="cash-debug-content"),
            dcc.Interval(
                id="cash-debug-interval",
                interval=10000,  # 10 seconds
                n_intervals=0
            )
        ])
    ])

def display_cash_debug_info(debug_data):
    """
    Display cash position debug information
    
    Args:
        debug_data (dict): Debug data from debug_cash_positions
        
    Returns:
        Component: Dash component with debug information
    """
    if not debug_data or 'error' in debug_data:
        return dbc.Alert(
            f"Error retrieving debug data: {debug_data.get('error', 'Unknown error')}", 
            color="danger"
        )
    
    # Display cash positions
    cash_positions_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("ID"),
                html.Th("Currency"),
                html.Th("Balance"),
                html.Th("Last Updated")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(pos['id']),
                html.Td(pos['currency']),
                html.Td(f"${pos['balance']:.2f}", 
                       style={"color": "green" if pos['balance'] >= 0 else "red"}),
                html.Td(pos['last_updated'])
            ]) for pos in debug_data.get('cash_positions', [])
        ])
    ], bordered=True, striped=True, size="sm", className="mb-3")
    
    # Display recent transactions
    transactions_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Type"),
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Price"),
                html.Th("Amount"),
                html.Th("Date")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(tx['type'].capitalize(), style={"color": "green" if tx['type'] == 'sell' else "red"}),
                html.Td(tx['symbol']),
                html.Td(f"{tx['shares']:.4f}"),
                html.Td(f"${tx['price']:.2f}"),
                html.Td(f"${tx['amount']:.2f}"),
                html.Td(tx['date'])
            ]) for tx in debug_data.get('recent_transactions', [])
        ])
    ], bordered=True, striped=True, size="sm", className="mb-3")
    
    # Display recent cash flows
    flows_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Type"),
                html.Th("Amount"),
                html.Th("Currency"),
                html.Th("Date")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(flow['type'].capitalize(), 
                       style={"color": "green" if flow['type'] == 'deposit' else "red"}),
                html.Td(f"${flow['amount']:.2f}"),
                html.Td(flow['currency']),
                html.Td(flow['date'])
            ]) for flow in debug_data.get('recent_cash_flows', [])
        ])
    ], bordered=True, striped=True, size="sm", className="mb-3")
    
    # Display recent currency exchanges
    exchanges_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("From"),
                html.Th("To"),
                html.Th("Rate"),
                html.Th("Date")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(f"{ex['from_amount']:.2f} {ex['from_currency']}"),
                html.Td(f"{ex['to_amount']:.2f} {ex['to_currency']}"),
                html.Td(f"{ex['rate']:.4f}"),
                html.Td(ex['date'])
            ]) for ex in debug_data.get('recent_exchanges', [])
        ])
    ], bordered=True, striped=True, size="sm", className="mb-3")
    
    return html.Div([
        html.H5("Current Cash Positions"),
        cash_positions_table,
        
        html.H5("Recent Transactions (Should Impact Cash)"),
        transactions_table,
        
        html.H5("Recent Cash Flows"),
        flows_table,
        
        html.H5("Recent Currency Exchanges"),
        exchanges_table,
        
        html.Small(f"Debug info as of: {debug_data.get('debug_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", className="text-muted")
    ])