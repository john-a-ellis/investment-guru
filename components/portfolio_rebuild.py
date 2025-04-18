# components/portfolio_rebuild.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime

def create_portfolio_rebuild_component():
    """
    Creates a component for rebuilding the portfolio from transaction history
    with enhanced cash position reconciliation.
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Reconciliation & Rebuild"),
        dbc.CardBody([
            html.P([
                "This tool rebuilds your portfolio positions and cash balances based on all historical: ",
                html.Ul([
                    html.Li("Security transactions (buys/sells)"),
                    html.Li("Dividend payments and reinvestments"),
                    html.Li("Cash deposits and withdrawals"),
                    html.Li("Currency exchanges")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Calculate Discrepancies", 
                        id="calculate-portfolio-button", 
                        color="info", 
                        className="me-2"
                    ),
                    dbc.Button(
                        "Rebuild Portfolio", 
                        id="rebuild-portfolio-button", 
                        color="danger",
                        className="ms-2"
                    )
                ], width=12, className="mb-3")
            ]),
            dbc.Alert(
                "This will recalculate both investment positions and cash balances based on all recorded transactions. "
                "The rebuild operation will overwrite current positions and cash balances.",
                color="warning"
            ),
            dbc.Alert(id="rebuild-status"),
            dbc.Collapse([
                html.H5("Rebuild Details", className="mt-3"),
                html.Div(id="rebuild-details")
            ], id="rebuild-details-collapse", is_open=False)
        ])
    ])

def display_rebuild_results(results, made_changes=True):
    """
    Create a detailed display of rebuild results, including cash position changes.
    
    Args:
        results (dict): Results from the enhanced_rebuild_portfolio function
        made_changes (bool): Whether actual changes were made to the database
    
    Returns:
        Component: Dash component with detailed results
    """
    action_text = "made" if made_changes else "would make"
    
    # Create summary card
    summary_card = dbc.Card([
        dbc.CardBody([
            html.H5("Summary of Changes", className="card-title"),
            html.P([
                f"The rebuild {action_text} the following changes:",
                html.Ul([
                    html.Li(f"Positions Added: {results['positions_added']}"),
                    html.Li(f"Positions Updated: {results['positions_updated']}"),
                    html.Li(f"Positions Removed: {results['positions_removed']}"),
                    html.Li(f"Cash Balances Updated: {results['cash_updated']}")
                ])
            ])
        ])
    ], className="mb-3")
    
    # Create cash positions card
    cash_rows = []
    for currency, balance in results['cash_positions'].items():
        if currency in results.get('cash_discrepancies', {}):
            discrepancy = results['cash_discrepancies'][currency]
            difference = discrepancy['difference']
            color = "text-success" if abs(difference) < 0.01 else "text-danger"
            
            cash_rows.append(
                html.Tr([
                    html.Td(currency),
                    html.Td(f"${discrepancy['current']:.2f}"),
                    html.Td(f"${discrepancy['calculated']:.2f}"),
                    html.Td(f"${difference:.2f}", className=color)
                ])
            )
        else:
            cash_rows.append(
                html.Tr([
                    html.Td(currency),
                    html.Td("N/A"),
                    html.Td(f"${balance:.2f}"),
                    html.Td("No change", className="text-muted")
                ])
            )
    
    cash_card = dbc.Card([
        dbc.CardHeader("Cash Position Changes"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Currency"),
                        html.Th("Current Balance"),
                        html.Th("Calculated Balance"),
                        html.Th("Difference")
                    ])
                ),
                html.Tbody(cash_rows)
            ], bordered=True, hover=True, striped=True)
        ])
    ], className="mb-3")
    
    # Create position changes card
    position_data = []  # We'll store the data first, then create rows
    for symbol, position in results['rebuilt_positions'].items():
        # Format shares and book value
        shares = position.get('shares', 0)
        book_value = position.get('book_value', 0)
        
        position_data.append({
            'symbol': symbol,
            'currency': position.get('currency', 'USD'),
            'shares': shares,
            'book_value': book_value,
            'avg_cost': book_value / shares if shares > 0 else 0
        })
    
    # Sort by symbol before creating the rows
    position_data.sort(key=lambda x: x['symbol'])
    
    # Now create the rows in sorted order
    position_rows = []
    for data in position_data:
        position_rows.append(
            html.Tr([
                html.Td(data['symbol']),
                html.Td(data['currency']),
                html.Td(f"{data['shares']:.4f}"),
                html.Td(f"${data['book_value']:.2f}"),
                html.Td(f"${data['avg_cost']:.4f}" if data['shares'] > 0 else "N/A")
            ])
        )
    
    position_card = dbc.Card([
        dbc.CardHeader("Investment Position Changes"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Currency"),
                        html.Th("Shares"),
                        html.Th("Book Value"),
                        html.Th("Avg. Cost")
                    ])
                ),
                html.Tbody(position_rows)
            ], bordered=True, hover=True, striped=True, responsive=True)
        ])
    ])
    
    # Create a collapsible event log for advanced users
    event_log_items = []
    for event in results.get('event_log', [])[:30]:  # Limit to first 30 events to avoid massive display
        event_date = event.get('date', 'Unknown')
        event_type = event.get('type', 'Unknown')
        event_details = event.get('details', {})
        
        if event_type == 'transaction':
            subtype = event_details.get('subtype', 'Unknown')
            symbol = event_details.get('symbol', 'Unknown')
            shares = event_details.get('shares', 0)
            price = event_details.get('price', 0)
            
            description = f"{subtype.capitalize()} {shares} shares of {symbol} at ${price:.2f}"
        elif event_type == 'cash_flow':
            subtype = event_details.get('subtype', 'Unknown')
            amount = event_details.get('amount', 0)
            currency = event_details.get('currency', 'Unknown')
            
            description = f"{subtype.capitalize()} of {currency} {amount:.2f}"
        elif event_type == 'dividend':
            symbol = event_details.get('symbol', 'Unknown')
            amount = event_details.get('total_amount', 0)
            currency = event_details.get('currency', 'Unknown')
            is_drip = event_details.get('is_drip', False)
            
            description = f"Dividend from {symbol}: {currency} {amount:.2f}" + (" (DRIP)" if is_drip else "")
        elif event_type == 'currency_exchange':
            from_currency = event_details.get('from_currency', 'Unknown')
            from_amount = event_details.get('from_amount', 0)
            to_currency = event_details.get('to_currency', 'Unknown')
            to_amount = event_details.get('to_amount', 0)
            
            description = f"Exchange: {from_amount:.2f} {from_currency} → {to_amount:.2f} {to_currency}"
        
        event_log_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"{event_date} - {event_type.capitalize()}"),
                    html.Span(description, className="ms-2")
                ]),
                html.Small([
                    "Cash After: ",
                    ", ".join([f"{curr}: ${bal:.2f}" for curr, bal in event.get('cash_after', {}).items()])
                ], className="text-muted d-block")
            ])
        )
    
    event_log = dbc.Card([
        dbc.CardHeader([
            html.H5(
                "Event Processing Log ", 
                className="d-inline"
            ),
            dbc.Badge(f"{len(results.get('event_log', []))} Events", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            html.P("This shows how each financial event was processed during the rebuild (limited to first 30 events):"),
            dbc.ListGroup(event_log_items, className="mb-3"),
            html.Div(
                "Note: Full event log is available in the server logs if needed for troubleshooting.",
                className="text-muted small"
            )
        ])
    ], className="mt-3")
    
    return html.Div([
        summary_card,
        cash_card,
        position_card,
        dbc.Collapse(
            event_log,
            id="event-log-collapse",
            is_open=False
        ),
        dbc.Button(
            "Show Processing Log",
            id="toggle-event-log",
            color="secondary",
            className="mt-2"
        )
    ])