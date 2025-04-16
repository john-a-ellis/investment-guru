# components/dividend_management.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime

def create_dividend_management_component():
    """
    Creates a component for recording and analyzing dividend income
    """
    return dbc.Card([
        dbc.CardHeader("Dividend Management"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Record Dividend Form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbol"),
                                dbc.Input(id="dividend-symbol-input", placeholder="e.g., MFC.TO")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Dividend Date"),
                                dbc.Input(id="dividend-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Record Date (Optional)"),
                                dbc.Input(id="dividend-record-date-input", type="date", placeholder="Record Date")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Currency"),
                                dbc.Select(
                                    id="dividend-currency-select",
                                    options=[
                                        {"label": "CAD", "value": "CAD"},
                                        {"label": "USD", "value": "USD"}
                                    ],
                                    value=""
                                )
                            ], width=3)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Amount Per Share"),
                                dbc.Input(id="dividend-amount-input", type="number", placeholder="Dividend per share", step="0.001")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shares Held"),
                                dbc.Input(id="dividend-shares-input", type="number", placeholder="Leave blank to use portfolio")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Total Amount (Calculated)"),
                                html.Div(id="dividend-total-display", className="border p-2 bg-light")
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Record Dividend", id="record-dividend-button", color="success", className="mt-4")
                            ], width=3)
                        ], className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Switch(
                                    id="dividend-drip-switch",
                                    label="Dividend Reinvestment Plan (DRIP)",
                                    value=False,
                                    className="mt-3"
                                )
                            ], width=5),
                        ]),
                        # DRIP Details (only visible when DRIP is selected)
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("DRIP Shares Acquired"),
                                dbc.Input(id="dividend-drip-shares-input", type="number", placeholder="Shares purchased", step="0.001")
                            ], width=4),
                            dbc.Col([
                                dbc.Label("DRIP Share Price"),
                                dbc.Input(id="dividend-drip-price-input", type="number", placeholder="Price per share")
                            ], width=4),
                            dbc.Col([
                                dbc.Label("DRIP Total (Calculated)"),
                                html.Div(id="dividend-drip-total-display", className="border p-2 bg-light")
                            ], width=4)
                        ], id="dividend-drip-row", style={"display": "none"}, className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Notes (Optional)"),
                                dbc.Textarea(id="dividend-notes-input", placeholder="Optional notes about this dividend")
                            ], width=12)
                        ], className="mt-3"),
                        html.Div(id="dividend-feedback", className="mt-3")
                    ])
                ], label="Record Dividend"),
                
                dbc.Tab([
                    # Dividend History
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Filter by Symbol"),
                                dbc.Input(id="dividend-filter-symbol-input", placeholder="Enter symbol to filter")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Date Range"),
                                dbc.RadioItems(
                                    id="dividend-date-range-select",
                                    options=[
                                        {"label": "Year to Date", "value": "ytd"},
                                        {"label": "Last 12 Months", "value": "1y"},
                                        {"label": "All Time", "value": "all"}
                                    ],
                                    value="1y",
                                    inline=True
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Button("Apply Filter", id="dividend-filter-button", color="primary", className="mt-4")
                            ], width=3)
                        ]),
                        html.Div(id="dividend-history-table", className="mt-3")
                    ])
                ], label="Dividend History"),
                
                dbc.Tab([
                    # Dividend Analysis
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Dividend Income Summary"),
                                    dbc.CardBody(id="dividend-summary-content")
                                ])
                            ], width=5),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Portfolio Yield"),
                                    dbc.CardBody(id="dividend-yield-content")
                                ])
                            ], width=7)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Dividend Income by Symbol", className="mt-4"),
                                dcc.Graph(id="dividend-by-symbol-chart")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Top Dividend Yields", className="mt-2"),
                                html.Div(id="top-dividend-yields-table", className="mt-2")
                            ], width=12)
                        ])
                    ])
                ], label="Dividend Analysis")
            ], id="dividend-tabs")
        ])
    ])

def create_dividend_history_table(dividends):
    """
    Create a table showing dividend history
    
    Args:
        dividends (list): List of dividend records
        
    Returns:
        Component: Dash component with dividend history table
    """
    if not dividends:
        return html.Div("No dividend records found for the selected criteria.")
    
    # Create the table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("Symbol"),
                html.Th("Amount Per Share"),
                html.Th("Shares"),
                html.Th("Total Amount"),
                html.Th("Currency"),
                html.Th("DRIP"),
                html.Th("Notes")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(div["dividend_date"]),
                html.Td(div["symbol"]),
                html.Td(f"${div['amount_per_share']:.4f}"),
                html.Td(f"{div['shares_held']:.2f}"),
                html.Td(f"${div['total_amount']:.2f}"),
                html.Td(div["currency"]),
                html.Td("Yes" if div["is_drip"] else "No"),
                html.Td(div.get("notes", ""))
            ]) for div in dividends
        ])
    ], bordered=True, striped=True, hover=True)

def create_dividend_summary(dividends, period="1y"):
    """
    Create a summary of dividend income for the given period
    
    Args:
        dividends (list): List of dividend records
        period (str): Period for summary ('ytd', '1y', 'all')
        
    Returns:
        Component: Dash component with dividend summary
    """
    # Sum dividend amounts by currency
    total_cad = sum(div["total_amount"] for div in dividends if div["currency"] == "CAD")
    total_usd = sum(div["total_amount"] for div in dividends if div["currency"] == "USD")
    
    # Get current exchange rate
    from modules.portfolio_utils import get_usd_to_cad_rate
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Convert USD to CAD for total income
    total_cad_equivalent = total_cad + (total_usd * usd_to_cad_rate)
    
    # Determine period text
    if period == "ytd":
        period_text = "Year to Date"
    elif period == "1y":
        period_text = "Last 12 Months"
    else:
        period_text = "All Time"
    
    # Calculate number of payments and symbols
    num_payments = len(dividends)
    symbols = set(div["symbol"] for div in dividends)
    num_symbols = len(symbols)
    
    return html.Div([
        html.H4(f"${total_cad_equivalent:.2f} CAD", className="text-success"),
        html.P(f"Total dividend income ({period_text})"),
        html.Hr(),
        html.P([
            html.Span("CAD Income: ", className="fw-bold"),
            f"${total_cad:.2f}"
        ]),
        html.P([
            html.Span("USD Income: ", className="fw-bold"),
            f"${total_usd:.2f} (${total_usd * usd_to_cad_rate:.2f} CAD)"
        ]),
        html.P([
            html.Span("Payments: ", className="fw-bold"),
            f"{num_payments} dividends from {num_symbols} securities"
        ])
    ])

def create_dividend_yield_display(yield_data):
    """
    Create a display of dividend yield information
    
    Args:
        yield_data (dict): Dividend yield data from get_dividend_yield
        
    Returns:
        Component: Dash component with dividend yield information
    """
    # Extract key metrics
    annualized_yield = yield_data.get("annualized_yield", 0)
    total_dividends = yield_data.get("total_dividends_combined_cad", 0)
    portfolio_value = yield_data.get("portfolio_value_combined_cad", 0)
    num_symbols = len(yield_data.get("symbols", []))
    period = yield_data.get("period", "1y")
    
    # Format period text
    if period == "1m":
        period_text = "Last Month"
    elif period == "3m":
        period_text = "Last 3 Months"
    elif period == "6m":
        period_text = "Last 6 Months"
    elif period == "1y":
        period_text = "Last 12 Months"
    else:
        period_text = "All Time"
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3(f"{annualized_yield:.2f}%", className="text-primary"),
                html.P("Annualized Dividend Yield", className="text-muted")
            ], width=6),
            dbc.Col([
                dbc.Progress(value=annualized_yield, max=10, 
                            color="success" if annualized_yield >= 4 else "warning" if annualized_yield >= 2 else "danger", 
                            style={"height": "30px"})
            ], width=6)
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Span("Period: ", className="fw-bold"),
                    period_text
                ])
            ], width=6),
            dbc.Col([
                html.P([
                    html.Span("Assets with Dividends: ", className="fw-bold"),
                    f"{num_symbols}"
                ])
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Span("Dividend Income: ", className="fw-bold"),
                    f"${total_dividends:.2f} CAD"
                ])
            ], width=6),
            dbc.Col([
                html.P([
                    html.Span("Portfolio Value: ", className="fw-bold"),
                    f"${portfolio_value:.2f} CAD"
                ])
            ], width=6)
        ]),
        html.Div([
            html.P("Yield Interpretation:", className="fw-bold mt-3"),
            html.Ul([
                html.Li("0-2%: Low yield, focus on growth", className="text-danger"),
                html.Li("2-4%: Moderate yield, balanced approach", className="text-warning"),
                html.Li("4%+: High yield, income-focused", className="text-success")
            ])
        ])
    ])

def create_top_dividend_yields_table(symbols_summary):
    """
    Create a table showing top dividend yields across portfolio holdings
    
    Args:
        symbols_summary (dict): Dividend summary by symbol
        
    Returns:
        Component: Dash component with top dividend yields table
    """
    if not symbols_summary:
        return html.Div("No dividend data available for yield analysis.")
    
    # Create a list for sorting
    symbol_list = []
    for symbol, data in symbols_summary.items():
        # Only include symbols with valid yield data
        if data.get("current_yield", 0) > 0:
            symbol_list.append({
                "symbol": symbol,
                "yield": data.get("current_yield", 0),
                "amount_per_share": data.get("latest_amount_per_share", 0),
                "estimated_annual": data.get("estimated_annual_dividend", 0),
                "price": data.get("current_price", 0),
                "total_annual_income": data.get("estimated_annual_dividend", 0) * data.get("current_shares", 0)
            })
    
    # Sort by yield (descending)
    symbol_list.sort(key=lambda x: x["yield"], reverse=True)
    
    # Create the table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbol"),
                html.Th("Current Yield"),
                html.Th("Last Dividend"),
                html.Th("Est. Annual"),
                html.Th("Current Price"),
                html.Th("Annual Income")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(item["symbol"]),
                html.Td(f"{item['yield']:.2f}%", 
                        style={"color": "green" if item['yield'] >= 4 else 
                               "orange" if item['yield'] >= 2 else "red"}),
                html.Td(f"${item['amount_per_share']:.4f}"),
                html.Td(f"${item['estimated_annual']:.4f}"),
                html.Td(f"${item['price']:.2f}"),
                html.Td(f"${item['total_annual_income']:.2f}")
            ]) for item in symbol_list
        ])
    ], bordered=True, striped=True, hover=True, size="sm")
