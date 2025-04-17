# components/currency_exchange_component.py
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd

def create_currency_exchange_component():
    """
    Creates a component for managing currency exchange for cross-border transactions.
    Handles both CAD-to-USD and USD-to-CAD exchanges.
    """
    return dbc.Card([
        dbc.CardHeader("Currency Exchange Manager"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Add direction selection at the top
                    html.H5("Currency Exchange", className="mt-2 mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Exchange Direction"),
                            dbc.Select(
                                id="exchange-direction-select",
                                options=[
                                    {"label": "CAD to USD", "value": "cad_to_usd"},
                                    {"label": "USD to CAD", "value": "usd_to_cad"}
                                ],
                                value="cad_to_usd"
                            )
                        ], width=4)
                    ], className="mb-3"),
                    
                    # Show/hide based on direction
                    html.Div(id="cad-to-usd-form", children=[
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("CAD Amount"),
                                    dbc.Input(id="cad-amount-input", type="number", placeholder="CAD Amount", step="0.01")
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Current Exchange Rate"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("1 USD ="),
                                        dbc.Input(id="exchange-rate-input", type="number", placeholder="CAD Rate", step="0.0001", 
                                                  value=1.35)
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Date"),
                                    dbc.Input(id="exchange-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Description"),
                                    dbc.Input(id="exchange-description-input", placeholder="Optional description")
                                ], width=3),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Resulting USD"),
                                    html.Div(id="resulting-usd-display", className="border p-2 bg-light")
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("Refresh Rate", id="refresh-exchange-rate-button", color="info", className="me-2 mt-3"),
                                    dbc.Button("Execute Exchange", id="execute-exchange-button", color="success", className="mt-3")
                                ], width=3),
                                dbc.Col(width=6)
                            ], className="mt-3")
                        ])
                    ]),
                    
                    # USD to CAD form (initially hidden)
                    html.Div(id="usd-to-cad-form", style={"display": "none"}, children=[
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("USD Amount"),
                                    dbc.Input(id="usd-amount-input", type="number", placeholder="USD Amount", step="0.01")
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Current Exchange Rate"),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("1 USD ="),
                                        dbc.Input(id="usd-to-cad-rate-input", type="number", placeholder="CAD Rate", step="0.0001", 
                                                  value=1.35)
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Date"),
                                    dbc.Input(id="usd-exchange-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Description"),
                                    dbc.Input(id="usd-exchange-description-input", placeholder="Optional description")
                                ], width=3),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Resulting CAD"),
                                    html.Div(id="resulting-cad-display", className="border p-2 bg-light")
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("Refresh Rate", id="refresh-usd-rate-button", color="info", className="me-2 mt-3"),
                                    dbc.Button("Execute Exchange", id="execute-usd-exchange-button", color="success", className="mt-3")
                                ], width=3),
                                dbc.Col(width=6)
                            ], className="mt-3")
                        ])
                    ]),
                    
                    html.Div(id="exchange-feedback", className="mt-3")
                ], label="Currency Exchange"),
                
                # Keep existing tabs
                dbc.Tab([
                    # Exchange History & Impact Analysis
                    html.Div(id="exchange-history-table")
                ], label="Exchange History & Impact", tab_id="exchange-history-tab"),
            ], id="exchange-tabs")
        ])
    ])

def create_exchange_history_table():
    """
    Create a table showing currency exchange history and rates
    """
    from modules.portfolio_utils import load_currency_exchanges
    
    exchanges = load_currency_exchanges()
    
    if not exchanges:
        return html.Div("No currency exchange transactions recorded yet.")
    
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("From Currency"),
                html.Th("From Amount"),
                html.Th("To Currency"),
                html.Th("To Amount"),
                html.Th("Exchange Rate"),
                html.Th("Description")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(exchange['date']),
                html.Td(exchange['from_currency']),
                html.Td(f"${exchange['from_amount']:.2f}"),
                html.Td(exchange['to_currency']),
                html.Td(f"${exchange['to_amount']:.2f}"),
                html.Td(f"{exchange['rate']:.4f}"),
                html.Td(exchange['description'])
            ]) for exchange in exchanges
        ])
    ], striped=True, bordered=True, hover=True)

def calculate_fx_impact(portfolio):
    """
    Calculate the FX impact on portfolio returns with USD investments
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        dict: FX impact analysis results
    """
    # Group investments by currency
    cad_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "CAD"}
    usd_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "USD"}
    
    # Get current exchange rate
    from modules.portfolio_utils import get_usd_to_cad_rate
    current_rate = get_usd_to_cad_rate()
    
    # Calculate total USD investment value in USD and CAD
    usd_value = sum(float(inv.get("current_value", 0)) for inv in usd_investments.values())
    usd_cost_basis = sum(float(inv.get("shares", 0)) * float(inv.get("purchase_price", 0)) for inv in usd_investments.values())
    
    # Get original exchange rates if available
    from modules.portfolio_utils import load_currency_exchanges
    exchanges = load_currency_exchanges()
    
    # Calculate weighted average historical exchange rate from actual exchanges
    if exchanges:
        total_cad_exchanged = sum(ex.get('from_amount', 0) for ex in exchanges if ex.get('from_currency') == 'CAD' and ex.get('to_currency') == 'USD')
        total_usd_received = sum(ex.get('to_amount', 0) for ex in exchanges if ex.get('from_currency') == 'CAD' and ex.get('to_currency') == 'USD')
        
        weighted_historical_rate = total_cad_exchanged / total_usd_received if total_usd_received > 0 else current_rate
    else:
        # Fall back to a historical average if no exchange data
        weighted_historical_rate = current_rate * 0.95  # Assume 5% change as example
    
    # Calculate values in CAD
    usd_cost_basis_in_cad_historical = usd_cost_basis * weighted_historical_rate
    usd_cost_basis_in_cad_current = usd_cost_basis * current_rate
    usd_value_in_cad = usd_value * current_rate
    
    # Calculate FX impact
    fx_gain_loss = usd_cost_basis_in_cad_current - usd_cost_basis_in_cad_historical
    fx_gain_loss_pct = (fx_gain_loss / usd_cost_basis_in_cad_historical) * 100 if usd_cost_basis_in_cad_historical > 0 else 0
    
    # Calculate performance with and without FX impact
    pure_investment_gain_usd = usd_value - usd_cost_basis
    pure_investment_gain_pct = (pure_investment_gain_usd / usd_cost_basis) * 100 if usd_cost_basis > 0 else 0
    
    total_gain_with_fx = usd_value_in_cad - usd_cost_basis_in_cad_historical
    total_gain_with_fx_pct = (total_gain_with_fx / usd_cost_basis_in_cad_historical) * 100 if usd_cost_basis_in_cad_historical > 0 else 0
    
    return {
        "usd_investments_count": len(usd_investments),
        "usd_value": usd_value,
        "usd_cost_basis": usd_cost_basis,
        "weighted_historical_rate": weighted_historical_rate,
        "current_rate": current_rate,
        "usd_cost_basis_in_cad_historical": usd_cost_basis_in_cad_historical,
        "usd_value_in_cad": usd_value_in_cad,
        "fx_gain_loss": fx_gain_loss,
        "fx_gain_loss_pct": fx_gain_loss_pct,
        "pure_investment_gain_usd": pure_investment_gain_usd,
        "pure_investment_gain_pct": pure_investment_gain_pct,
        "total_gain_with_fx": total_gain_with_fx,
        "total_gain_with_fx_pct": total_gain_with_fx_pct
    }

def create_fx_impact_summary(portfolio):
    """
    Create an FX impact summary component
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with FX impact summary
    """
    # Calculate FX impact
    fx_impact = calculate_fx_impact(portfolio)
    
    if fx_impact["usd_investments_count"] == 0:
        return dbc.Alert("No USD investments found in portfolio. FX impact analysis is not applicable.", color="info")
    
    # Create summary cards row
    summary_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("USD Holdings"),
                dbc.CardBody([
                    html.H3(f"${fx_impact['usd_value']:.2f} USD"),
                    html.P(f"(${fx_impact['usd_value_in_cad']:.2f} CAD at current rate)")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Exchange Rates"),
                dbc.CardBody([
                    html.Div([
                        html.Span("Historical Average: ", className="fw-bold"),
                        html.Span(f"{fx_impact['weighted_historical_rate']:.4f} CAD/USD")
                    ]),
                    html.Div([
                        html.Span("Current: ", className="fw-bold"),
                        html.Span(f"{fx_impact['current_rate']:.4f} CAD/USD")
                    ]),
                    html.Div([
                        html.Span("Change: ", className="fw-bold"),
                        html.Span(
                            f"{((fx_impact['current_rate'] / fx_impact['weighted_historical_rate']) - 1) * 100:.2f}%",
                            style={"color": "green" if fx_impact['current_rate'] > fx_impact['weighted_historical_rate'] else "red"}
                        )
                    ])
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FX Impact"),
                dbc.CardBody([
                    html.H3(
                        f"${fx_impact['fx_gain_loss']:.2f} CAD",
                        style={"color": "green" if fx_impact['fx_gain_loss'] >= 0 else "red"}
                    ),
                    html.P(f"({fx_impact['fx_gain_loss_pct']:.2f}% impact on USD positions)")
                ])
            ])
        ], width=4)
    ], className="mb-3")
    
    # Create performance comparison table
    performance_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Metric"),
                html.Th("Without FX Impact (USD)"),
                html.Th("With FX Impact (CAD)")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td("Investment Returns"),
                html.Td(
                    f"${fx_impact['pure_investment_gain_usd']:.2f} ({fx_impact['pure_investment_gain_pct']:.2f}%)",
                    style={"color": "green" if fx_impact['pure_investment_gain_usd'] >= 0 else "red"}
                ),
                html.Td(
                    f"${fx_impact['total_gain_with_fx']:.2f} ({fx_impact['total_gain_with_fx_pct']:.2f}%)",
                    style={"color": "green" if fx_impact['total_gain_with_fx'] >= 0 else "red"}
                )
            ]),
            html.Tr([
                html.Td("FX Contribution"),
                html.Td("-"),
                html.Td(
                    f"${fx_impact['fx_gain_loss']:.2f} ({fx_impact['fx_gain_loss_pct']:.2f}%)",
                    style={"color": "green" if fx_impact['fx_gain_loss'] >= 0 else "red"}
                )
            ]),
            html.Tr([
                html.Td("Cost Basis"),
                html.Td(f"${fx_impact['usd_cost_basis']:.2f} USD"),
                html.Td(f"${fx_impact['usd_cost_basis_in_cad_historical']:.2f} CAD")
            ]),
            html.Tr([
                html.Td("Current Value"),
                html.Td(f"${fx_impact['usd_value']:.2f} USD"),
                html.Td(f"${fx_impact['usd_value_in_cad']:.2f} CAD")
            ])
        ])
    ], bordered=True, hover=True)
    
    # Explanation card
    explanation_card = dbc.Card([
        dbc.CardHeader("Understanding FX Impact"),
        dbc.CardBody([
            html.P([
                "FX impact refers to the effect of currency exchange rate changes on the CAD value of your USD investments. ",
                "This analysis compares your returns with and without the effect of exchange rate fluctuations."
            ]),
            html.P([
                "If CAD weakens against USD (USD/CAD rate increases), your USD investments are worth more in CAD terms. ",
                "If CAD strengthens (USD/CAD rate decreases), your USD investments are worth less in CAD."
            ]),
            html.P([
                "Historical rate: The weighted average exchange rate at which you converted CAD to USD for purchases. ",
                "Current rate: Today's exchange rate used to convert current USD values to CAD."
            ])
        ])
    ], className="mt-3")
    
    return html.Div([
        summary_cards,
        performance_table,
        explanation_card,
        html.Div(
            f"Analysis as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            className="text-muted mt-2 small"
        )
    ])