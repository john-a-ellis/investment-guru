# Add to components/portfolio_summary.py

from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd

def create_portfolio_summary_component(portfolio=None, include_cards=True):
    """
    Create a comprehensive portfolio summary component with cash flow information
    and FX impact details for Canadian investors
    
    Args:
        portfolio (dict): Portfolio data (if None, will be loaded)
        include_cards (bool): Whether to include summary cards
        
    Returns:
        Component: Dash component with portfolio summary
    """
    from modules.portfolio_utils import load_portfolio, update_portfolio_data, get_cash_positions, load_cash_flows
    
    # Load portfolio if not provided
    if portfolio is None:
        portfolio = update_portfolio_data()
    
    # Get cash positions
    cash_positions = get_cash_positions()
    
    # Get recent cash flows (last 5)
    recent_flows = load_cash_flows()[:5]
    
    # Calculate portfolio metrics
    total_value_cad = 0
    total_value_usd = 0
    total_invested_cad = 0
    total_invested_usd = 0
    
    # Group investments by symbol for consolidation
    symbols = {}
    
    for inv_id, details in portfolio.items():
        currency = details.get("currency", "USD")
        current_value = float(details.get("current_value", 0))
        purchase_value = float(details.get("shares", 0)) * float(details.get("purchase_price", 0))
        symbol = details.get("symbol", "")
        
        # Update totals by currency
        if currency == "CAD":
            total_value_cad += current_value
            total_invested_cad += purchase_value
        else:  # Assume USD
            total_value_usd += current_value
            total_invested_usd += purchase_value
            
        # Group by symbol
        if symbol not in symbols:
            symbols[symbol] = {
                "current_value": 0,
                "invested_value": 0,
                "currency": currency,
                "asset_type": details.get("asset_type", "stock"),
                "shares": 0
            }
        
        symbols[symbol]["current_value"] += current_value
        symbols[symbol]["invested_value"] += purchase_value
        symbols[symbol]["shares"] += float(details.get("shares", 0))
    
    # Calculate cash totals
    cash_cad = float(cash_positions.get("CAD", {}).get("balance", 0))
    cash_usd = float(cash_positions.get("USD", {}).get("balance", 0))
    
    # Get current USD/CAD exchange rate
    from modules.portfolio_utils import get_usd_to_cad_rate, get_weighted_exchange_rate
    current_rate = get_usd_to_cad_rate()
    
    # Get weighted historical exchange rate for USD investments
    historical_rate = get_weighted_exchange_rate("CAD", "USD")
    if historical_rate > 0:
        historical_rate = 1 / historical_rate  # Convert to USD/CAD
    else:
        historical_rate = current_rate  # Fallback
    
    # Calculate total in CAD using current exchange rate
    usd_value_in_cad = total_value_usd * current_rate
    usd_invested_in_cad_current = total_invested_usd * current_rate
    usd_invested_in_cad_historical = total_invested_usd * historical_rate
    cash_usd_in_cad = cash_usd * current_rate
    
    # Calculate FX impact on USD investments
    fx_impact = usd_invested_in_cad_current - usd_invested_in_cad_historical
    fx_impact_pct = (fx_impact / usd_invested_in_cad_historical) * 100 if usd_invested_in_cad_historical > 0 else 0
    
    # Calculate grand totals in CAD
    grand_total_cad = total_value_cad + usd_value_in_cad + cash_cad + cash_usd_in_cad
    total_invested_cad_equiv = total_invested_cad + usd_invested_in_cad_historical
    
    # Calculate gain/loss (excluding cash but including FX impact)
    investment_value_cad = total_value_cad + usd_value_in_cad
    total_gain_loss_cad = investment_value_cad - total_invested_cad_equiv
    gain_loss_pct = (total_gain_loss_cad / total_invested_cad_equiv * 100) if total_invested_cad_equiv > 0 else 0
    
    # Calculate pure investment gain/loss (excluding FX impact)
    pure_gain_loss_cad = total_value_cad - total_invested_cad
    pure_gain_loss_usd = total_value_usd - total_invested_usd
    pure_gain_loss_usd_in_cad = pure_gain_loss_usd * current_rate
    total_pure_gain_loss = pure_gain_loss_cad + pure_gain_loss_usd_in_cad
    
    # Create asset type breakdown
    asset_types = {}
    for symbol_data in symbols.values():
        asset_type = symbol_data["asset_type"]
        if asset_type not in asset_types:
            asset_types[asset_type] = 0
        asset_types[asset_type] += symbol_data["current_value"]
        
        # Convert USD to CAD
        if symbol_data["currency"] == "USD":
            asset_types[asset_type] = asset_types[asset_type] * current_rate
            
    # Add cash positions
    if cash_cad > 0 or cash_usd > 0:
        if "cash" not in asset_types:
            asset_types["cash"] = 0
        asset_types["cash"] += cash_cad + cash_usd_in_cad
    
    # Calculate percentages
    asset_type_percentages = {}
    for asset_type, value in asset_types.items():
        asset_type_percentages[asset_type] = (value / grand_total_cad * 100) if grand_total_cad > 0 else 0
    
    # Format for display
    asset_type_rows = []
    for asset_type, percentage in sorted(asset_type_percentages.items(), key=lambda x: x[1], reverse=True):
        value = asset_types[asset_type]
        asset_type_rows.append(
            html.Tr([
                html.Td(asset_type.capitalize()),
                html.Td(f"${value:.2f}"),
                html.Td(f"{percentage:.1f}%")
            ])
        )
    
    # Create summary cards
    summary_cards = []
    
    if include_cards:
        summary_cards = [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Portfolio Value", className="card-title"),
                            html.H2(f"${grand_total_cad:.2f} CAD", className="text-primary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Invested", className="card-title"),
                            html.H2(f"${total_invested_cad_equiv:.2f} CAD", className="text-secondary")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Gain/Loss", className="card-title"),
                            html.H2(
                                f"${total_gain_loss_cad:.2f} ({gain_loss_pct:.2f}%)", 
                                className="text-success" if total_gain_loss_cad >= 0 else "text-danger"
                            )
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Cash Positions", className="card-title"),
                            html.Div([
                                html.Div(f"CAD: ${cash_cad:.2f}", className="d-flex justify-content-between"),
                                html.Div(f"USD: ${cash_usd:.2f} (${cash_usd_in_cad:.2f} CAD)", className="d-flex justify-content-between"),
                                html.Hr(className="my-1"),
                                html.Div(f"Total Cash: ${(cash_cad + cash_usd_in_cad):.2f} CAD", 
                                         className="fw-bold d-flex justify-content-between")
                            ])
                        ])
                    ])
                ], width=3)
            ], className="mb-4")
        ]
    
    # Create FX impact card if there are USD investments
    fx_impact_card = None
    if total_value_usd > 0:
        fx_impact_card = dbc.Card([
            dbc.CardHeader("Foreign Exchange Impact"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("USD Holdings"),
                        html.P(f"${total_value_usd:.2f} USD (${usd_value_in_cad:.2f} CAD)")
                    ], width=4),
                    dbc.Col([
                        html.H6("Exchange Rates"),
                        html.P([
                            f"Historical Avg: {historical_rate:.4f}, Current: {current_rate:.4f}",
                            html.Br(),
                            html.Span(
                                f"Change: {((current_rate / historical_rate) - 1) * 100:.2f}%",
                                style={"color": "green" if current_rate > historical_rate else "red"}
                            )
                        ])
                    ], width=4),
                    dbc.Col([
                        html.H6("FX Impact"),
                        html.P([
                            html.Span(
                                f"${fx_impact:.2f} CAD ({fx_impact_pct:.2f}%)",
                                style={"color": "green" if fx_impact >= 0 else "red"}
                            ),
                            html.Br(),
                            "Impact of exchange rate changes on USD investments"
                        ])
                    ], width=4)
                ])
            ])
        ], className="mb-3")
    
    # Create asset allocation table
    asset_allocation_table = dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Asset Type"),
                html.Th("Value (CAD)"),
                html.Th("Allocation %")
            ])
        ),
        html.Tbody(asset_type_rows)
    ], bordered=True, striped=True, size="sm")
    
    # Create recent cash flows table
    recent_flows_table = None
    if recent_flows:
        recent_flows_table = dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Date"),
                    html.Th("Type"),
                    html.Th("Amount"),
                    html.Th("Currency"),
                    html.Th("Description")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(flow['date']),
                    html.Td(flow['flow_type'].capitalize(), 
                           style={"color": "green" if flow['flow_type'] == "deposit" else "red"}),
                    html.Td(f"${flow['amount']:.2f}"),
            html.Td(flow['currency']),
                                html.Td(flow.get('description', ''))
                            ]) for flow in recent_flows[:5]  # Show up to 5 most recent
                        ])
                    ], bordered=True, hover=True, size="sm", className="mt-3")
                
                # Create final component
        return html.Div([
            *summary_cards,
            fx_impact_card if fx_impact_card else None,
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Asset Allocation"),
                        dbc.CardBody([
                            asset_allocation_table
                        ])
                    ])
                ], width=7),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Cash Flows"),
                        dbc.CardBody([
                            recent_flows_table if recent_flows_table else html.P("No recent cash flows recorded.")
                        ])
                    ])
                ], width=5)
            ]),
            html.Div(
                f"Exchange Rate: 1 USD = {current_rate:.4f} CAD | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                className="text-muted mt-2 small"
            )
        ], id="portfolio-summary")
