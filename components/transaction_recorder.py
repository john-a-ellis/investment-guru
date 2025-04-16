# components/transaction_recorder.py

import dash
import pandas as pd
from dash import html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from modules.transaction_tracker import record_transaction, load_transactions

def create_transaction_recorder_component():
    """Creates a component for recording buy/sell transactions with currency conversion support"""
    return dbc.Card([
        dbc.CardHeader("Transaction Recorder"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Buy Transaction Form with currency conversion support
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbol"),
                                dbc.Input(id="buy-symbol-input", placeholder="e.g., MFC.TO or AAPL")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shares"),
                                dbc.Input(id="buy-shares-input", type="number", placeholder="Number of shares", step="0.001")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Price"),
                                dbc.Input(id="buy-price-input", type="number", placeholder="Purchase price")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="buy-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Record Purchase", id="record-buy-button", color="success", className="mt-4")
                            ], width=3)
                        ]),
                        # Currency conversion options for US stocks
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id="buy-currency-convert-checkbox",
                                    label="Purchase in CAD with currency conversion",
                                    value=False,
                                    className="mt-2"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("CAD Amount"),
                                    dbc.Input(id="buy-cad-amount-input", type="number", placeholder="CAD amount", disabled=True)
                                ], className="mt-2")
                            ], width=3),
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Rate"),
                                    dbc.Input(id="buy-exchange-rate-input", type="number", placeholder="CAD per USD", 
                                              disabled=True, step="0.0001", value=1.35)
                                ], className="mt-2")
                            ], width=3)
                        ], id="buy-currency-conversion-row", style={"display": "none"}),
                        html.Div(id="buy-transaction-feedback", className="mt-2")
                    ])
                ], label="Buy"),
                
                dbc.Tab([
                    # Sell Transaction Form with currency conversion display
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbol"),
                                dbc.Input(id="sell-symbol-input", placeholder="e.g., MFC.TO or AAPL")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shares"),
                                dbc.Input(id="sell-shares-input", type="number", placeholder="Number of shares", step="0.001")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Price"),
                                dbc.Input(id="sell-price-input", type="number", placeholder="Selling price")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="sell-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=2),
                            dbc.Col([
                                dbc.Button("Record Sale", id="record-sell-button", color="danger", className="mt-4")
                            ], width=3)
                        ]),
                        # Currency conversion display for US stocks
                        dbc.Row([
                            dbc.Col([
                                dbc.Alert([
                                    html.P([
                                        "For USD investments: Proceeds will be converted from USD to CAD using the current exchange rate. ",
                                        "Adjust settings in Currency Exchange Manager if needed."
                                    ], className="mb-0")
                                ], color="info", className="mt-2")
                            ], width=12)
                        ]),
                        html.Div(id="sell-transaction-feedback", className="mt-2")
                    ])
                ], label="Sell"),
                
                dbc.Tab([
                    # Transaction History
                    html.Div(id="transaction-history-table")
                ], label="Transaction History")
            ],id="transaction-tabs")
        ])
    ])


def create_transaction_history_table():
    """Create a table to display transaction history"""
    transactions = load_transactions()
    
    if not transactions:
        return html.Div("No transactions recorded yet.")
    
    # Convert to DataFrame for easier table creation
    trans_list = []
    for trans_id, details in transactions.items():
        trans_list.append({
            "id": trans_id,
            "date": details.get("transaction_date", ""),
            "type": details.get("transaction_type", "").capitalize(),
            "symbol": details.get("symbol", ""),
            "shares": details.get("shares", 0),
            "price": details.get("price", 0),
            "total": details.get("total_amount", 0),
            "currency": details.get("currency", "USD")
        })
    
    # Sort by date, most recent first
    trans_df = pd.DataFrame(trans_list)
    trans_df = trans_df.sort_values("date", ascending=False)
    
    # Create table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("Type"),
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Price"),
                html.Th("Total Amount"),
                html.Th("Currency")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["date"]),
                html.Td(row["type"], style={"color": "green" if row["type"] == "Buy" else "red"}),
                html.Td(row["symbol"]),
                html.Td(f"{row['shares']:.3f}"),
                html.Td(f"${row['price']:.2f}"),
                html.Td(f"${row['total']:.2f}"),
                html.Td(row["currency"])
            ]) for _, row in trans_df.iterrows()
        ])
    ], striped=True, bordered=True, hover=True)

# Add to imports
from components.transaction_recorder import create_transaction_recorder_component, create_transaction_history_table
from modules.transaction_tracker import record_transaction

# Add the Transaction Recorder component to the layout before the Portfolio Management
dbc.Row([
    dbc.Col([
        create_transaction_recorder_component()
    ], width=12)
], className="mb-4"),

# Add callbacks for buy transactions
@callback(
    [Output("buy-transaction-feedback", "children"),
     Output("transaction-history-table", "children", allow_duplicate=True),
     Output("buy-symbol-input", "value"),
     Output("buy-shares-input", "value"),
     Output("buy-price-input", "value")],
    Input("record-buy-button", "n_clicks"),
    [State("buy-symbol-input", "value"),
     State("buy-shares-input", "value"),
     State("buy-price-input", "value"),
     State("buy-date-input", "value")],
    prevent_initial_call=True
)
def record_buy_transaction(n_clicks, symbol, shares, price, date):
    if n_clicks is None or not symbol or not shares or not price:
        raise dash.exceptions.PreventUpdate
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    success = record_transaction(symbol, "buy", shares, price, date)
    
    if success:
        return (
            dbc.Alert(f"Successfully recorded purchase of {shares} shares of {symbol}", color="success"),
            create_transaction_history_table(),
            "", None, None
        )
    else:
        return (
            dbc.Alert("Failed to record transaction", color="danger"),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )

# Add callbacks for sell transactions
@callback(
    [Output("sell-transaction-feedback", "children"),
     Output("transaction-history-table", "children", allow_duplicate=True),
     Output("sell-symbol-input", "value"),
     Output("sell-shares-input", "value"),
     Output("sell-price-input", "value")],
    Input("record-sell-button", "n_clicks"),
    [State("sell-symbol-input", "value"),
     State("sell-shares-input", "value"),
     State("sell-price-input", "value"),
     State("sell-date-input", "value")],
    prevent_initial_call=True
)
def record_sell_transaction(n_clicks, symbol, shares, price, date):
    if n_clicks is None or not symbol or not shares or not price:
        raise dash.exceptions.PreventUpdate
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    success = record_transaction(symbol, "sell", shares, price, date)
    
    if success:
        return (
            dbc.Alert(f"Successfully recorded sale of {shares} shares of {symbol}", color="success"),
            create_transaction_history_table(),
            "", None, None
        )
    else:
        return (
            dbc.Alert("Failed to record transaction. Check if you have enough shares to sell.", color="danger"),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )

# Display transaction history on tab selection
@callback(
    Output("transaction-history-table", "children"),
    Input("transaction-tabs", "active_tab"),
    prevent_initial_call=False
)
def load_transaction_history(active_tab):
    return create_transaction_history_table()

@callback(
    [Output("buy-currency-conversion-row", "style"),
     Output("buy-currency-convert-checkbox", "disabled"),
     Output("buy-cad-amount-input", "disabled"),
     Output("buy-exchange-rate-input", "disabled")],
    [Input("buy-symbol-input", "value"),
     Input("buy-currency-convert-checkbox", "value")]
)
def toggle_currency_conversion(symbol, convert_checked):
    """
    Show/hide and enable/disable currency conversion options based on symbol and checkbox
    """
    if not symbol:
        return {"display": "none"}, True, True, True
    
    # Check if this is likely a US stock (no .TO suffix)
    is_us_stock = (symbol and not symbol.endswith((".TO", ".V")) and not symbol.startswith("MAW"))
    
    # Show row for US stocks, but keep fields disabled unless checkbox is checked
    if is_us_stock:
        return {"display": "flex"}, False, not convert_checked, not convert_checked
    else:
        return {"display": "none"}, True, True, True

@callback(
    Output("buy-cad-amount-input", "value"),
    [Input("buy-price-input", "value"),
     Input("buy-shares-input", "value"),
     Input("buy-exchange-rate-input", "value"),
     Input("buy-currency-convert-checkbox", "value")]
)
def calculate_cad_amount(price, shares, exchange_rate, convert_enabled):
    """
    Calculate the CAD amount based on USD price and exchange rate
    """
    if not convert_enabled or price is None or shares is None or exchange_rate is None:
        return None
    
    usd_amount = price * shares
    cad_amount = usd_amount * exchange_rate
    
    return cad_amount

# Update the buy transaction callback to handle currency conversion
@callback(
    [Output("buy-transaction-feedback", "children"),
     Output("transaction-history-table", "children", allow_duplicate=True),
     Output("buy-symbol-input", "value"),
     Output("buy-shares-input", "value"),
     Output("buy-price-input", "value")],
    Input("record-buy-button", "n_clicks"),
    [State("buy-symbol-input", "value"),
     State("buy-shares-input", "value"),
     State("buy-price-input", "value"),
     State("buy-date-input", "value"),
     State("buy-currency-convert-checkbox", "value"),
     State("buy-cad-amount-input", "value"),
     State("buy-exchange-rate-input", "value")],
    prevent_initial_call=True
)
def record_buy_transaction_with_fx(n_clicks, symbol, shares, price, date, convert_currency, cad_amount, exchange_rate):
    """Enhanced buy transaction recording with currency conversion support"""
    if n_clicks is None or not symbol or not shares or not price:
        raise dash.exceptions.PreventUpdate
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    # Check if we need to handle currency conversion
    is_us_stock = not symbol.endswith((".TO", ".V")) and not symbol.startswith("MAW")
    
    if is_us_stock and convert_currency and cad_amount and exchange_rate:
        # This is a US stock purchased with CAD
        from modules.portfolio_utils import record_currency_exchange
        
        # First, convert CAD to USD
        usd_amount = price * shares
        
        # Record the currency exchange
        exchange_success = record_currency_exchange(
            "CAD", cad_amount, "USD", usd_amount, date, 
            f"Currency exchange for purchase of {shares} shares of {symbol}"
        )
        
        if not exchange_success:
            return (
                dbc.Alert("Failed to record currency exchange", color="danger"),
                dash.no_update, dash.no_update, dash.no_update, dash.no_update
            )
            
        # Then record the actual transaction
        transaction_success = record_transaction(symbol, "buy", shares, price, date)
        
        if transaction_success:
            return (
                dbc.Alert([
                    f"Successfully recorded purchase of {shares} shares of {symbol}",
                    html.Br(),
                    f"Converted ${cad_amount:.2f} CAD to ${usd_amount:.2f} USD at rate {exchange_rate:.4f}"
                ], color="success"),
                create_transaction_history_table(),
                "", None, None
            )
        else:
            return (
                dbc.Alert(f"Transaction recorded but currency conversion failed", color="warning"),
                create_transaction_history_table(),
                "", None, None
            )
    else:
        # Standard transaction without currency conversion
        success = record_transaction(symbol, "buy", shares, price, date)
        
        if success:
            return (
                dbc.Alert(f"Successfully recorded purchase of {shares} shares of {symbol}", color="success"),
                create_transaction_history_table(),
                "", None, None
            )
        else:
            return (
                dbc.Alert("Failed to record transaction", color="danger"),
                dash.no_update, dash.no_update, dash.no_update, dash.no_update
            )

# Update the sell transaction callback to handle currency conversion for proceeds
@callback(
    [Output("sell-transaction-feedback", "children"),
     Output("transaction-history-table", "children", allow_duplicate=True),
     Output("sell-symbol-input", "value"),
     Output("sell-shares-input", "value"),
     Output("sell-price-input", "value")],
    Input("record-sell-button", "n_clicks"),
    [State("sell-symbol-input", "value"),
     State("sell-shares-input", "value"),
     State("sell-price-input", "value"),
     State("sell-date-input", "value")],
    prevent_initial_call=True
)
def record_sell_transaction_with_fx(n_clicks, symbol, shares, price, date):
    """Enhanced sell transaction recording with automatic currency conversion for US stocks"""
    if n_clicks is None or not symbol or not shares or not price:
        raise dash.exceptions.PreventUpdate
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    # Check if this is a US stock
    is_us_stock = not symbol.endswith((".TO", ".V")) and not symbol.startswith("MAW")
    
    # First record the sell transaction
    success = record_transaction(symbol, "sell", shares, price, date)
    
    if success:
        # For US stocks, automatically convert USD proceeds to CAD
        if is_us_stock:
            from modules.portfolio_utils import record_currency_exchange, get_usd_to_cad_rate
            
            # Calculate USD proceeds
            usd_amount = price * shares
            
            # Get current exchange rate
            exchange_rate = get_usd_to_cad_rate()
            
            # Calculate CAD equivalent
            cad_amount = usd_amount * exchange_rate
            
            # Record the currency exchange
            exchange_success = record_currency_exchange(
                "USD", usd_amount, "CAD", cad_amount, date, 
                f"Currency exchange for proceeds from sale of {shares} shares of {symbol}"
            )
            
            if exchange_success:
                return (
                    dbc.Alert([
                        f"Successfully recorded sale of {shares} shares of {symbol}",
                        html.Br(),
                        f"Converted ${usd_amount:.2f} USD to ${cad_amount:.2f} CAD at rate {exchange_rate:.4f}"
                    ], color="success"),
                    create_transaction_history_table(),
                    "", None, None
                )
            else:
                return (
                    dbc.Alert([
                        f"Sale recorded successfully but currency conversion failed",
                        html.Br(),
                        f"Please manually convert ${usd_amount:.2f} USD to CAD in Currency Exchange Manager"
                    ], color="warning"),
                    create_transaction_history_table(),
                    "", None, None
                )
        else:
            # Standard response for Canadian stocks
            return (
                dbc.Alert(f"Successfully recorded sale of {shares} shares of {symbol}", color="success"),
                create_transaction_history_table(),
                "", None, None
            )
    else:
        return (
            dbc.Alert("Failed to record transaction. Check if you have enough shares to sell.", color="danger"),
            dash.no_update, dash.no_update, dash.no_update, dash.no_update
        )

# Add callback to refresh exchange rate when symbol changes
@callback(
    Output("buy-exchange-rate-input", "value"),
    Input("buy-symbol-input", "value"),
    prevent_initial_call=True
)
def refresh_exchange_rate(symbol):
    """
    Refresh the exchange rate when a US symbol is entered
    """
    if not symbol:
        return 1.35  # Default value
    
    # Check if this is a US stock
    is_us_stock = not symbol.upper().endswith((".TO", ".V")) and not symbol.upper().startswith("MAW")
    
    if is_us_stock:
        # Get current exchange rate
        from modules.portfolio_utils import get_usd_to_cad_rate
        rate = get_usd_to_cad_rate()
        return rate
    
    return 1.35  # Default value

# Keep the existing function for creating transaction history table
def create_transaction_history_table():
    """Create a table to display transaction history"""
    transactions = load_transactions()
    
    if not transactions:
        return html.Div("No transactions recorded yet.")
    
    # Convert to DataFrame for easier table creation
    trans_list = []
    for trans_id, details in transactions.items():
        trans_list.append({
            "id": trans_id,
            "date": details.get("transaction_date", ""),
            "type": details.get("transaction_type", "").capitalize(),
            "symbol": details.get("symbol", ""),
            "shares": details.get("shares", 0),
            "price": details.get("price", 0),
            "total": details.get("total_amount", 0),
            "currency": details.get("currency", "USD")
        })
    
    # Sort by date, most recent first
    trans_df = pd.DataFrame(trans_list)
    trans_df = trans_df.sort_values("date", ascending=False)
    
    # Create table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("Type"),
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Price"),
                html.Th("Total Amount"),
                html.Th("Currency")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["date"]),
                html.Td(row["type"], style={"color": "green" if row["type"] == "Buy" else "red"}),
                html.Td(row["symbol"]),
                html.Td(f"{row['shares']:.3f}"),
                html.Td(f"${row['price']:.2f}"),
                html.Td(f"${row['total']:.2f}"),
                html.Td(row["currency"])
            ]) for _, row in trans_df.iterrows()
        ])
    ], striped=True, bordered=True, hover=True)