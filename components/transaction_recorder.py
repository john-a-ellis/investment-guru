# components/transaction_recorder.py

import dash
import pandas as pd
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from modules.transaction_tracker import record_transaction, load_transactions

def create_transaction_recorder_component():
    """Creates a component for recording buy/sell transactions"""
    return dbc.Card([
        dbc.CardHeader("Transaction Recorder"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Buy Transaction Form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbol"),
                                dbc.Input(id="buy-symbol-input", placeholder="e.g., MFC.TO")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shares"),
                                dbc.Input(id="buy-shares-input", type="number", placeholder="Number of shares")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Price"),
                                dbc.Input(id="buy-price-input", type="number", placeholder="Purchase price")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="buy-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Record Purchase", id="record-buy-button", color="success", className="mt-4")
                            ], width=2)
                        ]),
                        html.Div(id="buy-transaction-feedback", className="mt-2")
                    ])
                ], label="Buy"),
                
                dbc.Tab([
                    # Sell Transaction Form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Symbol"),
                                dbc.Input(id="sell-symbol-input", placeholder="e.g., MFC.TO")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Shares"),
                                dbc.Input(id="sell-shares-input", type="number", placeholder="Number of shares")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Price"),
                                dbc.Input(id="sell-price-input", type="number", placeholder="Selling price")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="sell-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Record Sale", id="record-sell-button", color="danger", className="mt-4")
                            ], width=2)
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
                html.Td(f"{row['shares']:.2f}"),
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