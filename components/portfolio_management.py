# components/portfolio_management.py
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import json
import os
from datetime import datetime
import uuid

def create_portfolio_management_component():
    """
    Creates a component for tracking actual investments with performance data
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Management"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Add New Investment"),
                    dbc.InputGroup([
                        dbc.Input(id="investment-symbol-input", placeholder="Symbol (e.g., MFC.TO, MAW104)"),
                        dbc.Input(id="investment-shares-input", type="number", placeholder="Number of Shares/Units"),
                        dbc.Input(id="investment-price-input", type="number", placeholder="Purchase Price"),
                        dbc.Input(id="investment-date-input", type="date", 
                                 value=datetime.now().strftime("%Y-%m-%d")),
                        dbc.Select(
                            id="investment-type-select",  # This ID must match what's used in the callback
                            options=[
                                {"label": "Stock", "value": "stock"},
                                {"label": "ETF", "value": "etf"},
                                {"label": "Mutual Fund", "value": "mutual_fund"},
                                {"label": "Cryptocurrency", "value": "crypto"},
                                {"label": "Bond", "value": "bond"},
                                {"label": "Cash", "value": "cash"}
                            ],
                            value="stock",
                            placeholder="Asset Type"
                        ),
                        dbc.Button("Add Investment", id="add-investment-button", color="success")
                    ]),
                    html.Div(id="add-investment-feedback", className="mt-2")
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Current Portfolio"),
                    html.Div(id="portfolio-table")
                ], width=12)
            ])
        ])
    ])

def load_portfolio():
    """
    Load portfolio data from storage file
    """
    try:
        if os.path.exists('data/portfolio.json'):
            with open('data/portfolio.json', 'r') as f:
                return json.load(f)
        else:
            # Default empty portfolio if no file exists
            return {}
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return {}

def save_portfolio(portfolio):
    """
    Save portfolio data to storage file
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False

# Update in components/portfolio_management.py

# In components/portfolio_management.py

def load_transactions():
    """
    Load transaction records from storage file
    
    Returns:
        dict: Transaction records
    """
    try:
        if os.path.exists('data/transactions.json'):
            with open('data/transactions.json', 'r') as f:
                return json.load(f)
        else:
            # Default empty transactions if no file exists
            return {}
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return {}

# Update the create_portfolio_table function in portfolio_management.py
# Replace the existing function with this one

def create_portfolio_table(portfolio):
    """
    Create a table to display current portfolio investments with accordion components
    grouped by asset symbol, integrated with transaction history
    """
    print(f"Creating portfolio table with {len(portfolio)} investments")
    
    # Load transactions to enhance the portfolio view
    transactions = load_transactions()
    print(f"Loaded {len(transactions)} transactions")
    
    if not portfolio:
        return html.Div("No investments currently tracked.")
    
    # Convert to DataFrame for easier processing
    investments_list = []
    for investment_id, details in portfolio.items():
        print(f"Processing investment {investment_id}: {details}")
        
        currency = details.get("currency", "USD")
        
        investments_list.append({
            "id": investment_id,
            "symbol": details.get("symbol", ""),
            "shares": details.get("shares", 0),
            "purchase_price": details.get("purchase_price", 0),
            "purchase_date": details.get("purchase_date", ""),
            "current_price": details.get("current_price", 0),
            "current_value": details.get("current_value", 0),
            "gain_loss": details.get("gain_loss", 0),
            "gain_loss_percent": details.get("gain_loss_percent", 0),
            "currency": currency,
            "asset_type": details.get("asset_type", "stock")
        })
    
    df = pd.DataFrame(investments_list)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Group investments by symbol
    grouped_investments = {}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        if symbol not in grouped_investments:
            grouped_investments[symbol] = {
                "investments": [],
                "transactions": [],
                "total_shares": 0,
                "total_book_value": 0,
                "total_current_value": 0,
                "current_price": row["current_price"],  # Use the most recent price
                "currency": row["currency"],
                "asset_type": row["asset_type"]
            }
        
        # Add this investment to the group
        grouped_investments[symbol]["investments"].append(row)
        
        # Update group totals
        grouped_investments[symbol]["total_shares"] += row["shares"]
        grouped_investments[symbol]["total_book_value"] += row["shares"] * row["purchase_price"]
        grouped_investments[symbol]["total_current_value"] += row["current_value"]
    
    # Add transactions to each symbol group
    for transaction_id, transaction in transactions.items():
        symbol = transaction.get("symbol")
        if symbol in grouped_investments:
            grouped_investments[symbol]["transactions"].append({
                "id": transaction_id,
                **transaction  # Include all transaction details
            })
    
    # Calculate group gain/loss
    for symbol, group in grouped_investments.items():
        group["total_gain_loss"] = group["total_current_value"] - group["total_book_value"]
        if group["total_book_value"] > 0:
            group["total_gain_loss_percent"] = (group["total_gain_loss"] / group["total_book_value"]) * 100
        else:
            group["total_gain_loss_percent"] = 0
        
        # Sort transactions by date (newest first)
        if "transactions" in group:
            group["transactions"] = sorted(
                group["transactions"], 
                key=lambda x: x.get("transaction_date", ""), 
                reverse=True
            )
    
    # Create accordion items for each symbol group
    accordion_items = []
    
    # Sort groups by current value (descending)
    sorted_groups = sorted(grouped_investments.items(), key=lambda x: x[1]["total_current_value"], reverse=True)
    
    for symbol, group in sorted_groups:
        # Create the header with summary information - FIXED LAYOUT
        header = html.Div([
            dbc.Row([
                dbc.Col(html.Strong(symbol), width=2),
                dbc.Col(group["asset_type"].capitalize(), width=1),
                dbc.Col(f"{group['total_shares']:.2f} shares", width=2),
                dbc.Col(f"${group['current_price']:.2f} {group['currency']}", width=2),
                dbc.Col(f"${group['total_book_value']:.2f}", width=1),
                dbc.Col(f"${group['total_current_value']:.2f}", width=1),
                dbc.Col(
                    f"${group['total_gain_loss']:.2f}", 
                    style={"color": "green" if group['total_gain_loss'] >= 0 else "red"},
                    width=1
                ),
                dbc.Col(
                    f"{group['total_gain_loss_percent']:.2f}%", 
                    style={"color": "green" if group['total_gain_loss_percent'] >= 0 else "red"},
                    width=1
                ),
                dbc.Col(group["currency"], width=1)
            ], className="g-0 w-100")  # Added w-100 class to ensure full width
        ], className="w-100 portfolio-accordion-header")  # Added custom class for styling
        
        # Create detailed view with tabs for Positions and Transactions
        transaction_table = None
        if group.get("transactions"):
            transaction_table = dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Type"),
                        html.Th("Shares"),
                        html.Th("Price"),
                        html.Th("Total Amount"),
                        html.Th("Currency"),
                        html.Th("Notes")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(tx.get("transaction_date", "")),
                        html.Td(tx.get("transaction_type", "").capitalize()),
                        html.Td(f"{tx.get('shares', 0):.4f}"),
                        html.Td(f"${tx.get('price', 0):.2f}"),
                        html.Td(f"${tx.get('total_amount', 0):.2f}"),
                        html.Td(tx.get("currency", "")),
                        html.Td(tx.get("notes", ""))
                    ]) for tx in group["transactions"]
                ])
            ], bordered=True, hover=True, size="sm", className="mt-2")
        else:
            transaction_table = html.P("No transaction records found for this asset.")
        
        # Create table of individual positions
        positions_table = dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Purchase Date"),
                    html.Th("Shares"),
                    html.Th("Purchase Price"),
                    html.Th("Book Value"),
                    html.Th("Current Value"),
                    html.Th("Gain/Loss"),
                    html.Th("Gain/Loss (%)"),
                    html.Th("Actions")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(inv["purchase_date"]),
                    html.Td(f"{inv['shares']:.4f}"),
                    html.Td(f"${inv['purchase_price']:.2f}"),
                    html.Td(f"${inv['shares'] * inv['purchase_price']:.2f}"),
                    html.Td(f"${inv['current_value']:.2f}"),
                    html.Td(
                        f"${inv['gain_loss']:.2f}", 
                        style={"color": "green" if inv['gain_loss'] >= 0 else "red"}
                    ),
                    html.Td(
                        f"{inv['gain_loss_percent']:.2f}%", 
                        style={"color": "green" if inv['gain_loss_percent'] >= 0 else "red"}
                    ),
                    html.Td(
                        dbc.Button(
                            "Remove", 
                            id={"type": "remove-investment-button", "index": inv["id"]},
                            color="danger",
                            size="sm"
                        )
                    )
                ]) for inv in group["investments"]
            ])
        ], bordered=True, hover=True, size="sm", className="mt-2")
        
        # Create tabs for positions and transactions
        detailed_content = dbc.Tabs([
            dbc.Tab(positions_table, label="Positions"),
            dbc.Tab(transaction_table, label="Transactions")
        ], className="mt-3")
        
        # Create the accordion item for this group
        accordion_items.append(
            dbc.AccordionItem(
                detailed_content,
                title=header,
                item_id=f"acc-{symbol}"
            )
        )
    
    # Create accordion with all items
    accordion = dbc.Accordion(
        accordion_items,
        start_collapsed=True,
        flush=True,
        id="portfolio-accordion",
        class_name="portfolio-accordion"  # Added custom class for CSS targeting
    )
    
    # Add a summary row for the entire portfolio
    total_book_value = sum(group["total_book_value"] for group in grouped_investments.values())
    total_current_value = sum(group["total_current_value"] for group in grouped_investments.values())
    total_gain_loss = total_current_value - total_book_value
    total_gain_loss_percent = (total_gain_loss / total_book_value * 100) if total_book_value > 0 else 0
    
    portfolio_summary = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5("Portfolio Summary"), width=4),
                dbc.Col(html.H5(f"Book Value: ${total_book_value:.2f}"), width=2),
                dbc.Col(html.H5(f"Current Value: ${total_current_value:.2f}"), width=2),
                dbc.Col(
                    html.H5(f"Gain/Loss: ${total_gain_loss:.2f}"), 
                    style={"color": "green" if total_gain_loss >= 0 else "red"},
                    width=2
                ),
                dbc.Col(
                    html.H5(f"{total_gain_loss_percent:.2f}%"), 
                    style={"color": "green" if total_gain_loss_percent >= 0 else "red"},
                    width=2
                )
            ])
        ])
    ], className="mb-3")
    
    print("Portfolio accordion and summary created successfully")
    return html.Div([portfolio_summary, accordion])