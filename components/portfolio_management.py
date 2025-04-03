# components/portfolio_management.py
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime

def create_portfolio_management_component():
    """
    Creates a component for tracking actual investments with performance data
    and integrated transaction recording
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
                            id="investment-type-select",
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
            
            # Quick transaction row
            dbc.Row([
                dbc.Col([
                    html.H5("Quick Transaction Recording", className="mt-3"),
                    dbc.InputGroup([
                        dbc.Select(
                            id="quick-transaction-type",
                            options=[
                                {"label": "Buy", "value": "buy"},
                                {"label": "Sell", "value": "sell"}
                            ],
                            value="buy",
                            style={"width": "80px"}
                        ),
                        dbc.Input(id="quick-transaction-symbol", placeholder="Symbol", style={"width": "100px"}),
                        dbc.Input(id="quick-transaction-shares", type="number", placeholder="Shares", step="0.001", style={"width": "100px"}),
                        dbc.Input(id="quick-transaction-price", type="number", placeholder="Price", style={"width": "100px"}),
                        dbc.Input(id="quick-transaction-date", type="date", value=datetime.now().strftime("%Y-%m-%d"), style={"width": "150px"}),
                        dbc.Button("Record Transaction", id="record-quick-transaction", color="primary")
                    ]),
                    html.Div(id="quick-transaction-feedback", className="mt-2")
                ], width=12)
            ], className="mb-3"),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Current Portfolio"),
                    html.Div(id="portfolio-table")
                ], width=12)
            ])
        ])
    ])

# def load_portfolio():
#     """
#     Load portfolio data from storage file
#     """
#     try:
#         if os.path.exists('data/portfolio.json'):
#             with open('data/portfolio.json', 'r') as f:
#                 return json.load(f)
#         else:
#             # Default empty portfolio if no file exists
#             return {}
#     except Exception as e:
#         print(f"Error loading portfolio: {e}")
#         return {}

# def save_portfolio(portfolio):
#     """
#     Save portfolio data to storage file
#     """
#     try:
#         os.makedirs('data', exist_ok=True)
#         with open('data/portfolio.json', 'w') as f:
#             json.dump(portfolio, f, indent=4)
#         return True
#     except Exception as e:
#         print(f"Error saving portfolio: {e}")
#         return False

# def load_transactions():
#     """
#     Load transaction records from storage file
    
#     Returns:
#         dict: Transaction records
#     """
#     try:
#         if os.path.exists('data/transactions.json'):
#             with open('data/transactions.json', 'r') as f:
#                 return json.load(f)
#         else:
#             # Default empty transactions if no file exists
#             return {}
#     except Exception as e:
#         print(f"Error loading transactions: {e}")
#         return {}

def create_portfolio_table(portfolio):
    """
    Create a table to display current portfolio investments with accordion components
    grouped by asset symbol, integrated with transaction history and transaction forms
    """
    # Load transactions to enhance the portfolio view
    from modules.transaction_tracker import load_transactions
    transactions = load_transactions()
    
    if not portfolio:
        return html.Div("No investments currently tracked.")
    
    # Convert to DataFrame for easier processing
    investments_list = []
    for investment_id, details in portfolio.items():
        try:
            # Ensure all numeric values are converted to float
            shares = float(details.get("shares", 0))
            purchase_price = float(details.get("purchase_price", 0))
            current_price = float(details.get("current_price", 0))
            current_value = float(details.get("current_value", 0))
            gain_loss = float(details.get("gain_loss", 0))
            gain_loss_percent = float(details.get("gain_loss_percent", 0))
            
            currency = details.get("currency", "USD")
            
            investments_list.append({
                "id": investment_id,
                "symbol": details.get("symbol", ""),
                "shares": shares,
                "purchase_price": purchase_price,
                "purchase_date": details.get("purchase_date", ""),
                "current_price": current_price,
                "current_value": current_value,
                "gain_loss": gain_loss,
                "gain_loss_percent": gain_loss_percent,
                "currency": currency,
                "asset_type": details.get("asset_type", "stock")
            })
        except Exception as e:
            print(f"Error converting investment data: {e}")
            continue
    
    df = pd.DataFrame(investments_list)
    
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
                "current_price": row["current_price"],
                "currency": row["currency"],
                "asset_type": row["asset_type"]
            }
        
        # Add this investment to the group
        grouped_investments[symbol]["investments"].append(row)
        
        # Update group totals - ensure all values are float
        grouped_investments[symbol]["total_shares"] += float(row["shares"])
        grouped_investments[symbol]["total_book_value"] += float(row["shares"]) * float(row["purchase_price"])
        grouped_investments[symbol]["total_current_value"] += float(row["current_value"])
    
    # Add transactions to each symbol group
    for transaction_id, transaction in transactions.items():
        symbol = transaction.get("symbol", "").upper().strip()
        if symbol in grouped_investments:
            # Convert any Decimal values to float
            try:
                transaction_shares = float(transaction.get("shares", 0))
                transaction_price = float(transaction.get("price", 0))
                transaction_amount = float(transaction.get("amount", 0))
                
                grouped_investments[symbol]["transactions"].append({
                    "id": transaction_id,
                    "date": transaction.get("date", ""),
                    "type": transaction.get("type", ""),
                    "shares": transaction_shares,
                    "price": transaction_price,
                    "amount": transaction_amount,
                    "notes": transaction.get("notes", "")
                })
            except Exception as e:
                print(f"Error converting transaction data: {e}")
                continue
    
    # Calculate group gain/loss
    for symbol, group in grouped_investments.items():
        group["total_gain_loss"] = float(group["total_current_value"]) - float(group["total_book_value"])
        if float(group["total_book_value"]) > 0:
            group["total_gain_loss_percent"] = (float(group["total_gain_loss"]) / float(group["total_book_value"])) * 100
        else:
            group["total_gain_loss_percent"] = 0
        
        # Sort transactions by date (newest first)
        if group["transactions"]:
            group["transactions"] = sorted(
                group["transactions"], 
                key=lambda x: x.get("date", ""), 
                reverse=True
            )
    
    # Rest of function remains the same...
    # Create accordion items for each symbol group
    accordion_items = []
    
    # Sort groups by current value (descending)
    sorted_groups = sorted(grouped_investments.items(), key=lambda x: x[1]["total_current_value"], reverse=True)
    
    for symbol, group in sorted_groups:
        # Create the header with summary information
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
            ], className="g-0 w-100")
        ], className="w-100 portfolio-accordion-header")
        
        # Create transaction forms for this symbol
        transaction_forms = dbc.Row([
            dbc.Col([
                # Buy Form
                dbc.Card([
                    dbc.CardHeader("Record Buy Transaction"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Shares"),
                                    dbc.Input(
                                        id={"type": "buy-shares-input", "symbol": symbol},
                                        type="number", 
                                        placeholder="Shares",
                                        step="0.001"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Price"),
                                    dbc.Input(
                                        id={"type": "buy-price-input", "symbol": symbol},
                                        type="number", 
                                        placeholder="Price",
                                        value=group["current_price"]
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Date"),
                                    dbc.Input(
                                        id={"type": "buy-date-input", "symbol": symbol},
                                        type="date", 
                                        value=datetime.now().strftime("%Y-%m-%d")
                                    )
                                ], width=4)
                            ]),
                            dbc.Button(
                                "Buy", 
                                id={"type": "record-buy-button", "symbol": symbol},
                                color="success", 
                                className="mt-3"
                            ),
                            html.Div(id={"type": "buy-feedback", "symbol": symbol}, className="mt-2")
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                # Sell Form
                dbc.Card([
                    dbc.CardHeader("Record Sell Transaction"),
                    dbc.CardBody([
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Shares"),
                                    dbc.Input(
                                        id={"type": "sell-shares-input", "symbol": symbol},
                                        type="number", 
                                        placeholder="Shares",
                                        step="0.001"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Price"),
                                    dbc.Input(
                                        id={"type": "sell-price-input", "symbol": symbol},
                                        type="number", 
                                        placeholder="Price",
                                        value=group["current_price"]
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Date"),
                                    dbc.Input(
                                        id={"type": "sell-date-input", "symbol": symbol},
                                        type="date", 
                                        value=datetime.now().strftime("%Y-%m-%d")
                                    )
                                ], width=4)
                            ]),
                            dbc.Button(
                                "Sell", 
                                id={"type": "record-sell-button", "symbol": symbol},
                                color="danger", 
                                className="mt-3"
                            ),
                            html.Div(id={"type": "sell-feedback", "symbol": symbol}, className="mt-2")
                        ])
                    ])
                ])
            ], width=6)
        ], className="mt-3")
        
        # Create transaction table if available
        transaction_table = None
        if group["transactions"]:
            transaction_table = dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Type"),
                        html.Th("Shares"),
                        html.Th("Price"),
                        html.Th("Total Amount"),
                        html.Th("Notes")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(tx.get("date", "")),
                        html.Td(tx.get("type", "").capitalize(), 
                               style={"color": "green" if tx.get("type") == "buy" else "red"}),
                        html.Td(f"{tx.get('shares', 0):.4f}"),
                        html.Td(f"${tx.get('price', 0):.2f}"),
                        html.Td(f"${tx.get('amount', 0):.2f}"),
                        html.Td(tx.get("notes", ""))
                    ]) for tx in group["transactions"]
                ])
            ], bordered=True, hover=True, size="sm", className="mt-3")
        else:
            transaction_table = html.P("No transaction records found for this asset.")
        
        # Create positions table (investment lots)
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
                            # Convert ID to string to ensure it's properly passed
                            id={"type": "remove-investment-button", "index": str(inv["id"])},
                            color="danger",
                            size="sm"
                        )
                    )
                ]) for inv in group["investments"]
            ])
        ], bordered=True, hover=True, size="sm", className="mt-3")
        
        # Create tabs for positions, transactions, and transaction forms
        detailed_content = dbc.Tabs([
            dbc.Tab(positions_table, label="Positions"),
            dbc.Tab(transaction_table, label="Transaction History"),
            dbc.Tab(transaction_forms, label="Record Transaction")
        ], className="mt-3")
        
        # Create accordion item
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
        id="portfolio-accordion"
    )
    
    # Add portfolio summary
    total_book_value = sum(float(group["total_book_value"]) for group in grouped_investments.values())
    total_current_value = sum(float(group["total_current_value"]) for group in grouped_investments.values())
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
    
    return html.Div([portfolio_summary, accordion])