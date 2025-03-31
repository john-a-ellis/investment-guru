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
                        dbc.Input(id="investment-symbol-input", placeholder="Symbol (e.g., MFC.TO)"),
                        dbc.Input(id="investment-shares-input", type="number", placeholder="Number of Shares"),
                        dbc.Input(id="investment-price-input", type="number", placeholder="Purchase Price"),
                        dbc.Input(id="investment-date-input", type="date", 
                                 value=datetime.now().strftime("%Y-%m-%d")),
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

def create_portfolio_table(portfolio):
    """
    Create a table to display current portfolio investments grouped by ticker
    """
    if not portfolio:
        return html.Div("No investments currently tracked.")
    
    # Group investments by symbol
    grouped_investments = {}
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        if symbol not in grouped_investments:
            grouped_investments[symbol] = {
                "investments": [],
                "total_shares": 0,
                "total_book_value": 0,
                "total_current_value": 0,
                "total_gain_loss": 0,
                "name": details.get("name", symbol),
                "currency": details.get("currency", "USD")
            }
        
        # Add to group totals
        grouped_investments[symbol]["investments"].append({
            "id": investment_id,
            **details
        })
        
        grouped_investments[symbol]["total_shares"] += details.get("shares", 0)
        grouped_investments[symbol]["total_book_value"] += details.get("shares", 0) * details.get("purchase_price", 0)
        grouped_investments[symbol]["total_current_value"] += details.get("current_value", 0)
        grouped_investments[symbol]["total_gain_loss"] += details.get("gain_loss", 0)
    
    # Calculate gain/loss percentage for each group
    for symbol, group in grouped_investments.items():
        if group["total_book_value"] > 0:
            group["total_gain_loss_percent"] = (group["total_gain_loss"] / group["total_book_value"]) * 100
        else:
            group["total_gain_loss_percent"] = 0
    
    # Create accordion items for each symbol
    accordion_items = []
    total_portfolio_value = 0
    total_portfolio_book = 0
    total_portfolio_gain_loss = 0
    
    for symbol, group in grouped_investments.items():
        currency = group["currency"]
        total_portfolio_value += group["total_current_value"]
        total_portfolio_book += group["total_book_value"]
        total_portfolio_gain_loss += group["total_gain_loss"]
        
        # Create the header row with consistent alignment
        header = html.Div([
            dbc.Row([
                dbc.Col(symbol, width=2),
                dbc.Col(f"{group['total_shares']:.3f}", width=1, className="text-center"),
                dbc.Col(f"${group['total_book_value']:.2f} {currency}", width=2, className="text-center"),
                dbc.Col(f"${group['total_current_value']:.2f} {currency}", width=2, className="text-center"),
                dbc.Col(
                    f"${group['total_gain_loss']:.2f} {currency}", 
                    style={"color": "green" if group['total_gain_loss'] >= 0 else "red"},
                    width=2, className="text-center"
                ),
                dbc.Col(
                    f"{group['total_gain_loss_percent']:.2f}%", 
                    style={"color": "green" if group['total_gain_loss_percent'] >= 0 else "red"},
                    width=2, className="text-center"
                )
            ])
        ])
        
        # Create detailed transactions table
        transactions_table = dbc.Table([
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
                    html.Td(f"{inv['shares']:.3f}"),
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
        ], striped=True, bordered=True, hover=True, size="sm")
        
        # Add to accordion
        accordion_items.append(
            dbc.AccordionItem(
                transactions_table,
                title=header,
                item_id=symbol
            )
        )
    
    # Create portfolio totals row
    portfolio_total = html.Div([
        dbc.Row([
            dbc.Col(html.B("PORTFOLIO TOTAL"), width=2),
            dbc.Col("", width=1),
            dbc.Col(html.B(f"${total_portfolio_book:.2f}"), width=2, className="text-center"),
            dbc.Col(html.B(f"${total_portfolio_value:.2f}"), width=2, className="text-center"),
            dbc.Col(
                html.B(f"${total_portfolio_gain_loss:.2f}"), 
                style={"color": "green" if total_portfolio_gain_loss >= 0 else "red"},
                width=2, className="text-center"
            ),
            dbc.Col(
                html.B(f"{(total_portfolio_gain_loss / total_portfolio_book * 100) if total_portfolio_book > 0 else 0:.2f}%"), 
                style={"color": "green" if total_portfolio_gain_loss >= 0 else "red"},
                width=2, className="text-center"
            )
        ], className="bg-light p-2 mb-3 rounded")
    ])
    
    # Create column headers with exact same widths as accordion content
    headers = dbc.Row([
        dbc.Col(html.B("Symbol"), width=2, className="ps-3"),  # Add left padding to match accordion
        dbc.Col(html.B("Shares"), width=1, className="text-center"),
        dbc.Col(html.B("Book Value"), width=2, className="text-center"),
        dbc.Col(html.B("Current Value"), width=2, className="text-center"),
        dbc.Col(html.B("Gain/Loss"), width=2, className="text-center"),
        dbc.Col(html.B("Gain/Loss (%)"), width=2, className="text-center"),
        dbc.Col("", width=1)  # Empty space for the accordion arrow
    ], className="mb-2 fw-bold")
    
    # Apply container styles to ensure alignment
    return html.Div([
        portfolio_total,
        # Wrap headers in a container that matches accordion styling
        html.Div(
            headers,
            className="border-bottom"  # Match accordion styling
        ),
        dbc.Accordion(
            accordion_items,
            start_collapsed=True,
            flush=True,
            id="portfolio-accordion",
            className="accordion-alignment"  # Custom class for additional styling
        )
    ], className="portfolio-container")  # Container to group everything