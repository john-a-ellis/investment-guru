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

def create_portfolio_table(portfolio):
    """
    Create a table to display current portfolio investments
    """
    if not portfolio:
        return html.Div("No investments currently tracked.")
    
    # Convert to DataFrame for easier table creation
    investments_list = []
    for investment_id, details in portfolio.items():
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
            "currency": currency
        })
    
    df = pd.DataFrame(investments_list)
    
    # Create table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbol"),
                html.Th("Shares"),
                html.Th("Purchase Price"),
                html.Th("Purchase Date"),
                html.Th("Current Price"),
                html.Th("Current Value"),
                html.Th("Gain/Loss"),
                html.Th("Gain/Loss (%)"),
                html.Th("Actions")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["symbol"]),
                html.Td(f"{row['shares']:.2f}"),
                html.Td(f"${row['purchase_price']:.2f} {row['currency']}"),
                html.Td(row["purchase_date"]),
                html.Td(f"${row['current_price']:.2f} {row['currency']}"),
                html.Td(f"${row['current_value']:.2f} {row['currency']}"),
                html.Td(
                    f"${row['gain_loss']:.2f} {row['currency']}", 
                    style={"color": "green" if row['gain_loss'] >= 0 else "red"}
                ),
                html.Td(
                    f"{row['gain_loss_percent']:.2f}%", 
                    style={"color": "green" if row['gain_loss_percent'] >= 0 else "red"}
                ),
                html.Td(
                    dbc.Button(
                        "Remove", 
                        id={"type": "remove-investment-button", "index": row["id"]},
                        color="danger",
                        size="sm"
                    )
                )
            ]) for _, row in df.iterrows()
        ])
    ], striped=True, bordered=True, hover=True, responsive=True)