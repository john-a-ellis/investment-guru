# components/mutual_fund_manager.py
from dash import html
import dash_bootstrap_components as dbc
from datetime import datetime

def create_mutual_fund_manager_component():
    """
    Creates a component for managing mutual fund data
    """
    return dbc.Card([
        dbc.CardHeader("Mutual Fund Data Manager"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Add Price Point"),
                    dbc.InputGroup([
                        dbc.Input(id="fund-code-input", placeholder="Fund Code (e.g., MAW104)"),
                        dbc.Input(id="fund-date-input", type="date", 
                                 value=datetime.now().strftime("%Y-%m-%d")),
                        dbc.Input(id="fund-price-input", type="number", placeholder="NAV Price", 
                                 step="0.01"),
                        dbc.Button("Add Price", id="add-fund-price-button", color="success")
                    ]),
                    html.Div(id="add-fund-price-feedback", className="mt-2")
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Mutual Fund Price History"),
                    dbc.InputGroup([
                        dbc.Input(id="fund-filter-input", placeholder="Enter Fund Code to View"),
                        dbc.Button("View", id="view-fund-history-button", color="primary")
                    ], className="mb-3"),
                    html.Div(id="fund-price-history")
                ], width=12)
            ])
        ])
    ])
