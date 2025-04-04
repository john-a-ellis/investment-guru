# components/risk_metrics_component.py
"""
Dashboard component for visualizing portfolio risk metrics.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from modules.portfolio_risk_metrics import create_risk_metrics_component

def create_risk_visualization_component():
    """
    Creates a component for visualizing portfolio risk metrics
    
    Returns:
        dbc.Card: Risk metrics visualization component
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Risk Analysis"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div(id="risk-metrics-content")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Time Period"),
                    dbc.RadioItems(
                        id="risk-period-selector",
                        options=[
                            {"label": "3 Months", "value": "3m"},
                            {"label": "6 Months", "value": "6m"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "All Time", "value": "all"}
                        ],
                        value="1y",
                        inline=True
                    )
                ], width=12)
            ]),
            dcc.Interval(
                id="risk-update-interval",
                interval=3600000,  # 1 hour in milliseconds
                n_intervals=0
            )
        ])
    ])