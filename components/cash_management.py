# components/cash_management.py
from dash import html
import dash_bootstrap_components as dbc
from datetime import datetime

def create_cash_management_component():
    """
    Creates a component for managing cash positions (deposits and withdrawals)
    """
    return dbc.Card([
        dbc.CardHeader("Capital Management"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Deposit Form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Amount"),
                                dbc.Input(id="deposit-amount-input", type="number", placeholder="Amount", step="0.01")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Currency"),
                                dbc.Select(
                                    id="deposit-currency-select",
                                    options=[
                                        {"label": "CAD", "value": "CAD"},
                                        {"label": "USD", "value": "USD"}
                                    ],
                                    value="CAD"
                                )
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="deposit-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Description"),
                                dbc.Input(id="deposit-description-input", placeholder="Optional description")
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Record Deposit", id="record-deposit-button", color="success", className="mt-4")
                            ], width=1)
                        ]),
                        html.Div(id="deposit-feedback", className="mt-2")
                    ])
                ], label="Deposit"),
                
                dbc.Tab([
                    # Withdrawal Form
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Amount"),
                                dbc.Input(id="withdrawal-amount-input", type="number", placeholder="Amount", step="0.01")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Currency"),
                                dbc.Select(
                                    id="withdrawal-currency-select",
                                    options=[
                                        {"label": "CAD", "value": "CAD"},
                                        {"label": "USD", "value": "USD"}
                                    ],
                                    value="CAD"
                                )
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Date"),
                                dbc.Input(id="withdrawal-date-input", type="date", value=datetime.now().strftime("%Y-%m-%d"))
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Description"),
                                dbc.Input(id="withdrawal-description-input", placeholder="Optional description")
                            ], width=3),
                            dbc.Col([
                                dbc.Button("Record Withdrawal", id="record-withdrawal-button", color="danger", className="mt-4")
                            ], width=1)
                        ]),
                        html.Div(id="withdrawal-feedback", className="mt-2")
                    ])
                ], label="Withdrawal"),
                
                dbc.Tab([
                    # Cash Flow History
                    html.Div(id="cash-flow-history-table")
                ], label="Cash Flow History")
            ], id="cash-tabs")
        ])
    ])

def create_cash_flow_table():
    """
    Create a table showing cash flow history
    """
    from modules.portfolio_utils import load_cash_flows
    
    cash_flows = load_cash_flows()
    
    if not cash_flows:
        return html.Div("No cash flow transactions recorded yet.")
    
    return dbc.Table([
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
                html.Td(flow['flow_type'].capitalize(), style={"color": "green" if flow['flow_type'] == "deposit" else "red"}),
                html.Td(f"${flow['amount']:.2f}"),
                html.Td(flow['currency']),
                html.Td(flow['description'])
            ]) for flow in cash_flows
        ])
    ], striped=True, bordered=True, hover=True)