# components/wealthsimple_integration.py
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_wealthsimple_component():
    """
    Creates a component for Wealthsimple integration
    """
    return dbc.Card([
        dbc.CardHeader("Wealthsimple Integration"),
        dbc.CardBody([
            html.P("Connect your Wealthsimple account to import your current portfolio or export recommendations."),
            
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Wealthsimple Email"),
                        dbc.Input(
                            id="ws-email",
                            type="email",
                            placeholder="Enter your Wealthsimple email",
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Portfolio Type"),
                        dcc.Dropdown(
                            id="ws-portfolio-type",
                            options=[
                                {"label": "Personal", "value": "personal"},
                                {"label": "TFSA", "value": "tfsa"},
                                {"label": "RRSP", "value": "rrsp"}
                            ],
                            value="tfsa"
                        )
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Import Portfolio", id="import-ws-button", color="primary", className="mt-3 me-2"),
                        dbc.Button("Export Recommendations", id="export-ws-button", color="secondary", className="mt-3"),
                    ])
                ])
            ]),
            
            html.Div(id="ws-integration-status", className="mt-3")
        ])
    ])

def import_from_wealthsimple(email, portfolio_type):
    """
    Mock function to import portfolio from Wealthsimple
    In production, this would use the Wealthsimple API
    """
    # For demo purposes, we'll return a mock portfolio based on your current investments
    mock_portfolio = {
        'CGL.TO': {'name': 'iShares Gold Bullion ETF', 'shares': 50, 'value': 1125.50, 'cost_basis': 1050.00},
        'XTR.TO': {'name': 'iShares Diversified Monthly Income ETF', 'shares': 200, 'value': 2240.00, 'cost_basis': 2100.00},
        'CWW.TO': {'name': 'iShares Global Water Index ETF', 'shares': 30, 'value': 1500.00, 'cost_basis': 1380.00},
        'MFC.TO': {'name': 'Manulife Financial Corp.', 'shares': 100, 'value': 2600.00, 'cost_basis': 2400.00},
        'TRI.TO': {'name': 'Thomson Reuters Corp.', 'shares': 15, 'value': 2925.00, 'cost_basis': 2700.00},
        'PNG.V': {'name': 'Kraken Robotics Inc.', 'shares': 1000, 'value': 500.00, 'cost_basis': 600.00}
    }
    
    return mock_portfolio

# Add the necessary callbacks to main.py
"""
@app.callback(
    Output("ws-integration-status", "children"),
    [Input("import-ws-button", "n_clicks"),
     Input("export-ws-button", "n_clicks")],
    [State("ws-email", "value"),
     State("ws-portfolio-type", "value")],
    prevent_initial_call=True
)
def handle_wealthsimple_integration(import_clicks, export_clicks, email, portfolio_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "import-ws-button" and email:
        try:
            # In a real application, this would call the actual Wealthsimple API
            portfolio = import_from_wealthsimple(email, portfolio_type)
            
            # Here we would add the imported portfolio to our system
            return dbc.Alert(
                f"Successfully imported {len(portfolio)} holdings from your {portfolio_type.upper()} account.",
                color="success"
            )
        except Exception as e:
            return dbc.Alert(f"Error importing portfolio: {str(e)}", color="danger")
    
    elif button_id == "export-ws-button" and email:
        return dbc.Alert(
            "Export functionality would connect to Wealthsimple API to send recommendations.",
            color="info"
        )
    
    return ""
"""