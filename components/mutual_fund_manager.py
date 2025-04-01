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

# Add the necessary callbacks to main.py

"""
@app.callback(
    Output("add-fund-price-feedback", "children"),
    Input("add-fund-price-button", "n_clicks"),
    [State("fund-code-input", "value"),
     State("fund-date-input", "value"),
     State("fund-price-input", "value")],
    prevent_initial_call=True
)
def add_fund_price(n_clicks, fund_code, date, price):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    if not fund_code or not date or not price:
        return dbc.Alert("Fund code, date, and price are required", color="warning")
    
    # Standardize fund code
    fund_code = fund_code.upper().strip()
    
    # Add price to the provider
    provider = MutualFundProvider()
    success = provider.add_manual_price(fund_code, date, price)
    
    if success:
        return dbc.Alert(f"Successfully added price for {fund_code}", color="success")
    else:
        return dbc.Alert("Failed to add price", color="danger")

@app.callback(
    Output("fund-price-history", "children"),
    Input("view-fund-history-button", "n_clicks"),
    State("fund-filter-input", "value"),
    prevent_initial_call=True
)
def view_fund_history(n_clicks, fund_code):
    if not n_clicks or not fund_code:
        raise dash.exceptions.PreventUpdate
    
    # Standardize fund code
    fund_code = fund_code.upper().strip()
    
    # Get historical data
    provider = MutualFundProvider()
    hist_data = provider.get_historical_data(fund_code)
    
    if hist_data.empty:
        return dbc.Alert(f"No price data found for {fund_code}", color="warning")
    
    # Create a DataFrame with the data
    df = hist_data.reset_index()
    df.columns = ['Date', 'Price']
    
    # Format the date
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Create table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Date"),
                html.Th("Price")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row['Date']),
                html.Td(f"${row['Price']:.4f}")
            ]) for _, row in df.iterrows()
        ])
    ], striped=True, bordered=True, hover=True)
"""