# components/asset_tracker.py
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd


def create_asset_tracker_component():
    """
    Creates a component for users to add and remove tracked assets, including mutual funds
    """
    return dbc.Card([
        dbc.CardHeader("Tracked Assets Manager"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Add New Asset"),
                    dbc.InputGroup([
                        dbc.Input(id="asset-symbol-input", placeholder="Symbol (e.g., MFC.TO or fund code)"),
                        dbc.Input(id="asset-name-input", placeholder="Name (e.g., Manulife Financial)"),
                        dbc.Select(
                            id="asset-type-select",
                            options=[
                                {"label": "Stock", "value": "stock"},
                                {"label": "ETF", "value": "etf"},
                                {"label": "Mutual Fund", "value": "mutual_fund"},
                                {"label": "Cryptocurrency", "value": "crypto"},
                                {"label": "Bond", "value": "bond"},
                                {"label": "Cash", "value": "cash"}
                            ],
                            value="stock"
                        ),
                        dbc.Button("Add", id="add-asset-button", color="success")
                    ]),
                    html.Div(id="add-asset-feedback", className="mt-2")
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Currently Tracked Assets"),
                    html.Div(id="tracked-assets-table")
                ], width=12)
            ])
        ])
    ])

# def load_tracked_assets():
#     """
#     Load tracked assets from storage file
#     """
#     try:
#         if os.path.exists('data/tracked_assets.json'):
#             with open('data/tracked_assets.json', 'r') as f:
#                 return json.load(f)
#         else:
#             # Default assets if no file exists
#             default_assets = {
#                 "CGL.TO": {"name": "iShares Gold Bullion ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
#                 "XTR.TO": {"name": "iShares Diversified Monthly Income ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
#                 "CWW.TO": {"name": "iShares Global Water Index ETF", "type": "etf", "added_date": datetime.now().strftime("%Y-%m-%d")},
#                 "MFC.TO": {"name": "Manulife Financial Corp.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")},
#                 "TRI.TO": {"name": "Thomson Reuters Corp.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")},
#                 "PNG.V": {"name": "Kraken Robotics Inc.", "type": "stock", "added_date": datetime.now().strftime("%Y-%m-%d")}
#             }
#             save_tracked_assets(default_assets)
#             return default_assets
#     except Exception as e:
#         print(f"Error loading tracked assets: {e}")
#         return {}

# def save_tracked_assets(assets):
#     """
#     Save tracked assets to storage file
#     """
#     try:
#         os.makedirs('data', exist_ok=True)
#         with open('data/tracked_assets.json', 'w') as f:
#             json.dump(assets, f, indent=4)
#         return True
#     except Exception as e:
#         print(f"Error saving tracked assets: {e}")
#         return False

# Update the create_tracked_assets_table function in components/asset_tracker.py

def create_tracked_assets_table(assets):
    """
    Create a table to display tracked assets with consistent IDs for pattern-matching callbacks
    """
    if not assets:
        return html.Div("No assets currently tracked.")
    
    # Convert to DataFrame for easier table creation
    assets_list = []
    for symbol, details in assets.items():
        assets_list.append({
            "symbol": symbol,
            "name": details.get("name", ""),
            "type": details.get("type", ""),
            "added_date": details.get("added_date", "")
        })
    
    df = pd.DataFrame(assets_list)
    
    # Create table
    return dbc.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbol"),
                html.Th("Name"),
                html.Th("Type"),
                html.Th("Added Date"),
                html.Th("Actions")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["symbol"]),
                html.Td(row["name"]),
                html.Td(row["type"].capitalize()),
                html.Td(row["added_date"]),
                html.Td(
                    dbc.Button(
                        "Remove", 
                        id={
                            "type": "remove-asset-button", 
                            "index": row["symbol"]
                        },
                        color="danger",
                        size="sm",
                        n_clicks=0  # Initialize n_clicks to ensure it's tracking
                    )
                )
            ]) for _, row in df.iterrows()
        ])
    ], striped=True, bordered=True, hover=True)