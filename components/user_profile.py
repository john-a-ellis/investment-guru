# components/user_profile.py
from dash import dcc, html
import dash_bootstrap_components as dbc
from modules.portfolio_utils import load_user_profile

def create_user_profile_component():
    """
    Creates a component for user profile settings with persistence
    """
    # Load saved profile if it exists
    profile = load_user_profile()
    
    return dbc.Card([
        dbc.CardHeader("User Profile"),
        dbc.CardBody([
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Risk Tolerance"),
                        dcc.Slider(
                            id="risk-slider",
                            min=1,
                            max=10,
                            step=1,
                            marks={i: str(i) for i in range(1, 11)},
                            value=profile.get("risk_level", 5),
                            persistence=True,
                            persistence_type="local"
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Investment Horizon"),
                        dcc.Dropdown(
                            id="investment-horizon",
                            options=[
                                {"label": "Short Term (< 1 year)", "value": "short"},
                                {"label": "Medium Term (1-5 years)", "value": "medium"},
                                {"label": "Long Term (> 5 years)", "value": "long"}
                            ],
                            value=profile.get("investment_horizon", "medium"),
                            persistence=True,
                            persistence_type="local"
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Initial Investment"),
                        dbc.Input(
                            id="initial-investment",
                            type="number",
                            placeholder="Enter amount",
                            value=profile.get("initial_investment", 10000),
                            persistence=True,
                            persistence_type="local"
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Update Profile", id="update-profile-button", color="primary", className="mt-3")
                    ], width=12)
                ])
            ]),
            html.Div(id="profile-update-status", className="mt-3")
        ])
    ], className="mb-4")

# def load_user_profile():
#     """
#     Load user profile from storage file
#     """
#     try:
#         if os.path.exists('data/user_profile.json'):
#             with open('data/user_profile.json', 'r') as f:
#                 return json.load(f)
#         else:
#             # Default profile if no file exists
#             default_profile = {
#                 "risk_level": 5,
#                 "investment_horizon": "medium",
#                 "initial_investment": 10000,
#                 "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
#             save_user_profile(default_profile)
#             return default_profile
#     except Exception as e:
#         print(f"Error loading user profile: {e}")
#         return {}

# def save_user_profile(profile):
#     """
#     Save user profile to storage file
#     """
#     try:
#         os.makedirs('data', exist_ok=True)
#         with open('data/user_profile.json', 'w') as f:
#             json.dump(profile, f, indent=4)
#         return True
#     except Exception as e:
#         print(f"Error saving user profile: {e}")
#         return False