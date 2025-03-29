# Investment Recommendation System Architecture
# main.py - Entry point for the application

import dash
from dash import dcc, html, callback, Input, Output, State,  ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from dash.exceptions import PreventUpdate
import json
import os
from datetime import datetime
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom modules
from modules.data_collector import DataCollector
from modules.market_analyzer import MarketAnalyzer
from modules.news_analyzer import NewsAnalyzer
from modules.recommendation_engine import RecommendationEngine
from modules.portfolio_tracker import PortfolioTracker
from components.asset_tracker import create_asset_tracker_component, load_tracked_assets, create_tracked_assets_table
from components.user_profile import create_user_profile_component


# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # For production deployment

# Initialize system components
data_collector = DataCollector()
market_analyzer = MarketAnalyzer()
news_analyzer = NewsAnalyzer()
recommendation_engine = RecommendationEngine()
portfolio_tracker = PortfolioTracker()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("AI Investment Recommendation System", className="text-primary my-4"),
            html.P("Advanced investment strategies powered by AI market analysis")
        ], width=12)
    ]),
    # Add Asset Tracker component
    dbc.Row([
        dbc.Col([
            create_asset_tracker_component()
        ], width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            create_user_profile_component()
        ], width=4),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Market Overview - Tracked Assets"),
                    dbc.CardBody([
                        dcc.Graph(id="market-overview-graph"),
                        dcc.Interval(
                            id="market-interval",
                            interval=300000,  # 5 minutes in milliseconds
                            n_intervals=0
                        )
                    ])
                ], className="mb-4")
            ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("News & Events Analysis"),
                dbc.CardBody([
                    html.Div(id="news-analysis-content"),
                    dcc.Interval(
                        id="news-interval",
                        interval=3600000,  # 1 hour in milliseconds
                        n_intervals=0
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Investment Recommendations"),
                dbc.CardBody([
                    html.Div(id="recommendations-content"),
                    dbc.Button("Generate Recommendations", id="generate-recommendations-button", color="success", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="portfolio-performance-graph"),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Track New Investment", id="track-investment-button", color="info", className="mt-3"),
                                width="auto"
                            ),
                            dbc.Col(
                                dbc.Button("Export Performance Report", id="export-report-button", color="secondary", className="mt-3"),
                                width="auto"
                            )
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ])
    ])
])

# Callbacks
# Callback to load and display tracked assets on application startup
@app.callback(
    Output("tracked-assets-table", "children"),
    Input("add-asset-button", "n_clicks"),
    prevent_initial_call=False
)
def load_assets_table(n_clicks):
    """
    Load and display the tracked assets table when the app starts
    """
    from components.asset_tracker import load_tracked_assets, create_tracked_assets_table
    
    # Load current assets
    assets = load_tracked_assets()
    
    # Create and return the table
    return create_tracked_assets_table(assets)

# Callback to add new asset
@app.callback(
    [Output("add-asset-feedback", "children"),
     Output("tracked-assets-table", "children", allow_duplicate=True),
     Output("asset-symbol-input", "value"),
     Output("asset-name-input", "value")],
    Input("add-asset-button", "n_clicks"),
    [State("asset-symbol-input", "value"),
     State("asset-name-input", "value"),
     State("asset-type-select", "value")],
    prevent_initial_call=True
)
def add_new_asset(n_clicks, symbol, name, asset_type):
    """
    Add a new asset to tracked assets when the Add button is clicked
    """
    from components.asset_tracker import load_tracked_assets, save_tracked_assets, create_tracked_assets_table
    
    if n_clicks is None:
        raise PreventUpdate
    
    if not symbol or not name:
        return dbc.Alert("Symbol and name are required", color="warning"), dash.no_update, dash.no_update, dash.no_update
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    # Load current assets
    assets = load_tracked_assets()
    
    # Check if symbol already exists
    if symbol in assets:
        return dbc.Alert(f"Symbol {symbol} is already being tracked", color="warning"), dash.no_update, dash.no_update, dash.no_update
    
    # Add new asset
    assets[symbol] = {
        "name": name,
        "type": asset_type,
        "added_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Save updated assets
    if save_tracked_assets(assets):
        # Create updated table
        updated_table = create_tracked_assets_table(assets)
        
        # Return success message, updated table, and clear input fields
        return dbc.Alert(f"Successfully added {symbol}", color="success"), updated_table, "", ""
    else:
        return dbc.Alert("Failed to add asset", color="danger"), dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("profile-update-status", "children"),
    Input("update-profile-button", "n_clicks"),
    [State("risk-slider", "value"),
     State("investment-horizon", "value"),
     State("initial-investment", "value")],
    prevent_initial_call=True
)
def update_user_profile(n_clicks, risk_level, investment_horizon, initial_investment):
    """
    Save user profile when the Update Profile button is clicked
    """
    from components.user_profile import save_user_profile
    from datetime import datetime
    
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Create profile object
    profile = {
        "risk_level": risk_level,
        "investment_horizon": investment_horizon,
        "initial_investment": initial_investment,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
# Callback to remove asset
@app.callback(
    Output("tracked-assets-table", "children", allow_duplicate=True),
    Input({"type": "remove-asset-button", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_asset(n_clicks_list):
    """
    Remove an asset when its remove button is clicked
    """
    from components.asset_tracker import load_tracked_assets, save_tracked_assets, create_tracked_assets_table
    
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate
    
    # Find which button was clicked
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the clicked button ID
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        # Extract the symbol from the button ID (it's in JSON format)
        button_id_dict = json.loads(button_id)
        symbol_to_remove = button_id_dict.get("index")
        
        if symbol_to_remove:
            # Load current assets
            assets = load_tracked_assets()
            
            # Remove the asset
            if symbol_to_remove in assets:
                del assets[symbol_to_remove]
                
                # Save updated assets
                save_tracked_assets(assets)
                
                # Return updated table
                return create_tracked_assets_table(assets)
    except Exception as e:
        print(f"Error removing asset: {e}")
    
    # If something went wrong, just refresh the current table
    return create_tracked_assets_table(load_tracked_assets())

@app.callback(
    Output("market-overview-graph", "figure"),
    Input("market-interval", "n_intervals")
)
def update_market_overview(n):
    # Get market data from the data collector
    market_data = data_collector.get_market_data()
    
    # Create a figure using Plotly
    fig = go.Figure()
    
    # Sample data - in the real implementation, this would use actual market data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    values = np.cumsum(np.random.randn(len(dates))) + 1000
    
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='S&P 500'))
    
    fig.update_layout(
        title="Market Trends - Last 30 Days",
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_white"
    )
    
    return fig

@app.callback(
    Output("news-analysis-content", "children"),
    Input("news-interval", "n_intervals")
)
def update_news_analysis(n):
    # Get news and events from the news analyzer
    news_analysis = news_analyzer.analyze_recent_news()
    
    # Sample news - in the real implementation, this would use actual news data
    news_items = [
        {"title": "Fed Raises Interest Rates by 0.25%", "impact": "negative", "relevance": 0.85},
        {"title": "New Tech Regulations Announced", "impact": "neutral", "relevance": 0.65},
        {"title": "Strong Earnings Reports from Tech Sector", "impact": "positive", "relevance": 0.9}
    ]
    
    news_cards = []
    for item in news_items:
        if item["impact"] == "positive":
            color = "success"
        elif item["impact"] == "negative":
            color = "danger"
        else:
            color = "warning"
            
        news_cards.append(
            dbc.Card([
                dbc.CardHeader(f"Relevance: {item['relevance']:.2f}"),
                dbc.CardBody([
                    html.H5(item["title"], className="card-title"),
                    html.P(f"Market Impact: {item['impact'].capitalize()}", className=f"text-{color}")
                ])
            ], className="mb-2")
        )
    
    return news_cards

@app.callback(
    Output("recommendations-content", "children"),
    Input("generate-recommendations-button", "n_clicks"),
    State("risk-slider", "value"),
    State("investment-horizon", "value"),
    State("initial-investment", "value"),
    prevent_initial_call=True
)
def generate_recommendations(n_clicks, risk_level, investment_horizon, initial_investment):
    # Generate recommendations based on user profile and market analysis
    recommendations = recommendation_engine.generate_recommendations(
        risk_level=risk_level,
        investment_horizon=investment_horizon,
        investment_amount=initial_investment
    )
    
    # Sample recommendations - in the real implementation, this would use actual AI recommendations
    sample_recommendations = [
        {"asset_type": "Stocks", "allocation": 0.4, "specific": ["AAPL", "MSFT", "GOOG"], "reasoning": "Strong tech sector performance expected based on recent earnings and innovation trends."},
        {"asset_type": "Bonds", "allocation": 0.3, "specific": ["Treasury", "Corporate"], "reasoning": "Hedge against market volatility and provide stable income in uncertain economic conditions."},
        {"asset_type": "ETFs", "allocation": 0.2, "specific": ["VTI", "QQQ"], "reasoning": "Diversification across market segments provides balanced exposure."},
        {"asset_type": "Commodities", "allocation": 0.1, "specific": ["Gold", "Silver"], "reasoning": "Inflation protection based on recent monetary policy changes."}
    ]
    
    recommendation_cards = []
    for rec in sample_recommendations:
        recommendation_cards.append(
            dbc.Card([
                dbc.CardHeader(f"{rec['asset_type']} - {rec['allocation']*100:.0f}% Allocation"),
                dbc.CardBody([
                    html.H5(", ".join(rec["specific"]), className="card-title"),
                    html.P(rec["reasoning"]),
                    dbc.Button("Add to Portfolio", color="primary", size="sm", className="mt-2")
                ])
            ], className="mb-2")
        )
    
    return recommendation_cards

@app.callback(
    Output("portfolio-performance-graph", "figure"),
    Input("track-investment-button", "n_clicks")
)
def update_portfolio_performance(n_clicks):
    # Get portfolio performance data
    portfolio_data = portfolio_tracker.get_performance()
    
    # Sample portfolio data - in the real implementation, this would track actual investments
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
    portfolio_value = np.cumsum(np.random.randn(len(dates)) * 0.5) + 10000
    benchmark_value = np.cumsum(np.random.randn(len(dates)) * 0.4) + 10000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=portfolio_value, mode='lines', name='Your Portfolio'))
    fig.add_trace(go.Scatter(x=dates, y=benchmark_value, mode='lines', name='Benchmark (S&P 500)', line=dict(dash='dash')))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white"
    )
    
    return fig

if __name__ == "__main__":
    app.run(debug=True)