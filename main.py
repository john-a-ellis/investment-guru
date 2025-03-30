# Investment Recommendation System Architecture
# main.py - Entry point for the application

import dash
from dash import dcc, html, callback, Input, Output, State, ctx
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
from modules.currency_utils import get_usd_to_cad_rate, format_currency, get_combined_value_cad

from modules.data_collector import DataCollector
from modules.market_analyzer import MarketAnalyzer
from modules.news_analyzer import NewsAnalyzer
from modules.recommendation_engine import RecommendationEngine
from modules.portfolio_tracker import PortfolioTracker
from components.asset_tracker import create_asset_tracker_component, load_tracked_assets, create_tracked_assets_table
from components.user_profile import create_user_profile_component

# New imports for portfolio tracking
from components.portfolio_management import create_portfolio_management_component
from components.portfolio_visualizer import create_portfolio_visualizer_component, create_performance_graph, create_summary_stats
from modules.portfolio_data_updater import update_portfolio_data, add_investment, remove_investment, load_portfolio
from components.portfolio_analysis import (
    create_portfolio_analysis_component, 
    create_allocation_chart,
    create_sector_chart, 
    create_correlation_chart,
    create_allocation_details,
    create_sector_details,
    create_correlation_analysis
)

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
        ], width=8),
        dbc.Col([
            html.Img(src="assets/NearNorthClean.png", height="100px", className="float-end")
        ], width=4)
    ], className="mb-4"),

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
    ], className="mb-4"),
    
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
            create_portfolio_analysis_component()
        ], width=12)
    ], className="mb-4"),
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
    
    # Portfolio Management Component - NEW
    dbc.Row([
        dbc.Col([
            create_portfolio_management_component()
        ], width=12)
    ], className="mb-4"),

    # Portfolio Visualizer Component - NEW
    dbc.Row([
        dbc.Col([
            create_portfolio_visualizer_component()
        ], width=12)
    ], className="mb-4"),
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
    
    # Save profile
    if save_user_profile(profile):
        return dbc.Alert("Profile updated successfully", color="success")
    else:
        return dbc.Alert("Error updating profile", color="danger")

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

# NEW CALLBACKS FOR PORTFOLIO TRACKING

@app.callback(
    Output("portfolio-table", "children"),
    [Input("add-investment-button", "n_clicks"),
     Input("portfolio-update-interval", "n_intervals")],
    prevent_initial_call=False
)
def load_portfolio_table(n_clicks, n_intervals):
    """
    Load and display the portfolio table when the app starts or updates
    """
    # Update portfolio data with current market prices
    portfolio = update_portfolio_data()
    
    # Create and return the table
    from components.portfolio_management import create_portfolio_table
    return create_portfolio_table(portfolio)

@app.callback(
    [Output("add-investment-feedback", "children"),
     Output("portfolio-table", "children", allow_duplicate=True),
     Output("investment-symbol-input", "value"),
     Output("investment-shares-input", "value"),
     Output("investment-price-input", "value")],
    Input("add-investment-button", "n_clicks"),
    [State("investment-symbol-input", "value"),
     State("investment-shares-input", "value"),
     State("investment-price-input", "value"),
     State("investment-date-input", "value")],
    prevent_initial_call=True
)
def add_new_investment(n_clicks, symbol, shares, price, date):
    """
    Add a new investment to the portfolio when the Add button is clicked
    """
    from components.portfolio_management import create_portfolio_table
    
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    if not symbol or not shares or not price:
        return dbc.Alert("Symbol, shares, and price are required", color="warning"), dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Standardize symbol format
    symbol = symbol.upper().strip()
    
    # Add investment
    success = add_investment(symbol, shares, price, date)
    
    if success:
        # Update portfolio
        portfolio = update_portfolio_data()
        
        # Create updated table
        updated_table = create_portfolio_table(portfolio)
        
        # Return success message, updated table, and clear input fields
        return dbc.Alert(f"Successfully added {symbol}", color="success"), updated_table, "", None, None
    else:
        return dbc.Alert("Failed to add investment", color="danger"), dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("portfolio-table", "children", allow_duplicate=True),
    Input({"type": "remove-investment-button", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_portfolio_investment(n_clicks_list):
    """
    Remove an investment when its remove button is clicked
    """
    from components.portfolio_management import create_portfolio_table
    
    if not n_clicks_list or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    
    # Find which button was clicked
    if not dash.callback_context.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Get the clicked button ID
    button_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    
    try:
        # Extract the investment ID from the button ID (it's in JSON format)
        button_id_dict = json.loads(button_id)
        investment_id = button_id_dict.get("index")
        
        if investment_id:
            # Remove the investment
            remove_investment(investment_id)
            
            # Load updated portfolio
            portfolio = load_portfolio()
            
            # Return updated table
            return create_portfolio_table(portfolio)
    except Exception as e:
        print(f"Error removing investment: {e}")
    
    # If something went wrong, just refresh the current table
    return create_portfolio_table(load_portfolio())

@app.callback(
    Output("portfolio-performance-graph", "figure"),
    [Input("portfolio-update-interval", "n_intervals"),
     Input("performance-period-selector", "value")]
)
def update_portfolio_graph(n_intervals, period):
    """
    Update the portfolio performance graph
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return graph
    return create_performance_graph(portfolio, period)

@app.callback(
    Output("portfolio-summary-stats", "children"),
    Input("portfolio-update-interval", "n_intervals")
)
def update_portfolio_stats(n_intervals):
    """
    Update the portfolio summary statistics
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return summary stats
    return create_summary_stats(portfolio)
@app.callback(
    Output("portfolio-allocation-chart", "figure"),
    Input("analysis-update-interval", "n_intervals")
)
def update_allocation_chart(n):
    """
    Update the portfolio allocation chart
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return chart
    return create_allocation_chart(portfolio)

@app.callback(
    Output("portfolio-sector-chart", "figure"),
    Input("analysis-update-interval", "n_intervals")
)
def update_sector_chart(n):
    """
    Update the portfolio sector breakdown chart
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return chart
    return create_sector_chart(portfolio)

@app.callback(
    Output("portfolio-correlation-chart", "figure"),
    Input("analysis-update-interval", "n_intervals")
)
def update_correlation_chart(n):
    """
    Update the portfolio correlation chart
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return chart
    return create_correlation_chart(portfolio)

@app.callback(
    Output("allocation-details", "children"),
    Input("analysis-update-interval", "n_intervals")
)
def update_allocation_details(n):
    """
    Update the allocation details
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return details
    return create_allocation_details(portfolio)

@app.callback(
    Output("sector-details", "children"),
    Input("analysis-update-interval", "n_intervals")
)
def update_sector_details(n):
    """
    Update the sector details
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return details
    return create_sector_details(portfolio)

@app.callback(
    Output("correlation-analysis", "children"),
    Input("analysis-update-interval", "n_intervals")
)
def update_correlation_analysis(n):
    """
    Update the correlation analysis
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return analysis
    return create_correlation_analysis(portfolio)

if __name__ == "__main__":
    app.run(debug=True)