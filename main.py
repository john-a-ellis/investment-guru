# Investment Recommendation System Architecture
# main.py - Entry point for the application

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from dash.exceptions import PreventUpdate
import json
import yfinance as yf
import pandas as pd
from dash.dependencies import MATCH

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom modules - consolidated imports from our improved architecture
from modules.data_collector import DataCollector
from modules.market_analyzer import MarketAnalyzer
from modules.news_analyzer import NewsAnalyzer
from modules.recommendation_engine import RecommendationEngine
from modules.portfolio_tracker import PortfolioTracker
from modules.mutual_fund_provider import MutualFundProvider

# Import consolidated portfolio utilities
from modules.portfolio_utils import (
    load_portfolio, update_portfolio_data, add_investment, 
    remove_investment, load_tracked_assets, save_tracked_assets, 
    save_user_profile, record_transaction, load_user_profile
)

from components.risk_metrics_component import create_risk_visualization_component
from modules.portfolio_risk_metrics import create_risk_metrics_component
# Components
from components.asset_tracker import create_asset_tracker_component, create_tracked_assets_table
from components.user_profile import create_user_profile_component
from components.mutual_fund_manager import create_mutual_fund_manager_component
from components.portfolio_management import create_portfolio_management_component, create_portfolio_table
from components.portfolio_visualizer import (
    create_portfolio_visualizer_component
)

from components.portfolio_analysis import (
    create_portfolio_analysis_component, create_allocation_chart, create_sector_chart, 
    create_correlation_chart, create_allocation_details, create_sector_details, 
    create_correlation_analysis
)

from components.rebalancing_component import (
    create_rebalancing_component, create_allocation_sliders, create_current_vs_target_chart, 
    create_target_allocation_chart, create_allocation_drift_table, 
    create_rebalance_recommendations, create_rebalance_summary
)
from modules.portfolio_rebalancer import (
    analyze_current_vs_target, load_target_allocation, save_target_allocation, 
    create_rebalance_plan
)


myTitle = 'AIRS - AI Investment Recommendation System'

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # For production deployment

app.title = myTitle
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
                    html.Div([
                        dbc.Label("Time Period:"),
                        dbc.RadioItems(
                            id="market-timeframe-selector",
                            options=[
                                {"label": "1 Week", "value": "1w"},
                                {"label": "1 Month", "value": "1m"},
                                {"label": "3 Months", "value": "3m"},
                                {"label": "6 Months", "value": "6m"},
                                {"label": "1 Year", "value": "1y"},
                                {"label": "5 Years", "value": "5y"}
                            ],
                            value="1m",  # Default to 1 month
                            inline=True,
                            className="mt-2"
                        )
                    ]),
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
    
    # Mutual Fund Manager Component
    dbc.Row([
        dbc.Col([
            create_mutual_fund_manager_component()
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            create_portfolio_analysis_component()
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            create_risk_visualization_component()
        ], width=12)

    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            create_rebalancing_component()
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
    
    # Portfolio Management Component
    dbc.Row([
        dbc.Col([
            create_portfolio_management_component()
        ], width=12)
    ], className="mb-4"),

    # Portfolio Visualizer Component
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
    # Load current assets
    assets = load_tracked_assets()
    
    # Create and return the table
    return create_tracked_assets_table(assets)

# Callback to add new asset
@app.callback(
    Output("add-asset-feedback", "children"),
    Output("tracked-assets-table", "children", allow_duplicate=True),
    Output("asset-symbol-input", "value"),
    Output("asset-name-input", "value"),
    Input("add-asset-button", "n_clicks"),
    State("asset-symbol-input", "value"),
    State("asset-name-input", "value"),
    State("asset-type-select", "value"),
    prevent_initial_call=True
)
def add_new_asset(n_clicks, symbol, name, asset_type):
    """
    Add a new asset to tracked assets when the Add button is clicked
    """
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
    State("risk-slider", "value"),
    State("investment-horizon", "value"),
    State("initial-investment", "value"),
    prevent_initial_call=True
)
def update_user_profile(n_clicks, risk_level, investment_horizon, initial_investment):
    """
    Save user profile when the Update Profile button is clicked
    """
    if n_clicks is None:
        raise PreventUpdate
    
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
    Input({"type": "remove-asset-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_asset(n_clicks_list):
    """
    Remove an asset when its remove button is clicked
    """
    # Check if callback context has been triggered
    if not ctx.triggered:
        raise PreventUpdate
    
    # Check if any buttons have been clicked
    if not n_clicks_list or all(n is None for n in n_clicks_list):
        raise PreventUpdate
    
    # Find which button was clicked - only consider those with n_clicks > 0
    clicked_index = None
    for i, n_clicks in enumerate(n_clicks_list):
        if n_clicks and n_clicks > 0:  # Check for non-None and positive click count
            # Get the ID of the trigger
            trigger_id = ctx.triggered_id
            if trigger_id and 'index' in trigger_id:
                clicked_index = trigger_id['index']
                break
    
    # If no valid button was identified, prevent update
    if not clicked_index:
        raise PreventUpdate
    
    print(f"Removing asset with symbol: {clicked_index}")
    
    # Load current assets
    assets = load_tracked_assets()
    
    # Remove the asset if it exists
    if clicked_index in assets:
        del assets[clicked_index]
        save_tracked_assets(assets)
    
    # Return updated table
    return create_tracked_assets_table(assets)

@app.callback(
    Output("market-overview-graph", "figure"),
    [Input("market-interval", "n_intervals"),
     Input("market-timeframe-selector", "value")]
)
def update_market_overview(n, timeframe):
    """
    Update the market overview graph with actual historical data for tracked assets
    using FMP API instead of yfinance
    """
    # Load tracked assets
    tracked_assets = load_tracked_assets()
    
    if not tracked_assets:
        # Return empty figure if no assets are tracked
        fig = go.Figure()
        fig.update_layout(
            title="Market Overview - No Assets Tracked",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        return fig
    
    # Import FMP API
    from modules.fmp_api import fmp_api
    
    # Define time period based on selected timeframe
    end_date = datetime.now()
    
    if timeframe == "1w":
        start_date = end_date - timedelta(days=7)
        period_label = "Last Week"
    elif timeframe == "1m":
        start_date = end_date - timedelta(days=30)
        period_label = "Last Month"
    elif timeframe == "3m":
        start_date = end_date - timedelta(days=90)
        period_label = "Last 3 Months"
    elif timeframe == "6m":
        start_date = end_date - timedelta(days=180)
        period_label = "Last 6 Months"
    elif timeframe == "1y":
        start_date = end_date - timedelta(days=365)
        period_label = "Last Year"
    elif timeframe == "5y":
        start_date = end_date - timedelta(days=365*5)
        period_label = "Last 5 Years"
    else:
        # Default to 1 month
        start_date = end_date - timedelta(days=30)
        period_label = "Last Month"
    
    # Create a figure
    fig = go.Figure()
    
    # For longer timeframes, use a more appropriate interval
    data_interval = "1d"  # Default daily for shorter timeframes
    if timeframe in ["1y", "5y"]:
        data_interval = "1wk"  # Use weekly for 1y and 5y
    
    # Process data for each asset
    for symbol, details in tracked_assets.items():
        try:
            # Get historical price data from FMP API
            hist = fmp_api.get_historical_price(symbol, start_date=start_date, end_date=end_date)
            
            if not hist.empty and 'Close' in hist.columns:
                # Normalize values to start at 100 for better comparison
                first_valid_close = hist['Close'].dropna().iloc[0] if not hist['Close'].dropna().empty else 1
                if first_valid_close > 0:  # Avoid division by zero
                    normalized = hist['Close'] / first_valid_close * 100
                    
                    # Add to chart
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode='lines',
                        name=f"{symbol} - {details.get('name', '')}"
                    ))
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add a benchmark index (S&P/TSX Composite for Canadian focus)
    try:
        tsx_hist = fmp_api.get_historical_price("^GSPTSE", start_date=start_date, end_date=end_date)
        
        if not tsx_hist.empty and 'Close' in tsx_hist.columns:
            # Normalize TSX values
            first_valid_tsx = tsx_hist['Close'].dropna().iloc[0] if not tsx_hist['Close'].dropna().empty else 1
            if first_valid_tsx > 0:  # Avoid division by zero
                tsx_normalized = tsx_hist['Close'] / first_valid_tsx * 100
                
                # Add to chart with dashed line
                fig.add_trace(go.Scatter(
                    x=tsx_hist.index,
                    y=tsx_normalized,
                    mode='lines',
                    name="S&P/TSX Composite",
                    line=dict(dash='dash', width=3)
                ))
    except Exception as e:
        print(f"Error getting TSX data: {e}")
        import traceback
        traceback.print_exc()
    
    # Set appropriate tick spacing based on timeframe
    if timeframe == "1w":
        tick_spacing = 1  # Daily for 1 week
    elif timeframe == "1m":
        tick_spacing = 7  # Weekly for 1 month
    elif timeframe == "3m":
        tick_spacing = 14  # Bi-weekly for 3 months
    elif timeframe == "6m":
        tick_spacing = 30  # Monthly for 6 months
    elif timeframe == "1y":
        tick_spacing = 60  # Bi-monthly for 1 year
    elif timeframe == "5y":
        tick_spacing = 180  # Every 6 months for 5 years
    else:
        tick_spacing = 7  # Default to weekly
    
    # Update layout with explicit tick configuration
    fig.update_layout(
        title=f"Market Performance - {period_label} (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Value (Normalized)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )
    
    # Apply more explicit tick formatting
    fig.update_xaxes(
        tickmode='auto',
        nticks=int((end_date - start_date).days / tick_spacing),  # Calculate appropriate number of ticks
        tickformat='%Y-%m-%d'  # Ensure date format is consistent
    )
    
    return fig

@app.callback(
    Output("news-analysis-content", "children"),
    Input("news-interval", "n_intervals")
)
def update_news_analysis(n):
    """
    Update news analysis with relevant stories about tracked assets and portfolio holdings
    """
    # Load tracked assets and portfolio
    tracked_assets = load_tracked_assets()
    portfolio = load_portfolio()
    
    if not tracked_assets and not portfolio:
        return [
            dbc.Alert(
                "Add assets to your tracking list or portfolio to see relevant news.",
                color="info"
            )
        ]
    
    # Collect all symbols from both tracked assets and portfolio
    all_symbols = set(tracked_assets.keys()) | set(inv.get("symbol", "") for inv in portfolio.values())
    
    # Create search terms for companies
    search_terms = []
    company_names = {}
    
    for symbol in all_symbols:
        # Add the symbol itself
        search_terms.append(symbol)
        
        # Add the company name if available
        if symbol in tracked_assets:
            name = tracked_assets[symbol].get("name", "")
            if name:
                # Extract the main company name without suffixes
                main_name = name.split(" ")[0]
                if len(main_name) > 3:  # Avoid short names that might cause too many false matches
                    search_terms.append(main_name)
                    company_names[symbol] = name
    
    # Default news items if API call fails
    fallback_news = [
        {"title": "Market Update: TSX Gains on Energy Rally", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source"}},
        {"title": "Bank of Canada Holds Interest Rates Steady", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source"}},
        {"title": "Tech Stocks Lead Market Recovery", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source"}}
    ]
    
    # Since we don't have a functional news API here, we'll use the fallback news
    news_articles = fallback_news
    
    # Find relevant symbols in each article
    for article in news_articles:
        article["relevant_symbols"] = []
        article_title = article.get("title", "") or ""
        article_description = article.get("description", "") or ""
        
        for symbol in all_symbols:
            if symbol in article_title or symbol in article_description:
                article["relevant_symbols"].append(symbol)
            # Check company name
            elif symbol in company_names and company_names[symbol] in article_title:
                article["relevant_symbols"].append(symbol)
    
    # Create news cards
    news_cards = []
    for article in news_articles:
        # Create tags for relevant symbols
        symbol_badges = []
        for symbol in article.get("relevant_symbols", []):
            symbol_badges.append(
                dbc.Badge(symbol, color="info", className="me-1")
            )
        
        # Format date
        published_date = article.get("publishedAt", "")
        if published_date:
            try:
                date_obj = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ")
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = published_date
        else:
            formatted_date = "Unknown date"
        
        # Get source name safely
        source_name = article.get("source", {})
        if isinstance(source_name, dict):
            source_name = source_name.get("name", "Unknown source")
        
        # Get description safely and truncate if needed
        description = article.get("description", "No description") or "No description"
        if len(description) > 200:
            description = description[:200] + "..."
        
        # Create card
        news_cards.append(
            dbc.Card([
                dbc.CardHeader([
                    html.Div(symbol_badges, className="mb-1") if symbol_badges else "",
                    html.Small(f"{source_name} - {formatted_date}", className="text-muted")
                ]),
                dbc.CardBody([
                    html.H5(article.get("title", "No title"), className="card-title"),
                    html.P(description),
                    html.A("Read more", href=article.get("url", "#"), target="_blank", className="btn btn-sm btn-outline-primary")
                ])
            ], className="mb-3")
        )
    
    return news_cards

@app.callback(
    Output("recommendations-content", "children"),
    Input("generate-recommendations-button", "n_clicks"),
    [State("risk-slider", "value"),
     State("investment-horizon", "value"),
     State("initial-investment", "value")],
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

# Mutual Fund Manager Callbacks
@app.callback(
    Output("add-fund-price-feedback", "children"),
    Input("add-fund-price-button", "n_clicks"),
    [State("fund-code-input", "value"),
     State("fund-date-input", "value"),
     State("fund-price-input", "value")],
    prevent_initial_call=True
)
def add_fund_price(n_clicks, fund_code, date, price):
    """
    Add a price point for a mutual fund
    """
    if not n_clicks:
        raise PreventUpdate
    
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
    """
    View price history for a mutual fund
    """
    if not n_clicks or not fund_code:
        raise PreventUpdate
    
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

# Consolidated Portfolio Management Callback
@app.callback(
    [Output("add-investment-feedback", "children"),
     Output("portfolio-table", "children"),
     Output("investment-symbol-input", "value"),
     Output("investment-shares-input", "value"),
     Output("investment-price-input", "value")],
    [Input("add-investment-button", "n_clicks"),
     Input("portfolio-update-interval", "n_intervals"),
     Input({"type": "remove-investment-button", "index": ALL}, "n_clicks")],
    [State("investment-symbol-input", "value"),
     State("investment-shares-input", "value"),
     State("investment-price-input", "value"),
     State("investment-date-input", "value"),
     State("investment-type-select", "value")],
    prevent_initial_call=True
)
def manage_portfolio(add_clicks, update_interval, remove_clicks, symbol, shares, price, date, asset_type):
    """
    Consolidated callback to manage portfolio actions (add, update, remove)
    """
    # Initialize default return values
    feedback = None
    updated_table = dash.no_update
    clear_symbol = dash.no_update
    clear_shares = dash.no_update
    clear_price = dash.no_update
    
    # Get the triggered component
    if not ctx.triggered:
        return feedback, updated_table, clear_symbol, clear_shares, clear_price
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle add investment
    if trigger_id == "add-investment-button" and add_clicks:
        if not symbol or not shares or not price:
            feedback = dbc.Alert("Symbol, shares, and price are required", color="warning")
        else:
            # Standardize symbol format
            symbol_upper = symbol.upper().strip()
            
            # Add investment with asset type
            success = add_investment(symbol_upper, shares, price, date, asset_type)
            
            if success:
                feedback = dbc.Alert(f"Successfully added {symbol_upper}", color="success")
                clear_symbol = ""
                clear_shares = None
                clear_price = None
            else:
                feedback = dbc.Alert("Failed to add investment", color="danger")
    
    # Handle remove investment
    elif isinstance(trigger_id, str) and "{" in trigger_id:  # Pattern matching ID
        try:
            button_id = json.loads(trigger_id)
            if button_id.get("type") == "remove-investment-button":
                investment_id = button_id.get("index")
                if investment_id and any(remove_clicks):
                    # The investment_id here is already a string from the JSON parse
                    remove_investment(investment_id)
                    feedback = dbc.Alert(f"Investment removed", color="info")
        except Exception as e:
            print(f"Error processing remove action: {e}")
            feedback = dbc.Alert(f"Error removing investment: {str(e)}", color="danger")
    
    # Always update the table regardless of action
    portfolio = update_portfolio_data()
    updated_table = create_portfolio_table(portfolio)
    
    return feedback, updated_table, clear_symbol, clear_shares, clear_price

@app.callback(
    Output("portfolio-performance-graph", "figure"),
    [Input("portfolio-update-interval", "n_intervals"),
     Input("performance-period-selector", "value"),
     Input("performance-chart-type", "value"),
     Input("performance-calculation-method", "value")]
)
def update_portfolio_graph(n_intervals, period, chart_type, calculation_method):
    """
    Update the portfolio performance graph based on the selected time period,
    chart type, and calculation method with improved tick spacing
    """
    
    # Load portfolio data
    try:
        portfolio = load_portfolio()
        
        if not portfolio:
            # Return empty figure if no portfolio data
            fig = go.Figure()
            fig.update_layout(
                title="Portfolio Performance (No portfolio data available)",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                template="plotly_white"
            )
            return fig
        
        # Use appropriate calculation method
        if calculation_method == "twrr":
            # Time-Weighted Rate of Return
            from components.portfolio_visualizer import create_twrr_performance_graph
            fig = create_twrr_performance_graph(portfolio, period)
            
        elif calculation_method == "mwr":
            # Money-Weighted Return / IRR
            # For MWR, we'll show a summary with the IRR percentage,
            # but use the TWRR chart visualization
            from components.portfolio_visualizer import create_twrr_performance_graph
            fig = create_twrr_performance_graph(portfolio, period)
        
        else:  # "simple"
            # Use the original chart functions based on chart_type
            if chart_type == "normalized":
                # Use the normalized view where all assets start at 100
                from components.portfolio_visualizer import create_normalized_performance_graph
                fig = create_normalized_performance_graph(portfolio, period)
            elif chart_type == "relative":
                # Use the relative percentage change view
                from components.portfolio_visualizer import create_adaptive_scale_graph
                fig = create_adaptive_scale_graph(portfolio, period, relative_view=True)
            else:  # "value" (actual value)
                # Use the absolute value view
                from components.portfolio_visualizer import create_performance_graph
                fig = create_performance_graph(portfolio, period)
        
        # Define appropriate tick spacing based on period
        # Define date range based on period
        end_date = datetime.now()
        
        if period == "1m":
            start_date = end_date - timedelta(days=30)
            tick_spacing = 5  # Show a tick every 5 days for 1 month
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
            tick_spacing = 14  # Show a tick every 2 weeks for 3 months
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
            tick_spacing = 30  # Show a tick every month for 6 months
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
            tick_spacing = 60  # Show a tick every 2 months for 1 year
        else:  # "all"
            # Find earliest transaction date from the database
            earliest_date_query = """
            SELECT MIN(transaction_date) as earliest_date FROM transactions;
            """
            earliest_result = execute_query(earliest_date_query, fetchone=True)
            
            if earliest_result and earliest_result['earliest_date']:
                start_date = earliest_result['earliest_date']
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
            # Calculate days between start and end
            days_range = (end_date - start_date).days
            if days_range <= 365:
                tick_spacing = 60  # Every 2 months if less than a year
            else:
                tick_spacing = 90  # Every 3 months if over a year
        
        # Apply improved tick spacing to the figure
        fig.update_xaxes(
            tickmode='auto',
            nticks=max(5, int((end_date - start_date).days / tick_spacing)),  # Ensure at least 5 ticks
            tickformat='%b %d, %Y'  # More readable date format
        )
        
        return fig
            
    except Exception as e:
        print(f"Error in update_portfolio_graph: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return error figure
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig

# Update the summary stats callback to show IRR details when appropriate
@app.callback(
    Output("portfolio-summary-stats", "children"),
    [Input("portfolio-update-interval", "n_intervals"),
     Input("performance-period-selector", "value"),
     Input("performance-calculation-method", "value")]
)
def update_portfolio_stats(n_intervals, period, calculation_method):
    """
    Update the portfolio summary statistics based on the selected calculation method
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # If using money-weighted return (IRR) method, show the MWR summary
    if calculation_method == "mwr":
        from components.portfolio_visualizer import create_mwr_summary
        return create_mwr_summary(portfolio, period)
    else:
        # For TWRR or simple methods, show the standard summary stats
        from components.portfolio_visualizer import create_summary_stats
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

# Add this callback to main.py to ensure the table loads initially
@app.callback(
    Output("portfolio-table", "children", allow_duplicate=True),
    Input("portfolio-update-interval", "n_intervals"),
    prevent_initial_call='initial_duplicate' # This is important - allow initial call
)
def load_initial_portfolio_table(n_intervals):
    """
    Load and display the portfolio table when the app first loads
    """
    print("Initial portfolio table load triggered")
    try:
        # Update portfolio data with current market prices
        portfolio = update_portfolio_data()
        
        # Create and return the table
        return create_portfolio_table(portfolio)
    except Exception as e:
        print(f"Error loading initial portfolio: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading portfolio: {str(e)}")

@app.callback(
    Output("quick-transaction-feedback", "children"),
    Output("portfolio-table", "children", allow_duplicate=True),
    Input("record-quick-transaction", "n_clicks"),
    [State("quick-transaction-type", "value"),
     State("quick-transaction-symbol", "value"),
     State("quick-transaction-shares", "value"),
     State("quick-transaction-price", "value"),
     State("quick-transaction-date", "value")],
    prevent_initial_call=True
)
def record_quick_transaction(n_clicks, transaction_type, symbol, shares, price, date):
    if n_clicks is None or not symbol or shares is None or price is None:
        raise dash.exceptions.PreventUpdate
    
    # Standardize values
    symbol = symbol.upper().strip()
    
    # Record the transaction
    success = record_transaction(transaction_type, symbol, price, shares, date)
    
    # Update portfolio table
    portfolio = update_portfolio_data()
    updated_table = create_portfolio_table(portfolio)
    
    # Return appropriate feedback
    if success:
        action = "purchase" if transaction_type == "buy" else "sale"
        return (
            dbc.Alert(f"Successfully recorded {action} of {shares} shares of {symbol}", color="success"),
            updated_table
        )
    else:
        return (
            dbc.Alert(f"Failed to record transaction. Please check your inputs.", color="danger"),
            dash.no_update
        )

# Fixed pattern-matching callback for buy transactions from accordion
@app.callback(
    Output({"type": "buy-feedback", "symbol": dash.dependencies.MATCH}, "children"),
    Input({"type": "record-buy-button", "symbol": dash.dependencies.MATCH}, "n_clicks"),
    [State({"type": "buy-shares-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "buy-price-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "buy-date-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "record-buy-button", "symbol": dash.dependencies.MATCH}, "id")],
    prevent_initial_call=True
)
def record_buy_feedback(n_clicks, shares, price, date, button_id):
    if n_clicks is None or shares is None or price is None:
        raise dash.exceptions.PreventUpdate
    
    # Get the symbol from the button ID
    symbol = button_id["symbol"]
    
    # Record the transaction
    success = record_transaction("buy", symbol, price, shares, date)
    
    # Return appropriate feedback
    if success:
        return dbc.Alert(f"Successfully recorded purchase of {shares} shares of {symbol}", color="success")
    else:
        return dbc.Alert(f"Failed to record transaction. Please check your inputs.", color="danger")

# Update portfolio table after buy transaction
@app.callback(
    Output("portfolio-table", "children", allow_duplicate=True),
    Input({"type": "record-buy-button", "symbol": dash.dependencies.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_table_after_buy(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Update portfolio table
    portfolio = update_portfolio_data()
    return create_portfolio_table(portfolio)

# Fixed pattern-matching callback for sell transactions from accordion
@app.callback(
    Output({"type": "sell-feedback", "symbol": dash.dependencies.MATCH}, "children"),
    Input({"type": "record-sell-button", "symbol": dash.dependencies.MATCH}, "n_clicks"),
    [State({"type": "sell-shares-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "sell-price-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "sell-date-input", "symbol": dash.dependencies.MATCH}, "value"),
     State({"type": "record-sell-button", "symbol": dash.dependencies.MATCH}, "id")],
    prevent_initial_call=True
)
def record_sell_feedback(n_clicks, shares, price, date, button_id):
    if n_clicks is None or shares is None or price is None:
        raise dash.exceptions.PreventUpdate
    
    # Get the symbol from the button ID
    symbol = button_id["symbol"]
    
    # Record the transaction
    success = record_transaction("sell", symbol, price, shares, date)
    
    # Return appropriate feedback
    if success:
        return dbc.Alert(f"Successfully recorded sale of {shares} shares of {symbol}", color="success")
    else:
        return dbc.Alert(f"Failed to record sale. Please check that you have enough shares to sell.", color="danger")

# Update portfolio table after sell transaction
@app.callback(
    Output("portfolio-table", "children", allow_duplicate=True),
    Input({"type": "record-sell-button", "symbol": dash.dependencies.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_table_after_sell(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Update portfolio table
    portfolio = update_portfolio_data()
    return create_portfolio_table(portfolio)

@app.callback(
    Output("risk-metrics-content", "children"),
    [Input("risk-update-interval", "n_intervals"),
     Input("risk-period-selector", "value")]
)
def update_risk_metrics(n_intervals, period):
    """
    Update the risk metrics visualization based on the selected time period
    """
    # Load portfolio data
    portfolio = update_portfolio_data()
    
    # Create and return the risk metrics component
    return create_risk_metrics_component(portfolio, period)
@app.callback(
    Output("current-vs-target-chart", "figure"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_allocation_chart(n_intervals, active_tab):
    """
    Update the current vs target allocation chart
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Analyze current vs target allocation
    analysis = analyze_current_vs_target(portfolio)
    
    # Create and return the chart
    return create_current_vs_target_chart(analysis)

# Callback to update allocation drift table
@app.callback(
    Output("allocation-drift-table", "children"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_allocation_drift_table(n_intervals, active_tab):
    """
    Update the allocation drift table
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Analyze current vs target allocation
    analysis = analyze_current_vs_target(portfolio)
    
    # Create and return the table
    return create_allocation_drift_table(analysis)

# Callback to update rebalance summary
@app.callback(
    Output("rebalance-summary", "children"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_rebalance_summary(n_intervals, active_tab):
    """
    Update the rebalance summary
    """
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Analyze current vs target allocation
    analysis = analyze_current_vs_target(portfolio)
    
    # Create and return the summary
    return create_rebalance_summary(analysis)

# Callback to update rebalance recommendations
@app.callback(
    Output("rebalance-recommendations", "children"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_rebalance_recommendations(n_intervals, active_tab):
    """
    Update the rebalance recommendations
    """
    # Only update when on the Rebalancing Plan tab
    if active_tab == "rebalancing-plan-tab":
        # Load portfolio data
        portfolio = load_portfolio()
        
        # Get user risk level from profile
        profile = load_user_profile()
        risk_level = profile.get("risk_level", 5)
        
        # Create rebalance plan
        rebalance_plan = create_rebalance_plan(portfolio, risk_level)
        
        # Create and return the recommendations
        return create_rebalance_recommendations(rebalance_plan)
    
    # Return empty div if not on the right tab
    return html.Div()

# Callback to initialize target allocation sliders
# Callback to initialize target allocation sliders
@app.callback(
    Output("target-allocation-sliders", "children"),
    [Input("rebalancing-tabs", "active_tab")]
)
def initialize_target_sliders(active_tab):
    """
    Initialize the target allocation sliders with current values
    """
    # Only update when on the Target Settings tab
    if active_tab == "target-settings-tab":  # Using the new tab ID instead of "tab-2"
        # Load current target allocation
        target_allocation = load_target_allocation()
        
        # Create and return the sliders
        return create_allocation_sliders(target_allocation)
    
    # Return empty div if not on the right tab
    return html.Div()

# Callback to update target allocation chart based on slider values
@app.callback(
    [Output("target-allocation-chart", "figure"),
     Output("slider-total-warning", "children")],
    [Input({"type": "target-slider", "asset_type": ALL}, "value")]
)
def update_target_chart(slider_values):
    """
    Update the target allocation chart based on slider values
    """
    # Get current target allocation to match sliders with asset types
    target_allocation = load_target_allocation()
    asset_types = list(target_allocation.keys())
    
    # Create new target allocation based on slider values
    new_target = {}
    for i, asset_type in enumerate(asset_types):
        if i < len(slider_values):
            new_target[asset_type] = slider_values[i]
    
    # Calculate total allocation
    total_allocation = sum(new_target.values())
    
    # Create warning if total is not 100%
    if abs(total_allocation - 100) > 1:  # Allow for small rounding differences
        warning = dbc.Alert(
            f"Warning: Total allocation is {total_allocation:.1f}%. Target should sum to 100%.",
            color="warning"
        )
    else:
        warning = html.Div()
    
    # Create and return the chart
    return create_target_allocation_chart(new_target), warning

# Callback to update slider output text
@app.callback(
    Output({"type": "slider-output", "asset_type": MATCH}, "children"),
    Input({"type": "target-slider", "asset_type": MATCH}, "value")
)
def update_slider_output(value):
    """
    Update the output text for a slider
    """
    return f"{value}%"

# Callback to save target allocation
@app.callback(
    Output("target-allocation-feedback", "children"),
    Input("save-target-allocation", "n_clicks"),
    State({"type": "target-slider", "asset_type": ALL}, "id"),
    State({"type": "target-slider", "asset_type": ALL}, "value"),
    prevent_initial_call=True
)
def save_target_settings(n_clicks, slider_ids, slider_values):
    """
    Save the target allocation settings
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Create new target allocation from slider values
    new_target = {}
    for i, slider_id in enumerate(slider_ids):
        asset_type = slider_id["asset_type"]
        if i < len(slider_values):
            new_target[asset_type] = slider_values[i]
    
    # Check if total allocation is 100%
    total_allocation = sum(new_target.values())
    if abs(total_allocation - 100) > 1:  # Allow for small rounding differences
        return dbc.Alert(
            f"Error: Total allocation must sum to 100%. Current total: {total_allocation:.1f}%",
            color="danger"
        )
    
    # Save target allocation
    if save_target_allocation(new_target):
        return dbc.Alert(
            "Target allocation saved successfully.",
            color="success"
        )
    else:
        return dbc.Alert(
            "Error saving target allocation.",
            color="danger"
        )
    
if __name__ == "__main__":
    app.run(debug=True)