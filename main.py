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
import multiprocessing

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom modules - consolidated imports from our improved architecture
from components.currency_exchange_component import create_currency_exchange_component
from modules.help_system import get_help_content_html, get_available_topics
from components.help_display import create_help_display_component
from components.cash_management import create_cash_management_component, create_cash_flow_table
from modules.market_analyzer import MarketAnalyzer
from modules.news_analyzer import NewsAnalyzer
from modules.recommendation_engine import RecommendationEngine
# from modules.portfolio_tracker import PortfolioTracker
from modules.mutual_fund_provider import MutualFundProvider
from modules.data_provider import data_provider
# Import consolidated portfolio utilities
from modules.portfolio_utils import (
    load_portfolio, update_portfolio_data, add_investment,
    remove_investment, load_tracked_assets, save_tracked_assets,
    save_user_profile, record_transaction, load_user_profile,
    get_earliest_transaction_date, load_transactions, # Added load_transactions if needed elsewhere
    load_target_allocation, save_target_allocation, # Moved from rebalancer
    analyze_current_vs_target # Moved from rebalancer
)
from modules.portfolio_tracker import PortfolioTracker
from modules.portfolio_rebalancer import create_rebalance_plan
from components.risk_metrics_component import create_risk_visualization_component
from modules.portfolio_risk_metrics import create_risk_metrics_component
from components.asset_tracker import create_asset_tracker_component, create_tracked_assets_table
from components.user_profile import create_user_profile_component
from components.mutual_fund_manager import create_mutual_fund_manager_component
from components.portfolio_management import create_portfolio_management_component, create_portfolio_table
from components.portfolio_visualizer import create_portfolio_visualizer_component
from components.portfolio_summary import create_portfolio_summary_component
from components.portfolio_analysis import (
    create_portfolio_analysis_component, create_allocation_chart, create_sector_chart,
    create_correlation_chart, create_allocation_details, create_sector_details,
    create_correlation_analysis
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
from components.ml_prediction_component import create_ml_prediction_component, register_ml_prediction_callbacks
    # Import the help component creation function
    # Import the help component creation function


myTitle = 'AIRS - AI Investment Recommendation System'

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CERULEAN],
                suppress_callback_exceptions=True)

server = app.server  # For production deployment

app.title = myTitle
# Initialize system components
# data_collector = DataCollector()
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
        ], width=10),
        dbc.Col(dbc.Button("Help", id="open-help-button", color="info", className="float-end bi bi-question-circle-fill"), width=1),
        dbc.Col([
            html.Img(src="assets/NearNorthClean.png", height="100px", className="float-end")
        ], width=1)
    ], className="mb-4"),

    # Portfolio Summary Component
    dbc.Row([
        dbc.Col([
            html.Div(id="portfolio-summary", children=create_portfolio_summary_component())
        ], width=11),
        dbc.Col([
        dbc.Button(
            html.I(className="bi bi-arrow-clockwise"), 
            id="refresh-portfolio-summary",
            color="light",
            size="sm",
            className="mt-2"
        )
        ], width=1)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            create_cash_management_component()
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            create_currency_exchange_component()
        ], width=12)
    ], className="mb-4"), 

    dbc.Row([
            dbc.Col([
                create_risk_visualization_component()
            ], width=12)

        ], className="mb-4"),

    # Portfolio Analysis Component
    dbc.Row([
            dbc.Col([
                create_portfolio_analysis_component()
            ], width=12)
        ], className="mb-4"),

        
    # Rebalancing Component
    dbc.Row([
            dbc.Col([
                create_rebalancing_component()
            ], width=12)
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

    # ML Prediction Component
    dbc.Row([
        dbc.Col([
            create_ml_prediction_component()
        ], width=12)
    ], className="mb-4"),

    # Portfolio Visualizer Component
    dbc.Row([
        dbc.Col([
            create_portfolio_visualizer_component()
        ], width=12)
    ], className="mb-4"),

    html.Div(id="help-modal-container") # Placeholder for the modal

])

# Callbacks
# recondiliation
@app.callback(
    Output("reconciliation-results", "children"),
    Input("reconcile-portfolio-button", "n_clicks"),
    prevent_initial_call=True
)
def run_portfolio_reconciliation(n_clicks):
    """
    Run portfolio reconciliation process to detect discrepancies
    between transaction history and current holdings
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Import the reconciliation function
    from modules.portfolio_utils import reconcile_portfolio_holdings
    
    # Run reconciliation
    results = reconcile_portfolio_holdings()
    
    # Display results
    if results['is_reconciled']:
        return dbc.Alert(
            f"Portfolio is fully reconciled. {results['portfolio_count']} positions match transaction history.",
            color="success"
        )
    else:
        # Create a detailed report for discrepancies
        discrepancy_rows = []
        for disc in results['discrepancies']:
            discrepancy_rows.append(
                html.Tr([
                    html.Td(disc['symbol']),
                    html.Td(disc['issue']),
                    html.Td(f"{disc['expected_shares']:.4f}"),
                    html.Td(f"{disc['actual_shares']:.4f}"),
                    html.Td(f"{disc.get('difference', 'N/A')}")
                ])
            )
        
        return html.Div([
            dbc.Alert(
                f"Found {len(results['discrepancies'])} discrepancies between transaction history and portfolio holdings.",
                color="warning"
            ),
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Issue"),
                        html.Th("Expected Shares"),
                        html.Th("Actual Shares"),
                        html.Th("Difference")
                    ])
                ),
                html.Tbody(discrepancy_rows)
            ], bordered=True, striped=True, hover=True)
        ])


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
    # from modules.fmp_api import fmp_api
    
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
            hist = data_provider.get_historical_price(symbol, start_date=start_date, end_date=end_date)
            
            if not hist.empty and 'close' in hist.columns:
                # Normalize values to start at 100 for better comparison
                first_valid_close = hist['close'].dropna().iloc[0] if not hist['close'].dropna().empty else 1
                if first_valid_close > 0:  # Avoid division by zero
                    normalized = hist['close'] / first_valid_close * 100
                    
                    # Add to chart
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode='lines',
                        name=f"{symbol} - {details.get('name', '')}",
                        visible='legendonly'
                    ))
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add a benchmark index (S&P/TSX Composite for Canadian focus)
    try:
        tsx_hist = data_provider.get_historical_price("^GSPTSE", start_date=start_date, end_date=end_date)
        
        if not tsx_hist.empty and 'close' in tsx_hist.columns:
            # Normalize TSX values
            first_valid_tsx = tsx_hist['close'].dropna().iloc[0] if not tsx_hist['close'].dropna().empty else 1
            if first_valid_tsx > 0:  # Avoid division by zero
                tsx_normalized = tsx_hist['close'] / first_valid_tsx * 100
                
                # Add to chart with dashed line
                fig.add_trace(go.Scatter(
                    x=tsx_hist.index,
                    y=tsx_normalized,
                    mode='lines',
                    name="S&P/TSX Composite",
                    line=dict(dash='dash', width=3)
                ))
        else:
            print("Warning: No benchmark data ('^GSPTSE') found for market overview.")

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

    
    # Create news cards
@app.callback(
    Output("news-analysis-content", "children"),
    Input("news-interval", "n_intervals")
)
def update_news_analysis(n):
    """
    Update news analysis with relevant stories about tracked assets and portfolio holdings
    using the Financial Modeling Prep API, including sentiment analysis
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
    
    # Remove any None values that might have crept in
    all_symbols = {symbol for symbol in all_symbols if symbol}
    
    # Split symbols into Canadian and US for better news targeting
    canadian_symbols = [s for s in all_symbols if s.endswith(".TO") or s.endswith(".V")]
    us_symbols = [s for s in all_symbols if not s.endswith(".TO") and not s.endswith(".V") and not s.startswith("MAW")]
    
    # # Import FMP API
    # from modules.fmp_api import fmp_api
    
    # Get news for Canadian symbols (focusing on TSX for Canadian market news)
    canadian_news = []
    if canadian_symbols:
        try:
            # Add TSX index to improve Canadian market news coverage
            if "^GSPTSE" not in canadian_symbols:
                canadian_symbols.append("^GSPTSE")
                
            canadian_news = data_provider.get_news(tickers=canadian_symbols, limit=15)
        except Exception as e:
            print(f"Error fetching Canadian news: {e}")
    
    # Get news for US symbols
    us_news = []
    if us_symbols:
        try:
            us_news = data_provider.get_news(tickers=us_symbols, limit=15)
        except Exception as e:
            print(f"Error fetching US news: {e}")
    
    # Get general market news if we don't have enough specific news
    general_news = []
    if len(canadian_news) + len(us_news) < 5:
        try:
            general_news = data_provider.get_news(limit=10)
        except Exception as e:
            print(f"Error fetching general news: {e}")
    
    # Combine all news sources and eliminate duplicates (by title)
    all_news = []
    seen_titles = set()
    
    for article in canadian_news + us_news + general_news:
        title = article.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            all_news.append(article)
    
    # Sort by date (most recent first)
    all_news.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
    
    # Limit to 10 most recent articles
    news_articles = all_news[:10]
    
    # If we still have no news (API issues), use fallback news
    if not news_articles:
        news_articles = [
            {"title": "Market Update: TSX Gains on Energy Rally", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source (API Unavailable)"}},
            {"title": "Bank of Canada Holds Interest Rates Steady", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source (API Unavailable)"}},
            {"title": "Tech Stocks Lead Market Recovery", "url": "#", "publishedAt": datetime.now().strftime("%Y-%m-%d"), "source": {"name": "Sample Source (API Unavailable)"}}
        ]
    
    # Import the sentiment analysis function
    def analyze_news_sentiment(article_text):
        """
        Perform basic sentiment analysis on news article text.
        
        Args:
            article_text (str): Combined title and description text
            
        Returns:
            dict: Sentiment analysis result with sentiment and score
        """
        # Simple keyword-based sentiment analysis
        positive_words = [
            'gain', 'gains', 'rise', 'rises', 'rising', 'up', 'upward', 'growth', 'grew', 'growing',
            'positive', 'profit', 'profitable', 'success', 'successful', 'bullish', 'strong', 'stronger',
            'rally', 'rallies', 'recover', 'recovery', 'climb', 'climbs', 'climbing', 'surge', 'surges',
            'high', 'higher', 'record', 'exceed', 'exceeds', 'beat', 'beats', 'outperform', 'increase',
            'increases', 'boost', 'boosts', 'upgrade', 'upgrades', 'buy', 'opportunity', 'opportunities'
        ]
        
        negative_words = [
            'loss', 'losses', 'fall', 'falls', 'falling', 'down', 'downward', 'decline', 'declines',
            'negative', 'deficit', 'weak', 'weaker', 'bearish', 'poor', 'plunge', 'plunges', 'drop',
            'drops', 'shrink', 'shrinks', 'shrinking', 'slump', 'slumps', 'slide', 'slides', 'tumble',
            'tumbles', 'low', 'lower', 'miss', 'misses', 'fail', 'fails', 'disappoint', 'disappoints',
            'underperform', 'decrease', 'decreases', 'cut', 'cuts', 'downgrade', 'downgrades', 'sell',
            'warning', 'warnings', 'risk', 'risks', 'concern', 'concerns', 'worried', 'worry', 'worries'
        ]
        
        # Convert text to lowercase and split into words
        words = article_text.lower().split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total_count = positive_count + negative_count
        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0
        
        # Determine sentiment category
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score
        }

    # Find relevant symbols and analyze sentiment for each article
    for article in news_articles:
        article["relevant_symbols"] = []
        article_title = article.get("title", "") or ""
        article_description = article.get("description", "") or ""
        article_content = (article_title + " " + article_description).lower()
        
        # Find relevant symbols
        for symbol in all_symbols:
            # Check if symbol is directly mentioned
            if symbol.lower() in article_content:
                article["relevant_symbols"].append(symbol)
            # Check company name if available
            elif symbol in tracked_assets and tracked_assets[symbol].get("name", "").lower() in article_content:
                article["relevant_symbols"].append(symbol)
            # For Canadian stocks, check without .TO suffix
            elif symbol.endswith(".TO") and symbol[:-3].lower() in article_content:
                article["relevant_symbols"].append(symbol)
        
        # Analyze sentiment
        article["sentiment"] = analyze_news_sentiment(article_title + " " + article_description)

    # Create news cards with sentiment indicators
    news_cards = []
    for article in news_articles:
        # Create tags for relevant symbols
        symbol_badges = []
        for symbol in article.get("relevant_symbols", []):
            symbol_badges.append(
                dbc.Badge(symbol, color="info", className="me-1")
            )
        
        # If no specific symbols are relevant, add a general market badge
        if not symbol_badges:
            symbol_badges.append(
                dbc.Badge("Market", color="secondary", className="me-1")
            )
        
        # Add sentiment badge with appropriate color
        sentiment = article.get("sentiment", {}).get("sentiment", "neutral")
        sentiment_color = {
            "positive": "success",
            "negative": "danger",
            "neutral": "secondary"
        }.get(sentiment, "secondary")
        
        sentiment_badge = dbc.Badge(
            sentiment.capitalize(), 
            color=sentiment_color,
            className="ms-1"
        )
        
        # Format date
        published_date = article.get("publishedAt", "")
        if published_date:
            try:
                if 'T' in published_date:  # ISO format with time
                    date_obj = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ")
                else:  # Just date
                    date_obj = datetime.strptime(published_date, "%Y-%m-%d")
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
        
        # Create card with improved styling
        news_cards.append(
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span(symbol_badges),
                        html.Span(sentiment_badge, className="float-end")
                    ], className="mb-1"),
                    html.Small(f"{source_name} - {formatted_date}", className="text-muted")
                ]),
                dbc.CardBody([
                    html.H5(article.get("title", "No title"), className="card-title"),
                    html.P(description, className="card-text"),
                    html.A("Read more", href=article.get("url", "#"), target="_blank", className="btn btn-sm btn-outline-primary")
                ])
            ], className="mb-3", outline=True, color=sentiment_color)
        )

    # Create summary of news sentiment
    positive_count = sum(1 for article in news_articles if article.get("sentiment", {}).get("sentiment") == "positive")
    negative_count = sum(1 for article in news_articles if article.get("sentiment", {}).get("sentiment") == "negative")
    neutral_count = sum(1 for article in news_articles if article.get("sentiment", {}).get("sentiment") == "neutral")

    # Determine overall market sentiment
    if positive_count > negative_count:
        overall_sentiment = "Positive"
        overall_color = "success"
    elif negative_count > positive_count:
        overall_sentiment = "Negative"
        overall_color = "danger"
    else:
        overall_sentiment = "Neutral"
        overall_color = "secondary"

    # Create sentiment summary component with separate progress bars (compatible with DBC 2.0.0)
    total_articles = positive_count + negative_count + neutral_count

    # Calculate percentages for each sentiment category
    positive_pct = (positive_count / total_articles * 100) if total_articles > 0 else 0
    neutral_pct = (neutral_count / total_articles * 100) if total_articles > 0 else 0
    negative_pct = (negative_count / total_articles * 100) if total_articles > 0 else 0

    sentiment_summary = dbc.Card([
        dbc.CardBody([
            html.H5("Market Sentiment Analysis", className="card-title"),
            html.P([
                f"Based on recent news, overall market sentiment appears to be ",
                html.Span(overall_sentiment, className=f"text-{overall_color} fw-bold")
            ]),
            
            # Use separate progress bars instead of the multi parameter
            html.Div([
                html.Div([
                    html.Span(f"Positive: {positive_count}", className="text-white px-2"),
                    dbc.Progress(value=positive_pct, color="success", style={"height": "24px"})
                ], className="mb-1"),
                html.Div([
                    html.Span(f"Neutral: {neutral_count}", className="text-white px-2"),
                    dbc.Progress(value=neutral_pct, color="secondary", style={"height": "24px"})
                ], className="mb-1"),
                html.Div([
                    html.Span(f"Negative: {negative_count}", className="text-white px-2"),
                    dbc.Progress(value=negative_pct, color="danger", style={"height": "24px"})
                ])
            ])
        ])
    ], className="mb-3")

    # Add a loading indicator and timestamp
    news_component = [
        sentiment_summary,
        html.Div([
            html.Small(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", className="text-muted mb-3 d-block"),
            *news_cards
        ])
    ]

    return news_component

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
#create portfolio summary component
@app.callback(
    Output("portfolio-summary", "children"),
    [Input("portfolio-update-interval", "n_intervals")]
)
def update_portfolio_summary(n_intervals):
    """
    Update the portfolio summary component with latest data
    """
    # Update portfolio data to get latest prices
    portfolio = update_portfolio_data()
    
    # Return the updated summary component
    return create_portfolio_summary_component(portfolio)

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
     Output("investment-name-input", "value"),  # New output for name field
     Output("investment-shares-input", "value"),
     Output("investment-price-input", "value")],
    [Input("add-investment-button", "n_clicks"),
     Input("portfolio-update-interval", "n_intervals"),
     Input({"type": "remove-investment-button", "index": ALL}, "n_clicks")],
    [State("investment-symbol-input", "value"),
     State("investment-name-input", "value"),  # New state for name field
     State("investment-shares-input", "value"),
     State("investment-price-input", "value"),
     State("investment-date-input", "value"),
     State("investment-type-select", "value")],
    prevent_initial_call=True
)
def manage_portfolio(add_clicks, update_interval, remove_clicks, symbol, name, shares, price, date, asset_type):
    """
    Consolidated callback to manage portfolio actions (add, update, remove)
    """
    # Initialize default return values
    feedback = None
    updated_table = dash.no_update
    clear_symbol = dash.no_update
    clear_name = dash.no_update  # New default for name
    clear_shares = dash.no_update
    clear_price = dash.no_update
    
    # Get the triggered component
    if not ctx.triggered:
        return feedback, updated_table, clear_symbol, clear_name, clear_shares, clear_price
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle add investment
    if trigger_id == "add-investment-button" and add_clicks:
        if not symbol or not shares or not price:
            feedback = dbc.Alert("Symbol, shares, and price are required", color="warning")
        else:
            # Standardize symbol format
            symbol_upper = symbol.upper().strip()
            
            # Get asset name (use symbol if not provided)
            asset_name = name.strip() if name else symbol_upper
            
            # Add investment with asset type and name
            # Note: you'll need to modify add_investment function to accept the name parameter
            success = add_investment(symbol_upper, shares, price, date, asset_type, asset_name)
            
            if success:
                feedback = dbc.Alert(f"Successfully added {asset_name} ({symbol_upper})", color="success")
                clear_symbol = ""
                clear_name = ""  # Clear name field
                clear_shares = None
                clear_price = None
            else:
                feedback = dbc.Alert("Failed to add investment", color="danger")
    
    # Handle remove investment (unchanged)
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
    
    return feedback, updated_table, clear_symbol, clear_name, clear_shares, clear_price

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

            # earliest_date_query = """
            # SELECT MIN(transaction_date) as earliest_date FROM transactions;
            # """
            earliest_result = get_earliest_transaction_date()
            
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
    Output("portfolio-summary", "children", allow_duplicate=True),
    [Input("refresh-portfolio-summary", "n_clicks")],
    prevent_initial_call=True
)
def refresh_portfolio_summary(n_clicks):
    """
    Manually refresh the portfolio summary
    """
    if n_clicks:
        portfolio = update_portfolio_data()
        return create_portfolio_summary_component(portfolio)
    raise dash.exceptions.PreventUpdate


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
    [Output("quick-transaction-feedback", "children"),
     Output("portfolio-table", "children", allow_duplicate=True),
     Output("quick-transaction-symbol", "value"),
     Output("quick-transaction-name", "value", allow_duplicate=True),
     Output("quick-transaction-shares", "value"),
     Output("quick-transaction-price", "value")],
    Input("record-quick-transaction", "n_clicks"),
    [State("quick-transaction-type", "value"),
     State("quick-transaction-symbol", "value"),
     State("quick-transaction-name", "value"),
     State("quick-transaction-asset-type", "value"),
     State("quick-transaction-shares", "value"),
     State("quick-transaction-price", "value"),
     State("quick-transaction-date", "value")],
    prevent_initial_call=True
)
def record_quick_transaction(n_clicks, transaction_type, symbol, name, asset_type, shares, price, date):
    """
    Record a transaction using the quick transaction form with name and asset type
    """
    if n_clicks is None or not symbol or shares is None or price is None:
        raise dash.exceptions.PreventUpdate
    
    # Standardize values
    symbol = symbol.upper().strip()
    
    try:
        # For buy transactions, we'll need to update the record_transaction function
        # to handle name and asset_type
        if transaction_type == "buy":
            # Check if this is a new asset not in the portfolio
            portfolio = load_portfolio()
            symbol_exists = any(details.get("symbol", "").upper() == symbol for _, details in portfolio.items())
            
            if not symbol_exists:
                # This is a new asset, so add it to the portfolio
                success = add_investment(symbol, shares, price, date, asset_type, name)
                action_desc = "purchase and addition"
            else:
                # Just record the transaction for existing asset
                success = record_transaction(transaction_type, symbol, price, shares, date, 
                                            notes="", asset_name=name, asset_type=asset_type)
                action_desc = "purchase"
        else:
            # For sells, just use the standard transaction recording
            success = record_transaction(transaction_type, symbol, price, shares, date)
            action_desc = "sale"

        
        # Update portfolio table
        portfolio = update_portfolio_data()
        updated_table = create_portfolio_table(portfolio)
        
        # Return appropriate feedback
        if success:
            return (
                dbc.Alert(f"Successfully recorded {action_desc} of {shares} shares of {symbol}", color="success"),
                updated_table,
                "",  # Clear symbol
                "",  # Clear name
                None,  # Clear shares
                None   # Clear price
            )
        else:
            return (
                dbc.Alert(f"Failed to record transaction. Please check your inputs.", color="danger"),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update
            )
    except Exception as e:
        return (
            dbc.Alert(f"Error recording transaction: {str(e)}", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
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
# @app.callback(
#     Output("current-vs-target-chart", "figure", allow_duplicate=True),
#     Input("rebalance-update-interval", "n_intervals"),
#     Input("rebalancing-tabs", "active_tab"),
#     prevent_initial_call='initial_duplicate'
# )
# def update_allocation_chart(n_intervals, active_tab):
#     """
#     Update the current vs target allocation chart
#     """
#     # Load portfolio data
#     portfolio = load_portfolio()
    
#     # Analyze current vs target allocation
#     analysis = analyze_current_vs_target(portfolio)
    
#     # Create and return the chart
#     return create_current_vs_target_chart(analysis)

# Callback to update allocation drift table
@app.callback(
    Output("current-vs-target-chart", "figure", allow_duplicate=True),
    Input("rebalance-update-interval", "n_intervals"),
    Input("rebalancing-tabs", "active_tab"),
    prevent_initial_call='initial_duplicate'
)
def update_allocation_chart_rebalance(n_intervals, active_tab): # Renamed to avoid conflict
    """Update the current vs target allocation chart"""
    portfolio = load_portfolio() # From portfolio_utils
    analysis = analyze_current_vs_target(portfolio) # From portfolio_utils
    return create_current_vs_target_chart(analysis) # From components

def update_allocation_drift_table_rebalance(n_intervals, active_tab): # Renamed
    """Update the allocation drift table"""
    portfolio = load_portfolio() # From portfolio_utils
    analysis = analyze_current_vs_target(portfolio) # From portfolio_utils
    return create_allocation_drift_table(analysis) # From components


# Callback to update rebalance summary
@app.callback(
    Output("rebalance-summary", "children"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_rebalance_summary_rebalance(n_intervals, active_tab): # Renamed
    """Update the rebalance summary"""
    portfolio = load_portfolio() # From portfolio_utils
    analysis = analyze_current_vs_target(portfolio) # From portfolio_utils
    return create_rebalance_summary(analysis) # From components

# Callback to update rebalance recommendations
@app.callback(
    Output("rebalance-recommendations", "children"),
    [Input("rebalance-update-interval", "n_intervals"),
     Input("rebalancing-tabs", "active_tab")]
)
def update_rebalance_recommendations_rebalance(n_intervals, active_tab): # Renamed
    """Update the rebalance recommendations"""
    if active_tab == "rebalancing-plan-tab":
        portfolio = load_portfolio() # From portfolio_utils
        profile = load_user_profile() # From portfolio_utils
        risk_level = profile.get("risk_level", 5)
        # create_rebalance_plan is still in portfolio_rebalancer, but uses analyze_current_vs_target from portfolio_utils
        rebalance_plan = create_rebalance_plan(portfolio, risk_level)
        return create_rebalance_recommendations(rebalance_plan) # From components
    return html.Div()

# Callback to initialize target allocation sliders
@app.callback(
    Output("target-allocation-sliders", "children"),
    [Input("rebalancing-tabs", "active_tab")]
)
def initialize_target_sliders_rebalance(active_tab): # Renamed
    """Initialize the target allocation sliders with current values"""
    if active_tab == "target-settings-tab":
        target_allocation = load_target_allocation() # From portfolio_utils
        return create_allocation_sliders(target_allocation) # From components
    return html.Div()

# Callback to update target allocation chart based on slider values
@app.callback(
    [Output("target-allocation-chart", "figure"),
     Output("slider-total-warning", "children")],
    [Input({"type": "target-slider", "asset_type": ALL}, "value"),
     Input({"type": "target-slider", "asset_type": ALL}, "id")]
)
def update_target_chart_rebalance(slider_values, slider_ids): # Renamed
    """Update the target allocation chart based on slider values"""
    # ... (logic remains the same, uses create_target_allocation_chart from components) ...
    new_target = {}
    for i, slider_id in enumerate(slider_ids):
        if i < len(slider_values):
            asset_type = slider_id["asset_type"]
            new_target[asset_type] = slider_values[i]
    total_allocation = sum(new_target.values())
    warning = dbc.Alert(f"Warning: Total allocation is {total_allocation:.1f}%. Target should sum to 100%.", color="warning") if abs(total_allocation - 100) > 1 else html.Div()
    return create_target_allocation_chart(new_target), warning # From components


@app.callback(
    Output("target-allocation-feedback", "children"),
    Input("save-target-allocation", "n_clicks"),
    State({"type": "target-slider", "asset_type": ALL}, "id"),
    State({"type": "target-slider", "asset_type": ALL}, "value"),
    prevent_initial_call=True
)

def save_target_settings_rebalance(n_clicks, slider_ids, slider_values): # Renamed
    """Save the target allocation settings"""
    if n_clicks is None: raise dash.exceptions.PreventUpdate
    new_target = {}
    for i, slider_id in enumerate(slider_ids):
        asset_type = slider_id["asset_type"]
        if i < len(slider_values): new_target[asset_type] = slider_values[i]

    total_allocation = sum(new_target.values())
    if abs(total_allocation - 100) > 1:
        return dbc.Alert(f"Error: Total allocation must sum to 100%. Current total: {total_allocation:.1f}%", color="danger")

    if save_target_allocation(new_target): # From portfolio_utils
        return dbc.Alert("Target allocation saved successfully.", color="success")
    else:
        return dbc.Alert("Error saving target allocation.", color="danger")

@app.callback(
    Output("help-offcanvas", "is_open"),
    Input("open-help-button", "n_clicks"),
    State("help-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_help_offcanvas(n_clicks, is_open):
    """
    Toggle the visibility of the help offcanvas when the help button is clicked
    """
    if n_clicks:
        print(f"Help button clicked (n_clicks={n_clicks}). Toggling state from {is_open} to {not is_open}")
        return not is_open
    return is_open

@app.callback(
    Output("help-modal-container", "children"),
    Input("open-help-button", "n_clicks"),
    prevent_initial_call=False  # Run on initial load to ensure component is ready
)
def load_help_component(n_clicks):
    """
    Load the help system component on button click or initial load
    """

    # Return the created component
    return create_help_display_component()

@app.callback(
    [Output("help-content-area", "children"),
     Output("help-content-title", "children"),
     Output({"type": "help-topic-link", "index": ALL}, "active")],
    [Input({"type": "help-topic-link", "index": ALL}, "n_clicks"),
     Input("open-help-button", "n_clicks")],  # Add this to handle initial state
    prevent_initial_call=True,
)
def display_help_topic(topic_clicks, help_button_clicks):
    """
    Display the selected help topic content or default topic on initial load
    """
    ctx_triggered = ctx.triggered_id
    
    # If triggered by the help button or initial load, show default content
    if ctx_triggered == "open-help-button" or ctx_triggered is None:
        # Get available topics
        topics = get_available_topics()
        if topics:
            # Show introduction by default
            default_topic = "introduction" if "introduction" in topics else next(iter(topics))
            html_content = get_help_content_html(default_topic)
            title = topics.get(default_topic, "Help")
            # Set the introduction or first topic as active
            active_states = [topic_id == default_topic for topic_id in topics.keys()]
            return dcc.Markdown(html_content, dangerously_allow_html=True), title, active_states
        else:
            return "No help topics available.", "Help", []
    
    # Otherwise, topic link was clicked
    if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "help-topic-link":
        topic_id = ctx_triggered["index"]
        html_content = get_help_content_html(topic_id)
        topics = get_available_topics()
        title = topics.get(topic_id, "Help Content")
        
        # Update active state for the clicked link
        active_states = [tid == topic_id for tid in topics.keys()]
        
        return dcc.Markdown(html_content, dangerously_allow_html=True), title, active_states
    
    # Default return if none of the above conditions are met
    raise PreventUpdate

@app.callback(
    [Output("deposit-feedback", "children"),
     Output("cash-flow-history-table", "children", allow_duplicate=True),
     Output("deposit-amount-input", "value"),
     Output("deposit-description-input", "value")],
    Input("record-deposit-button", "n_clicks"),
    [State("deposit-amount-input", "value"),
     State("deposit-currency-select", "value"),
     State("deposit-date-input", "value"),
     State("deposit-description-input", "value")],
    prevent_initial_call=True
)
def record_deposit(n_clicks, amount, currency, date, description):
    """
    Record a deposit (capital entering the portfolio)
    """
    from modules.portfolio_utils import record_cash_flow
    
    if n_clicks is None or not amount:
        raise dash.exceptions.PreventUpdate
    
    # Record the deposit
    success = record_cash_flow("deposit", amount, currency, date, description)
    
    if success:
        # Update cash flow history and clear form
        return (
            dbc.Alert(f"Successfully recorded deposit of {currency} {amount}", color="success"),
            create_cash_flow_table(),
            None,
            ""
        )
    else:
        return (
            dbc.Alert("Failed to record deposit", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update
        )
    
@app.callback(
    Output("resulting-usd-display", "children"),
    [Input("cad-amount-input", "value"),
     Input("exchange-rate-input", "value"),
     Input("refresh-exchange-rate-button", "n_clicks")]
)
def update_usd_display(cad_amount, exchange_rate, refresh_clicks):
    """
    Update the displayed USD amount based on CAD amount and exchange rate
    """
    # If refresh button clicked, get the latest rate
    ctx_triggered = ctx.triggered_id
    if ctx_triggered == "refresh-exchange-rate-button" and refresh_clicks:
        from modules.portfolio_utils import get_usd_to_cad_rate
        current_rate = get_usd_to_cad_rate()
        exchange_rate = 1 / current_rate if current_rate > 0 else 0

    if cad_amount and exchange_rate and exchange_rate > 0:
        usd_amount = cad_amount / exchange_rate
        return f"${usd_amount:.2f} USD"
    return "—"

@app.callback(
    [Output("exchange-feedback", "children"),
     Output("exchange-history-table", "children", allow_duplicate=True),
     Output("cad-amount-input", "value"),
     Output("exchange-description-input", "value"),
     Output("exchange-rate-input", "value", allow_duplicate=True)],
    Input("execute-exchange-button", "n_clicks"),
    [State("cad-amount-input", "value"),
     State("exchange-rate-input", "value"),
     State("exchange-date-input", "value"),
     State("exchange-description-input", "value")],
    prevent_initial_call=True
)
def execute_currency_exchange(n_clicks, cad_amount, exchange_rate, date, description):
    """
    Record a currency exchange from CAD to USD
    """
    from modules.portfolio_utils import record_currency_exchange
    
    if n_clicks is None or not cad_amount or not exchange_rate or exchange_rate <= 0:
        raise dash.exceptions.PreventUpdate
    
    # Calculate USD amount
    usd_amount = cad_amount / exchange_rate
    
    # Record the exchange
    success = record_currency_exchange("CAD", cad_amount, "USD", usd_amount, date, description)
    
    if success:
        # Update history table and clear form
        return (
            dbc.Alert(f"Successfully converted ${cad_amount:.2f} CAD to ${usd_amount:.2f} USD at rate {exchange_rate:.4f}", color="success"),
            create_exchange_history_table(),
            None,
            "",
            exchange_rate  # Keep the same exchange rate
        )
        # If exchange fails
    else:
        return (
            dbc.Alert("Failed to record currency exchange", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update
        )
    
@app.callback(
    [Output("withdrawal-feedback", "children"),
     Output("cash-flow-history-table", "children", allow_duplicate=True),
     Output("withdrawal-amount-input", "value"),
     Output("withdrawal-description-input", "value")],
    Input("record-withdrawal-button", "n_clicks"),
    [State("withdrawal-amount-input", "value"),
     State("withdrawal-currency-select", "value"),
     State("withdrawal-date-input", "value"),
     State("withdrawal-description-input", "value")],
    prevent_initial_call=True
)
def record_withdrawal(n_clicks, amount, currency, date, description):
    """
    Record a withdrawal (capital leaving the portfolio)
    """
    from modules.portfolio_utils import record_cash_flow
    
    if n_clicks is None or not amount:
        raise dash.exceptions.PreventUpdate
    
    # Record the withdrawal
    success = record_cash_flow("withdrawal", amount, currency, date, description)
    
    if success:
        # Update cash flow history and clear form
        return (
            dbc.Alert(f"Successfully recorded withdrawal of {currency} {amount}", color="success"),
            create_cash_flow_table(),
            None,
            ""
        )
    else:
        return (
            dbc.Alert("Failed to record withdrawal", color="danger"),
            dash.no_update,
            dash.no_update,
            dash.no_update
        )

@app.callback(
    Output("exchange-history-table", "children"),
    Input("exchange-tabs", "active_tab"),
    prevent_initial_call=False
)
def load_exchange_history(active_tab):
    """
    Load and display exchange history when the tab is selected
    """
    if active_tab == "tab-2":  # Adjust based on the actual tab ID
        return create_exchange_history_table()
    return dash.no_update

@app.callback(
    Output("fx-impact-summary", "children"),
    Input("analyze-fx-impact-button", "n_clicks"),
    prevent_initial_call=True
)
def show_fx_impact_analysis(n_clicks):
    """
    Analyze and display the FX impact on portfolio returns
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create and return FX impact summary
    from components.currency_exchange_component import create_fx_impact_summary
    return create_fx_impact_summary(portfolio)

@app.callback(
    Output("portfolio-summary", "children", allow_duplicate=True),
    [Input("execute-exchange-button", "n_clicks")],
    prevent_initial_call=True
)
def refresh_summary_after_exchange(n_clicks):
    """
    Refresh the portfolio summary after a currency exchange
    """
    if n_clicks:
        portfolio = update_portfolio_data()
        return create_portfolio_summary_component(portfolio)
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("cash-flow-history-table", "children"),
    Input("cash-tabs", "active_tab"),
    prevent_initial_call=False
)
def load_cash_flow_history(active_tab):
    """
    Load and display cash flow history when the tab is selected
    """
    if active_tab == "tab-2":  # Adjust based on the actual tab ID
        return create_cash_flow_table()
    return dash.no_update

# Update the portfolio summary to include cash positions and flows
@app.callback(
    Output("portfolio-summary", "children", allow_duplicate=True),
    [Input("record-deposit-button", "n_clicks"),
     Input("record-withdrawal-button", "n_clicks")],
    prevent_initial_call=True
)
def refresh_summary_after_cash_flow(deposit_clicks, withdrawal_clicks):
    """
    Refresh the portfolio summary after a deposit or withdrawal
    """
    portfolio = update_portfolio_data()
    return create_portfolio_summary_component(portfolio)

# Update the portfolio table to include cash positions and flows
@app.callback(
    [Output("quick-transaction-name", "value"),
     Output("quick-transaction-asset-type", "value")],
    Input("quick-transaction-symbol", "value"),
    prevent_initial_call=True
)
def populate_transaction_fields(symbol):
    """
    Auto-populate name and asset type fields when a symbol is entered
    """
    if not symbol:
        return "", "stock"  # Default values if symbol is empty
    
    # Standardize symbol
    symbol_upper = symbol.upper().strip()
    
    # Try to find in portfolio first
    portfolio = load_portfolio()
    for inv_id, details in portfolio.items():
        if details.get("symbol", "").upper() == symbol_upper:
            return details.get("name", symbol_upper), details.get("asset_type", "stock")
    
    # If not in portfolio, check tracked assets
    tracked_assets = load_tracked_assets()
    if symbol_upper in tracked_assets:
        return tracked_assets[symbol_upper].get("name", symbol_upper), tracked_assets[symbol_upper].get("type", "stock")
    
    # If symbol not found, just return the symbol as name and default asset type
    return symbol_upper, "stock"

register_ml_prediction_callbacks(app)

if __name__ == "__main__":
    multiprocessing.freeze_support() # For PyInstaller compatibility
    app.run(debug=True)