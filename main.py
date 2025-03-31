# Investment Recommendation System Architecture
# main.py - Entry point for the application

import dash
from dash import dcc, html, callback, Input, Output, State, ctx, no_update, ALL
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
from components.transaction_recorder import create_transaction_recorder_component, create_transaction_history_table
from modules.transaction_tracker import record_transaction
from modules.currency_utils import get_usd_to_cad_rate, format_currency, get_combined_value_cad

from modules.data_collector import DataCollector
from modules.market_analyzer import MarketAnalyzer
from modules.news_analyzer import NewsAnalyzer
from modules.recommendation_engine import RecommendationEngine
from modules.portfolio_tracker import PortfolioTracker
from components.asset_tracker import create_asset_tracker_component, load_tracked_assets, create_tracked_assets_table,  save_tracked_assets
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
    dbc.Row([
        dbc.Col([
            create_transaction_recorder_component()
        ], width=12)
    ], className="mb-4"),

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
    # from components.asset_tracker import load_tracked_assets, save_tracked_assets, create_tracked_assets_table
    
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
    Input({"type": "remove-asset-button", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def remove_asset(n_clicks_list):
    """
    Remove an asset when its remove button is clicked
    """
    # Print debug information
    print(f"Callback triggered with clicks: {n_clicks_list}")
    
    if not ctx.triggered_id:
        print("No trigger detected")
        raise PreventUpdate
    
    # Get the clicked button pattern-matching ID directly from ctx.triggered_id
    clicked_id = ctx.triggered_id
    print(f"Clicked button ID: {clicked_id}")
    
    if clicked_id and 'index' in clicked_id:
        symbol_to_remove = clicked_id['index']
        print(f"Symbol to remove: {symbol_to_remove}")
        
        # Load current assets
        assets = load_tracked_assets()
        print(f"Current assets: {list(assets.keys())}")
        
        # Remove the asset
        if symbol_to_remove in assets:
            print(f"Removing {symbol_to_remove}")
            del assets[symbol_to_remove]
            
            # Save updated assets
            save_success = save_tracked_assets(assets)
            print(f"Save success: {save_success}")
            
        # Return updated table
        return create_tracked_assets_table(assets)
    
    # If something went wrong, just refresh the current table
    return create_tracked_assets_table(load_tracked_assets())

@app.callback(
    Output("market-overview-graph", "figure"),
    Input("market-interval", "n_intervals")
)
def update_market_overview(n):
    """
    Update the market overview graph with actual historical data for tracked assets
    """
    from components.asset_tracker import load_tracked_assets
    import yfinance as yf
    from datetime import datetime, timedelta
    
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
    
    # Define time period (30 days by default)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create a figure
    fig = go.Figure()
    
    # Add data for each tracked asset
    for symbol, details in tracked_assets.items():
        try:
            # Get historical data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if not hist.empty:
                # Normalize values to start at 100 for better comparison
                normalized = hist['Close'] / hist['Close'].iloc[0] * 100
                
                # Add to chart
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=f"{symbol} - {details.get('name', '')}"
                ))
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
    
    # Add a benchmark index (S&P/TSX Composite for Canadian focus)
    try:
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date)
        
        if not tsx_hist.empty:
            # Normalize TSX values
            tsx_normalized = tsx_hist['Close'] / tsx_hist['Close'].iloc[0] * 100
            
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
    
    # Update layout
    fig.update_layout(
        title="Market Performance - Last 30 Days (Normalized to 100)",
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
    
    return fig

@app.callback(
    Output("news-analysis-content", "children"),
    Input("news-interval", "n_intervals")
)
def update_news_analysis(n):
    """
    Update news analysis with relevant stories about tracked assets and portfolio holdings
    """
    from components.asset_tracker import load_tracked_assets
    from modules.portfolio_data_updater import load_portfolio
    import requests
    from datetime import datetime, timedelta
    import os
    from dotenv import load_dotenv

    # Load environment variables for API keys
    load_dotenv()
    
    # Get API key from environment variable
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        return [
            dbc.Alert(
                "News API key not found in environment. Please add NEWS_API_KEY to your .env file.",
                color="warning"
            )
        ]
    
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
    
    try:
        # Calculate date for news search (7 days ago)
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Prepare for API call
        news_articles = []
        
        # Call News API for each search term (with rate limiting)
        for term in search_terms[:10]:  # Limit to 10 terms to avoid excessive API calls
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": term,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": news_api_key,
                "pageSize": 5  # Limit to 5 articles per term
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok" and data.get("articles"):
                        # Add relevant articles
                        news_articles.extend(data["articles"])
            except Exception as e:
                print(f"Error retrieving news for {term}: {e}")
        
        # Deduplicate articles
        unique_articles = {}
        for article in news_articles:
            title = article.get("title", "")
            if title and title not in unique_articles:
                unique_articles[title] = article
        
        # Use deduplicated articles
        if unique_articles:
            news_articles = list(unique_articles.values())
        else:
            news_articles = fallback_news
            
    except Exception as e:
        print(f"Error retrieving news: {e}")
        news_articles = fallback_news
    
    # Sort by date
    news_articles = sorted(news_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)
    
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
    
    # Filter out articles with no relevant symbols and limit to top 10
    relevant_articles = [a for a in news_articles if a.get("relevant_symbols")][:10]
    
    # Use relevant articles if we found any, otherwise use fallback
    if not relevant_articles:
        news_articles = fallback_news
    else:
        news_articles = relevant_articles
    
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