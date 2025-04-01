# components/portfolio_visualizer.py
"""
Optimized and simplified portfolio visualization component.
"""
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

# Import from consolidated utilities
from modules.portfolio_utils import (
    get_usd_to_cad_rate, get_historical_usd_to_cad_rates, 
    format_currency, get_combined_value_cad
)

def create_portfolio_visualizer_component():
    """
    Creates a component for visualizing portfolio performance
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Performance"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="portfolio-performance-graph"),
                    dcc.Interval(
                        id="portfolio-update-interval",
                        interval=3600000,  # 1 hour in milliseconds
                        n_intervals=0
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Time Period"),
                    dbc.RadioItems(
                        id="performance-period-selector",
                        options=[
                            {"label": "1 Month", "value": "1m"},
                            {"label": "3 Months", "value": "3m"},
                            {"label": "6 Months", "value": "6m"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "All Time", "value": "all"}
                        ],
                        value="3m",
                        inline=True
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Chart Type"),
                    dbc.RadioItems(
                        id="performance-chart-type",
                        options=[
                            {"label": "Actual Value", "value": "value"},
                            {"label": "Normalized (100)", "value": "normalized"}
                        ],
                        value="normalized",  # Default to normalized view
                        inline=True
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="portfolio-summary-stats", className="mt-3")
                ], width=12)
            ])
        ])
    ])

def get_portfolio_historical_data(portfolio, period="3m"):
    """
    Get historical performance data for the portfolio with robust timezone handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for
        
    Returns:
        DataFrame: Historical portfolio data
    """
    from modules.mutual_fund_provider import MutualFundProvider
    
    if not portfolio:
        return pd.DataFrame()
    
    # Define date range based on period
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if period == "1m":
        start_date = end_date - timedelta(days=30)
        resample_freq = 'D'  # Daily for shorter periods
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
        resample_freq = 'D'  # Daily
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
        resample_freq = 'D'  # Daily
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
        resample_freq = 'D'  # Daily
    else:  # "all"
        # Find earliest purchase date
        earliest_date = end_date
        for investment in portfolio.values():
            try:
                purchase_date = datetime.strptime(investment.get("purchase_date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                purchase_date = purchase_date.replace(hour=0, minute=0, second=0, microsecond=0)
                if purchase_date < earliest_date:
                    earliest_date = purchase_date
            except Exception as e:
                print(f"Error parsing purchase date: {e}")
                continue
        
        # Go back at least 1 day before earliest purchase
        start_date = earliest_date - timedelta(days=1)
        
        # Use weekly frequency for "all" time to avoid overcrowding the chart
        if (end_date - start_date).days > 365:
            resample_freq = 'W'  # Weekly
        else:
            resample_freq = 'D'  # Daily
    
    # Initialize mutual fund provider
    mutual_fund_provider = MutualFundProvider()
    
    # Separate investments by type to handle them differently
    regular_investments = []  # Stocks, ETFs, etc.
    mutual_fund_investments = []  # Mutual funds
    
    for inv_id, inv in portfolio.items():
        asset_type = inv.get("asset_type", "stock")
        if asset_type == "mutual_fund":
            mutual_fund_investments.append((inv_id, inv))
        else:
            regular_investments.append((inv_id, inv))
    
    # Get historical data for regular investments using yfinance
    regular_symbol_data = {}
    regular_symbols = list(set(inv["symbol"] for _, inv in regular_investments))
    
    # Process regular investments (stocks, ETFs, etc.)
    if regular_symbols:
        for symbol in regular_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Add one day to end_date to ensure we include the current day
                hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                
                if not hist.empty and 'Close' in hist.columns:
                    # Convert all dates to strings for consistent handling
                    hist_df = pd.DataFrame(hist['Close'])
                    hist_df.index = hist_df.index.strftime('%Y-%m-%d')
                    regular_symbol_data[symbol] = hist_df['Close']
            except Exception as e:
                print(f"Error getting historical data for {symbol}: {e}")
    
    # Process mutual fund investments
    mutual_fund_data = {}
    mutual_fund_symbols = list(set(inv["symbol"] for _, inv in mutual_fund_investments))
    
    if mutual_fund_symbols:
        for symbol in mutual_fund_symbols:
            try:
                fund_hist = mutual_fund_provider.get_historical_data(symbol, start_date, end_date)
                
                if not fund_hist.empty and 'Close' in fund_hist.columns:
                    # Convert all dates to strings for consistent handling
                    fund_df = pd.DataFrame(fund_hist['Close'])
                    fund_df.index = fund_df.index.strftime('%Y-%m-%d')
                    mutual_fund_data[symbol] = fund_df['Close']
            except Exception as e:
                print(f"Error getting mutual fund data for {symbol}: {e}")
    
    # Combine all price data
    all_symbols_data = {**regular_symbol_data, **mutual_fund_data}
    
    if not all_symbols_data:
        return pd.DataFrame()
    
    # Create DataFrame with all price data - string index dates
    price_df = pd.DataFrame(all_symbols_data)
    
    if price_df.empty:
        return pd.DataFrame()
    
    # Fill missing days with forward fill method
    price_df = price_df.ffill()  # Use ffill instead of fillna with method
    
    # Create a new dataframe for portfolio values using the same string dates
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total_CAD'] = 0
    portfolio_values['Total_USD'] = 0
    
    # Process all investments
    for investment_id, investment in portfolio.items():
        symbol = investment.get("symbol")
        shares = investment.get("shares", 0)
        currency = investment.get("currency", "USD")
        
        if not symbol or not shares or symbol not in price_df.columns:
            continue
        
        try:
            # Skip investments before their purchase date
            purchase_date_str = investment.get("purchase_date")
            if not purchase_date_str:
                continue
                
            # Convert purchase date to string format
            purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
            purchase_date_str = purchase_date.strftime('%Y-%m-%d')
            
            # Calculate value for each day
            investment_value = price_df[symbol] * shares
            
            # Column name for this investment
            column_name = f"{symbol}_{investment_id[-6:]}"
            
            # Initialize column with zeros
            portfolio_values[column_name] = 0
            
            # Create a mask for dates after or equal to purchase date (string comparison)
            mask = pd.Series(portfolio_values.index >= purchase_date_str, index=portfolio_values.index)
            
            # Set values only for dates after purchase
            portfolio_values.loc[mask, column_name] = investment_value.loc[mask]
            
            # Add to appropriate currency total
            if currency == "CAD":
                portfolio_values.loc[mask, 'Total_CAD'] += investment_value.loc[mask]
            else:  # USD
                portfolio_values.loc[mask, 'Total_USD'] += investment_value.loc[mask]
        except Exception as e:
            print(f"Error processing investment {investment_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Get historical USD to CAD exchange rates
    try:
        # Convert index back to datetime for exchange rate lookup
        start_idx = datetime.strptime(portfolio_values.index.min(), '%Y-%m-%d')
        end_idx = datetime.strptime(portfolio_values.index.max(), '%Y-%m-%d')
        
        if start_idx and end_idx:
            # Get exchange rates
            exchange_rates = get_historical_usd_to_cad_rates(start_idx, end_idx)
            
            # Convert the exchange rate index to strings for consistency
            exchange_rates_dict = {
                date.strftime('%Y-%m-%d'): rate 
                for date, rate in zip(exchange_rates.index, exchange_rates.values)
            }
            
            # Convert USD to CAD
            if 'Total_USD' in portfolio_values.columns and not portfolio_values['Total_USD'].empty:
                portfolio_values['USD_in_CAD'] = 0
                
                # For each date in portfolio_values
                for date_str in portfolio_values.index:
                    # Find the exchange rate for this date
                    if date_str in exchange_rates_dict:
                        rate = exchange_rates_dict[date_str]
                    else:
                        # Use the latest rate as fallback
                        rate = list(exchange_rates_dict.values())[-1]
                    
                    # Convert USD to CAD
                    portfolio_values.loc[date_str, 'USD_in_CAD'] = portfolio_values.loc[date_str, 'Total_USD'] * rate
                
                # Calculate total in CAD
                portfolio_values['Total'] = portfolio_values['Total_CAD'] + portfolio_values['USD_in_CAD']
            else:
                # If no USD values, total is just the CAD total
                portfolio_values['Total'] = portfolio_values['Total_CAD']
    except Exception as e:
        print(f"Error converting currencies: {e}")
        import traceback
        traceback.print_exc()
        # Fallback - just use CAD values
        portfolio_values['Total'] = portfolio_values['Total_CAD']
    
    # Get benchmark data - S&P/TSX Composite (already in CAD)
    try:
        # Get TSX data
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date + timedelta(days=1))
        
        if not tsx_hist.empty:
            # Convert TSX dates to strings
            tsx_hist_df = pd.DataFrame(tsx_hist['Close'])
            tsx_hist_df.index = tsx_hist_df.index.strftime('%Y-%m-%d')
            
            # Add TSX data for matching dates
            common_dates = set(portfolio_values.index).intersection(set(tsx_hist_df.index))
            
            if common_dates:
                for date_str in common_dates:
                    portfolio_values.loc[date_str, 'TSX'] = tsx_hist_df.loc[date_str, 'Close']
                
                # Fill any missing TSX values
                if 'TSX' in portfolio_values.columns:
                    portfolio_values['TSX'] = portfolio_values['TSX'].ffill()
    except Exception as e:
        print(f"Error adding benchmark data: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert the string index back to datetime for proper date formatting in charts
    portfolio_values.index = pd.to_datetime(portfolio_values.index)
    
    # Resample data to the chosen frequency
    if resample_freq != 'D':
        portfolio_values = portfolio_values.resample(resample_freq).last()
    
    return portfolio_values


def create_performance_graph(portfolio, period="3m"):
    """
    Create a performance graph for the portfolio with enhanced error handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    # Get historical data
    try:
        historical_data = get_portfolio_historical_data(portfolio, period)
    except Exception as e:
        print(f"Error getting historical data: {e}")
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio line
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Total'],
        mode='lines',
        name='Portfolio Total',
        line=dict(color='#2C3E50', width=3)
    ))
    
    # Add individual investment lines
    investment_columns = [col for col in historical_data.columns if col not in ['Total', 'TSX', 'Total_CAD', 'Total_USD', 'USD_in_CAD']]
    
    # Limit to top 5 investments by final value to avoid cluttering the chart
    if investment_columns:
        try:
            final_values = {col: historical_data[col].iloc[-1] for col in investment_columns 
                          if not historical_data[col].empty and historical_data[col].iloc[-1] > 0}
            
            top_investments = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for col, value in top_investments:
                # Extract the symbol from the column name
                symbol = col.split('_')[0] if '_' in col else col
                
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data[col],
                    mode='lines',
                    name=f"{symbol}",
                    line=dict(width=1.5),
                    visible='legendonly'  # Hidden by default
                ))
        except Exception as e:
            print(f"Error adding individual investment lines: {e}")
    
    # Add benchmark line if available
    if 'TSX' in historical_data.columns and not historical_data['TSX'].empty:
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['TSX'],
            mode='lines',
            name='S&P/TSX Composite',
            line=dict(color='#3498DB', width=2, dash='dash')
        ))
    
    # Calculate performance metrics
    try:
        if len(historical_data) > 1 and not historical_data['Total'].empty:
            initial_value = historical_data['Total'].iloc[0]
            final_value = historical_data['Total'].iloc[-1]
            
            # Skip if initial value is zero (to avoid division by zero)
            if initial_value > 0:
                performance_pct = ((final_value / initial_value) - 1) * 100
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Performance: {performance_pct:.2f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="white"
                    ),
                    align="left",
                    bgcolor="#2C3E50",
                    bordercolor="#1ABC9C",
                    borderwidth=2,
                    borderpad=4,
                    opacity=0.8
                )
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
    
    # Format date on x-axis based on period
    dtick = None
    if period == "1m":
        dtick = "d7"  # Weekly ticks for 1 month view
    elif period == "3m":
        dtick = "d14"  # Bi-weekly ticks for 3 month view
    elif period == "6m":
        dtick = "M1"  # Monthly ticks for 6 month view
    elif period == "1y":
        dtick = "M2"  # Bi-monthly ticks for 1 year view
    
    # Update layout
    period_text = {"1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year", "all": "All Time"}
    title_text = f"Portfolio Performance - {period_text.get(period, period)}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value ($)",
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
    
    # Apply date formatting if specified
    if dtick:
        fig.update_xaxes(dtick=dtick)
    
    return fig

def create_summary_stats(portfolio):
    """
    Create summary statistics for the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with summary statistics
    """
    if not portfolio:
        return html.Div("No investments to display.")
    
    # Group investments by currency
    cad_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "CAD"}
    usd_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "USD"}
    
    # Calculate total portfolio value and investment in CAD
    total_value_cad = sum(inv.get("current_value", 0) for inv in cad_investments.values())
    total_investment_cad = sum(inv.get("shares", 0) * inv.get("purchase_price", 0) for inv in cad_investments.values())
    
    # Calculate total portfolio value and investment in USD
    total_value_usd = sum(inv.get("current_value", 0) for inv in usd_investments.values())
    total_investment_usd = sum(inv.get("shares", 0) * inv.get("purchase_price", 0) for inv in usd_investments.values())
    
    # Get current exchange rate
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Convert USD to CAD
    total_value_usd_in_cad = total_value_usd * usd_to_cad_rate
    total_investment_usd_in_cad = total_investment_usd * usd_to_cad_rate
    
    # Calculate totals in CAD
    total_value = total_value_cad + total_value_usd_in_cad
    total_investment = total_investment_cad + total_investment_usd_in_cad
    
    # Calculate gain/loss
    total_gain_loss = total_value - total_investment
    total_gain_loss_pct = (total_value / total_investment - 1) * 100 if total_investment > 0 else 0
    
    # Group investments by ticker symbol to consolidate multiple purchases of the same security
    consolidated_investments = {}
    for inv_id, inv in portfolio.items():
        symbol = inv.get("symbol", "")
        if symbol not in consolidated_investments:
            consolidated_investments[symbol] = {
                "symbol": symbol,
                "currency": inv.get("currency", "USD"),
                "total_shares": 0,
                "total_investment": 0,
                "total_current_value": 0
            }
        
        # Add shares and values to the consolidated investment
        consolidated_investments[symbol]["total_shares"] += inv.get("shares", 0)
        consolidated_investments[symbol]["total_investment"] += inv.get("shares", 0) * inv.get("purchase_price", 0)
        consolidated_investments[symbol]["total_current_value"] += inv.get("current_value", 0)
    
    # Calculate consolidated gain/loss percentages
    investments_list = []
    for symbol, inv in consolidated_investments.items():
        if inv["total_investment"] > 0:
            gain_loss_pct = (inv["total_current_value"] / inv["total_investment"] - 1) * 100
            inv["gain_loss_pct"] = gain_loss_pct
            investments_list.append(inv)
    
    if investments_list:
        best_investment = max(investments_list, key=lambda x: x["gain_loss_pct"])
        worst_investment = min(investments_list, key=lambda x: x["gain_loss_pct"])
    else:
        best_investment = {"symbol": "N/A", "gain_loss_pct": 0, "currency": "CAD"}
        worst_investment = {"symbol": "N/A", "gain_loss_pct": 0, "currency": "CAD"}
    
    # Create summary cards
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Value (CAD)", className="card-title"),
                    html.H3(f"${total_value:.2f}", className="text-primary")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Gain/Loss (CAD)", className="card-title"),
                    html.H3(
                        f"${total_gain_loss:.2f} ({total_gain_loss_pct:.2f}%)", 
                        className=f"text-{'success' if total_gain_loss >= 0 else 'danger'}"
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Best Performer", className="card-title"),
                    html.H3(
                        f"{best_investment['symbol']} ({best_investment['gain_loss_pct']:.2f}%)", 
                        className="text-success"
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Worst Performer", className="card-title"),
                    html.H3(
                        f"{worst_investment['symbol']} ({worst_investment['gain_loss_pct']:.2f}%)", 
                        className="text-danger"
                    )
                ])
            ])
        ], width=3)
    ])# components/portfolio_visualizer.py
"""
Optimized and simplified portfolio visualization component with timezone handling fix.
"""
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import pytz

# Import from consolidated utilities
from modules.portfolio_utils import (
    get_usd_to_cad_rate, get_historical_usd_to_cad_rates, 
    format_currency, get_combined_value_cad
)
def create_portfolio_visualizer_component():
    """
    Creates a component for visualizing portfolio performance
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Performance"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="portfolio-performance-graph"),
                    dcc.Interval(
                        id="portfolio-update-interval",
                        interval=3600000,  # 1 hour in milliseconds
                        n_intervals=0
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Time Period"),
                    dbc.RadioItems(
                        id="performance-period-selector",
                        options=[
                            {"label": "1 Month", "value": "1m"},
                            {"label": "3 Months", "value": "3m"},
                            {"label": "6 Months", "value": "6m"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "All Time", "value": "all"}
                        ],
                        value="3m",
                        inline=True
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Chart Type"),
                    dbc.RadioItems(
                        id="performance-chart-type",
                        options=[
                            {"label": "Actual Value", "value": "value"},
                            {"label": "Normalized (100)", "value": "normalized"}
                        ],
                        value="normalized",  # Default to normalized view
                        inline=True
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="portfolio-summary-stats", className="mt-3")
                ], width=12)
            ])
        ])
    ])


# Replace the get_portfolio_historical_data function in components/portfolio_visualizer.py with this version

def get_portfolio_historical_data(portfolio, period="3m"):
    """
    Get historical performance data for the portfolio with robust timezone handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for
        
    Returns:
        DataFrame: Historical portfolio data
    """
    from modules.mutual_fund_provider import MutualFundProvider
    
    if not portfolio:
        return pd.DataFrame()
    
    # Define date range based on period
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    if period == "1m":
        start_date = end_date - timedelta(days=30)
        resample_freq = 'D'  # Daily for shorter periods
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
        resample_freq = 'D'  # Daily
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
        resample_freq = 'D'  # Daily
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
        resample_freq = 'D'  # Daily
    else:  # "all"
        # Find earliest purchase date
        earliest_date = end_date
        for investment in portfolio.values():
            try:
                purchase_date = datetime.strptime(investment.get("purchase_date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                purchase_date = purchase_date.replace(hour=0, minute=0, second=0, microsecond=0)
                if purchase_date < earliest_date:
                    earliest_date = purchase_date
            except Exception as e:
                print(f"Error parsing purchase date: {e}")
                continue
        
        # Go back at least 1 day before earliest purchase
        start_date = earliest_date - timedelta(days=1)
        
        # Use weekly frequency for "all" time to avoid overcrowding the chart
        if (end_date - start_date).days > 365:
            resample_freq = 'W'  # Weekly
        else:
            resample_freq = 'D'  # Daily
    
    # Initialize mutual fund provider
    mutual_fund_provider = MutualFundProvider()
    
    # Separate investments by type to handle them differently
    regular_investments = []  # Stocks, ETFs, etc.
    mutual_fund_investments = []  # Mutual funds
    
    for inv_id, inv in portfolio.items():
        asset_type = inv.get("asset_type", "stock")
        if asset_type == "mutual_fund":
            mutual_fund_investments.append((inv_id, inv))
        else:
            regular_investments.append((inv_id, inv))
    
    # Get historical data for regular investments using yfinance
    regular_symbol_data = {}
    regular_symbols = list(set(inv["symbol"] for _, inv in regular_investments))
    
    # Process regular investments (stocks, ETFs, etc.)
    if regular_symbols:
        for symbol in regular_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Add one day to end_date to ensure we include the current day
                hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                
                if not hist.empty and 'Close' in hist.columns:
                    # Convert all dates to strings for consistent handling
                    hist_df = pd.DataFrame(hist['Close'])
                    hist_df.index = hist_df.index.strftime('%Y-%m-%d')
                    regular_symbol_data[symbol] = hist_df['Close']
            except Exception as e:
                print(f"Error getting historical data for {symbol}: {e}")
    
    # Process mutual fund investments
    mutual_fund_data = {}
    mutual_fund_symbols = list(set(inv["symbol"] for _, inv in mutual_fund_investments))
    
    if mutual_fund_symbols:
        for symbol in mutual_fund_symbols:
            try:
                fund_hist = mutual_fund_provider.get_historical_data(symbol, start_date, end_date)
                
                if not fund_hist.empty and 'Close' in fund_hist.columns:
                    # Convert all dates to strings for consistent handling
                    fund_df = pd.DataFrame(fund_hist['Close'])
                    fund_df.index = fund_df.index.strftime('%Y-%m-%d')
                    mutual_fund_data[symbol] = fund_df['Close']
            except Exception as e:
                print(f"Error getting mutual fund data for {symbol}: {e}")
    
    # Combine all price data
    all_symbols_data = {**regular_symbol_data, **mutual_fund_data}
    
    if not all_symbols_data:
        return pd.DataFrame()
    
    # Create DataFrame with all price data - string index dates
    price_df = pd.DataFrame(all_symbols_data)
    
    if price_df.empty:
        return pd.DataFrame()
    
    # Fill missing days with forward fill method
    price_df = price_df.ffill()  # Use ffill instead of fillna with method
    
    # Create a new dataframe for portfolio values using the same string dates
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total_CAD'] = 0
    portfolio_values['Total_USD'] = 0
    
    # Process all investments
    for investment_id, investment in portfolio.items():
        symbol = investment.get("symbol")
        shares = investment.get("shares", 0)
        currency = investment.get("currency", "USD")
        
        if not symbol or not shares or symbol not in price_df.columns:
            continue
        
        try:
            # Skip investments before their purchase date
            purchase_date_str = investment.get("purchase_date")
            if not purchase_date_str:
                continue
                
            # Convert purchase date to string format
            purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
            purchase_date_str = purchase_date.strftime('%Y-%m-%d')
            
            # Calculate value for each day
            investment_value = price_df[symbol] * shares
            
            # Column name for this investment
            column_name = f"{symbol}_{investment_id[-6:]}"
            
            # Initialize column with zeros
            portfolio_values[column_name] = 0
            
            # Create a mask for dates after or equal to purchase date (string comparison)
            mask = pd.Series(portfolio_values.index >= purchase_date_str, index=portfolio_values.index)
            
            # Set values only for dates after purchase
            portfolio_values.loc[mask, column_name] = investment_value.loc[mask]
            
            # Add to appropriate currency total
            if currency == "CAD":
                portfolio_values.loc[mask, 'Total_CAD'] += investment_value.loc[mask]
            else:  # USD
                portfolio_values.loc[mask, 'Total_USD'] += investment_value.loc[mask]
        except Exception as e:
            print(f"Error processing investment {investment_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Get historical USD to CAD exchange rates
    try:
        # Convert index back to datetime for exchange rate lookup
        start_idx = datetime.strptime(portfolio_values.index.min(), '%Y-%m-%d')
        end_idx = datetime.strptime(portfolio_values.index.max(), '%Y-%m-%d')
        
        if start_idx and end_idx:
            # Get exchange rates
            exchange_rates = get_historical_usd_to_cad_rates(start_idx, end_idx)
            
            # Convert the exchange rate index to strings for consistency
            exchange_rates_dict = {
                date.strftime('%Y-%m-%d'): rate 
                for date, rate in zip(exchange_rates.index, exchange_rates.values)
            }
            
            # Convert USD to CAD
            if 'Total_USD' in portfolio_values.columns and not portfolio_values['Total_USD'].empty:
                portfolio_values['USD_in_CAD'] = 0
                
                # For each date in portfolio_values
                for date_str in portfolio_values.index:
                    # Find the exchange rate for this date
                    if date_str in exchange_rates_dict:
                        rate = exchange_rates_dict[date_str]
                    else:
                        # Use the latest rate as fallback
                        rate = list(exchange_rates_dict.values())[-1]
                    
                    # Convert USD to CAD
                    portfolio_values.loc[date_str, 'USD_in_CAD'] = portfolio_values.loc[date_str, 'Total_USD'] * rate
                
                # Calculate total in CAD
                portfolio_values['Total'] = portfolio_values['Total_CAD'] + portfolio_values['USD_in_CAD']
            else:
                # If no USD values, total is just the CAD total
                portfolio_values['Total'] = portfolio_values['Total_CAD']
    except Exception as e:
        print(f"Error converting currencies: {e}")
        import traceback
        traceback.print_exc()
        # Fallback - just use CAD values
        portfolio_values['Total'] = portfolio_values['Total_CAD']
    
    # Get benchmark data - S&P/TSX Composite (already in CAD)
    try:
        # Get TSX data
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date + timedelta(days=1))
        
        if not tsx_hist.empty:
            # Convert TSX dates to strings
            tsx_hist_df = pd.DataFrame(tsx_hist['Close'])
            tsx_hist_df.index = tsx_hist_df.index.strftime('%Y-%m-%d')
            
            # Add TSX data for matching dates
            common_dates = set(portfolio_values.index).intersection(set(tsx_hist_df.index))
            
            if common_dates:
                for date_str in common_dates:
                    portfolio_values.loc[date_str, 'TSX'] = tsx_hist_df.loc[date_str, 'Close']
                
                # Fill any missing TSX values
                if 'TSX' in portfolio_values.columns:
                    portfolio_values['TSX'] = portfolio_values['TSX'].ffill()
    except Exception as e:
        print(f"Error adding benchmark data: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert the string index back to datetime for proper date formatting in charts
    portfolio_values.index = pd.to_datetime(portfolio_values.index)
    
    # Resample data to the chosen frequency
    if resample_freq != 'D':
        portfolio_values = portfolio_values.resample(resample_freq).last()
    
    return portfolio_values

def create_performance_graph(portfolio, period="3m"):
    """
    Create a performance graph for the portfolio with enhanced error handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    # Get historical data
    try:
        historical_data = get_portfolio_historical_data(portfolio, period)
    except Exception as e:
        print(f"Error getting historical data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio line
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Total'],
        mode='lines',
        name='Portfolio Total',
        line=dict(color='#2C3E50', width=3)
    ))
    
    # Add individual investment lines
    investment_columns = [col for col in historical_data.columns if col not in ['Total', 'TSX', 'Total_CAD', 'Total_USD', 'USD_in_CAD']]
    
    # Limit to top 5 investments by final value to avoid cluttering the chart
    if investment_columns:
        try:
            final_values = {col: historical_data[col].iloc[-1] for col in investment_columns 
                          if not historical_data[col].empty and historical_data[col].iloc[-1] > 0}
            
            top_investments = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for col, value in top_investments:
                # Extract the symbol from the column name
                symbol = col.split('_')[0] if '_' in col else col
                
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data[col],
                    mode='lines',
                    name=f"{symbol}",
                    line=dict(width=1.5),
                    visible='legendonly'  # Hidden by default
                ))
        except Exception as e:
            print(f"Error adding individual investment lines: {e}")
    
    # Add benchmark line if available
    if 'TSX' in historical_data.columns and not historical_data['TSX'].empty:
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['TSX'],
            mode='lines',
            name='S&P/TSX Composite',
            line=dict(color='#3498DB', width=2, dash='dash')
        ))
    
    # Calculate performance metrics
    try:
        if len(historical_data) > 1 and not historical_data['Total'].empty:
            initial_value = historical_data['Total'].iloc[0]
            final_value = historical_data['Total'].iloc[-1]
            
            # Skip if initial value is zero (to avoid division by zero)
            if initial_value > 0:
                performance_pct = ((final_value / initial_value) - 1) * 100
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Performance: {performance_pct:.2f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="white"
                    ),
                    align="left",
                    bgcolor="#2C3E50",
                    bordercolor="#1ABC9C",
                    borderwidth=2,
                    borderpad=4,
                    opacity=0.8
                )
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
    
    # Format date on x-axis based on period
    dtick = None
    if period == "1m":
        dtick = "d7"  # Weekly ticks for 1 month view
    elif period == "3m":
        dtick = "d14"  # Bi-weekly ticks for 3 month view
    elif period == "6m":
        dtick = "M1"  # Monthly ticks for 6 month view
    elif period == "1y":
        dtick = "M2"  # Bi-monthly ticks for 1 year view
    
    # Update layout
    period_text = {"1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year", "all": "All Time"}
    title_text = f"Portfolio Performance - {period_text.get(period, period)}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value ($)",
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
    
    # Apply date formatting if specified
    if dtick:
        fig.update_xaxes(dtick=dtick)
    
    return fig

def create_summary_stats(portfolio):
    """
    Create summary statistics for the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with summary statistics
    """
    if not portfolio:
        return html.Div("No investments to display.")
    
    # Group investments by currency
    cad_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "CAD"}
    usd_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "USD"}
    
    # Calculate total portfolio value and investment in CAD
    total_value_cad = sum(inv.get("current_value", 0) for inv in cad_investments.values())
    total_investment_cad = sum(inv.get("shares", 0) * inv.get("purchase_price", 0) for inv in cad_investments.values())
    
    # Calculate total portfolio value and investment in USD
    total_value_usd = sum(inv.get("current_value", 0) for inv in usd_investments.values())
    total_investment_usd = sum(inv.get("shares", 0) * inv.get("purchase_price", 0) for inv in usd_investments.values())
    
    # Get current exchange rate
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Convert USD to CAD
    total_value_usd_in_cad = total_value_usd * usd_to_cad_rate
    total_investment_usd_in_cad = total_investment_usd * usd_to_cad_rate
    
    # Calculate totals in CAD
    total_value = total_value_cad + total_value_usd_in_cad
    total_investment = total_investment_cad + total_investment_usd_in_cad
    
    # Calculate gain/loss
    total_gain_loss = total_value - total_investment
    total_gain_loss_pct = (total_value / total_investment - 1) * 100 if total_investment > 0 else 0
    
    # Group investments by ticker symbol to consolidate multiple purchases of the same security
    consolidated_investments = {}
    for inv_id, inv in portfolio.items():
        symbol = inv.get("symbol", "")
        if symbol not in consolidated_investments:
            consolidated_investments[symbol] = {
                "symbol": symbol,
                "currency": inv.get("currency", "USD"),
                "total_shares": 0,
                "total_investment": 0,
                "total_current_value": 0
            }
        
        # Add shares and values to the consolidated investment
        consolidated_investments[symbol]["total_shares"] += inv.get("shares", 0)
        consolidated_investments[symbol]["total_investment"] += inv.get("shares", 0) * inv.get("purchase_price", 0)
        consolidated_investments[symbol]["total_current_value"] += inv.get("current_value", 0)
    
    # Calculate consolidated gain/loss percentages
    investments_list = []
    for symbol, inv in consolidated_investments.items():
        if inv["total_investment"] > 0:
            gain_loss_pct = (inv["total_current_value"] / inv["total_investment"] - 1) * 100
            inv["gain_loss_pct"] = gain_loss_pct
            investments_list.append(inv)
    
    if investments_list:
        best_investment = max(investments_list, key=lambda x: x["gain_loss_pct"])
        worst_investment = min(investments_list, key=lambda x: x["gain_loss_pct"])
    else:
        best_investment = {"symbol": "N/A", "gain_loss_pct": 0, "currency": "CAD"}
        worst_investment = {"symbol": "N/A", "gain_loss_pct": 0, "currency": "CAD"}
    
    # Create summary cards
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Value (CAD)", className="card-title"),
                    html.H3(f"${total_value:.2f}", className="text-primary")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Gain/Loss (CAD)", className="card-title"),
                    html.H3(
                        f"${total_gain_loss:.2f} ({total_gain_loss_pct:.2f}%)", 
                        className=f"text-{'success' if total_gain_loss >= 0 else 'danger'}"
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Best Performer", className="card-title"),
                    html.H3(
                        f"{best_investment['symbol']} ({best_investment['gain_loss_pct']:.2f}%)", 
                        className="text-success"
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Worst Performer", className="card-title"),
                    html.H3(
                        f"{worst_investment['symbol']} ({worst_investment['gain_loss_pct']:.2f}%)", 
                        className="text-danger"
                    )
                ])
            ])
        ], width=3)
    ])

def create_normalized_performance_graph(portfolio, period="3m"):
    """
    Create a normalized performance graph where all assets start at 100,
    making trend comparison much easier.
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with normalized performance graph
    """
    # Get historical data
    try:
        historical_data = get_portfolio_historical_data(portfolio, period)
    except Exception as e:
        print(f"Error getting historical data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Normalized Value (Starting at 100)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Normalized Value (Starting at 100)",
            template="plotly_white"
        )
        return fig
    
    # Create a new dataframe for normalized values
    normalized_data = pd.DataFrame(index=historical_data.index)
    
    # Get all relevant columns for normalization (portfolio total and individual investments)
    relevant_columns = ['Total']
    
    # Add TSX benchmark if available
    if 'TSX' in historical_data.columns:
        relevant_columns.append('TSX')
    
    # Add individual investment columns
    investment_columns = [col for col in historical_data.columns 
                          if col not in ['Total', 'TSX', 'Total_CAD', 'Total_USD', 'USD_in_CAD']]
    
    # Find top investments by final value to reduce clutter
    if investment_columns:
        try:
            final_values = {}
            for col in investment_columns:
                if col in historical_data.columns and not historical_data[col].empty and len(historical_data[col]) > 0:
                    # Get the last non-zero, non-NaN value
                    last_valid = historical_data[col].replace(0, np.nan).dropna()
                    if not last_valid.empty:
                        final_values[col] = last_valid.iloc[-1]
            
            # Sort investments by final value and take top 5
            top_investments = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:5]
            investment_columns = [col for col, _ in top_investments]
        except Exception as e:
            print(f"Error finding top investments: {e}")
    
    # Add top investment columns to the list
    relevant_columns.extend(investment_columns)
    
    # Normalize each column to start at 100
    for col in relevant_columns:
        if col in historical_data.columns:
            try:
                # Get series with non-zero values
                non_zero_series = historical_data[col].replace(0, np.nan).dropna()
                
                if not non_zero_series.empty:
                    # Get the first non-zero value
                    first_valid_value = non_zero_series.iloc[0]
                    
                    if first_valid_value > 0:
                        # Normalize to 100
                        normalized_data[col] = historical_data[col] / first_valid_value * 100
                        
                        # Replace NaN with 0 for dates before first valid value
                        normalized_data[col] = normalized_data[col].fillna(0)
            except Exception as e:
                print(f"Error normalizing column {col}: {e}")
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio line
    if 'Total' in normalized_data.columns:
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data['Total'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#2C3E50', width=3)
        ))
    
    # Add benchmark line if available
    if 'TSX' in normalized_data.columns:
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data['TSX'],
            mode='lines',
            name='S&P/TSX Composite',
            line=dict(color='#3498DB', width=2, dash='dash')
        ))
    
    # Add individual investment lines
    for col in investment_columns:
        if col in normalized_data.columns:
            # Extract the symbol from the column name
            symbol = col.split('_')[0] if '_' in col else col
            
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[col],
                mode='lines',
                name=f"{symbol}",
                line=dict(width=1.5),
                visible='legendonly'  # Hidden by default
            ))
    
    # Calculate performance metrics (since normalization starts at 100)
    try:
        if 'Total' in normalized_data.columns and len(normalized_data) > 1:
            # Get first and last valid values
            total_series = normalized_data['Total'].replace(0, np.nan).dropna()
            
            if not total_series.empty and len(total_series) > 1:
                first_valid = 100  # Normalized start
                final_valid = total_series.iloc[-1]
                
                final_performance = final_valid - first_valid  # Performance percentage
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Portfolio Performance: {final_performance:.2f}%",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="white"
                    ),
                    align="left",
                    bgcolor="#2C3E50",
                    bordercolor="#1ABC9C",
                    borderwidth=2,
                    borderpad=4,
                    opacity=0.8
                )
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
    
    # Format date on x-axis based on period
    dtick = None
    if period == "1m":
        dtick = "d7"  # Weekly ticks for 1 month view
    elif period == "3m":
        dtick = "d14"  # Bi-weekly ticks for 3 month view
    elif period == "6m":
        dtick = "M1"  # Monthly ticks for 6 month view
    elif period == "1y":
        dtick = "M2"  # Bi-monthly ticks for 1 year view
    
    # Update layout
    period_text = {"1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year", "all": "All Time"}
    title_text = f"Normalized Performance - {period_text.get(period, period)}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Normalized Value (Starting at 100)",
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
    
    # Add a reference line at 100 (starting value)
    fig.add_shape(
        type="line",
        x0=0,
        y0=100,
        x1=1,
        y1=100,
        xref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dot",
        )
    )
    
    # Add zones for positive and negative performance
    fig.add_shape(
        type="rect",
        x0=0,
        y0=100,
        x1=1,
        y1=200,  # Arbitrary upper limit
        xref="paper",
        fillcolor="rgba(0, 255, 0, 0.04)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,  # Arbitrary lower limit
        x1=1,
        y1=100,
        xref="paper",
        fillcolor="rgba(255, 0, 0, 0.04)",
        line=dict(width=0),
        layer="below"
    )
    
    # Apply date formatting if specified
    if dtick:
        fig.update_xaxes(dtick=dtick)
    
    return fig