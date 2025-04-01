# components/portfolio_visualizer.py
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from modules.currency_utils import get_historical_usd_to_cad_rates, get_usd_to_cad_rate, format_currency, get_combined_value_cad
from modules.historical_price_converter import get_corrected_historical_data

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
                            {"label": "1 Month", "value": "1mo"},
                            {"label": "3 Months", "value": "3mo"},
                            {"label": "6 Months", "value": "6mo"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "All Time", "value": "all"}
                        ],
                        value="3mo",
                        inline=True
                    )
                ], width=12)
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
    Get historical performance data for the portfolio with mutual fund support
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for
        
    Returns:
        DataFrame: Historical portfolio data
    """
    from modules.mutual_fund_provider import MutualFundProvider
    
    print(f"Starting get_portfolio_historical_data with period: {period}")
    
    if not portfolio:
        print("Portfolio is empty")
        return pd.DataFrame()
    
    # Map our UI periods to yfinance compatible periods
    yf_period_map = {
        "1m": "1mo",
        "3m": "3mo",
        "6m": "6mo",
        "1y": "1y",
        "all": "max"
    }
    
    # Determine date range based on period
    end_date = datetime.now()
    
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
    
    print(f"Date range: {start_date} to {end_date}")
    
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
    
    print(f"Regular symbols to fetch: {regular_symbols}")
    yf_period = yf_period_map.get(period, "max")
    
    # Process regular investments (stocks, ETFs, etc.)
    if regular_symbols:
        try:
            print("Using individual ticker downloads for regular investments")
            for symbol in regular_symbols:
                try:
                    print(f"Fetching data for {symbol} with period {yf_period}...")
                    ticker = yf.Ticker(symbol)
                    
                    # Use start/end date approach
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if hist.empty:
                        print(f"No data returned for {symbol}")
                        continue
                        
                    if 'Close' in hist.columns:
                        regular_symbol_data[symbol] = hist['Close']
                        print(f"Added {symbol} data with {len(regular_symbol_data[symbol])} rows")
                    else:
                        print(f"Missing 'Close' column for {symbol}")
                except Exception as e:
                    print(f"Error getting historical data for {symbol}: {e}")
        except Exception as e:
            print(f"Error processing regular investments: {e}")
    
    # Process mutual fund investments
    mutual_fund_data = {}
    mutual_fund_symbols = list(set(inv["symbol"] for _, inv in mutual_fund_investments))
    
    print(f"Mutual fund symbols to fetch: {mutual_fund_symbols}")
    if mutual_fund_symbols:
        for symbol in mutual_fund_symbols:
            try:
                print(f"Fetching mutual fund data for {symbol}...")
                fund_hist = mutual_fund_provider.get_historical_data(symbol, start_date, end_date)
                
                if not fund_hist.empty and 'Close' in fund_hist.columns:
                    mutual_fund_data[symbol] = fund_hist['Close']
                    print(f"Added mutual fund {symbol} data with {len(mutual_fund_data[symbol])} rows")
                else:
                    print(f"No historical data for mutual fund {symbol}")
            except Exception as e:
                print(f"Error getting mutual fund data for {symbol}: {e}")
    
    # Combine all price data
    all_symbols_data = {**regular_symbol_data, **mutual_fund_data}
    
    if not all_symbols_data:
        print("Failed to retrieve data for any symbols")
        return pd.DataFrame()
    
    # Create DataFrame with all price data
    price_df = pd.DataFrame(all_symbols_data)
    
    if price_df.empty:
        print("Combined price DataFrame is empty")
        return pd.DataFrame()
        
    print(f"Combined price data shape: {price_df.shape}")
    
    # Fill missing days with forward fill method
    price_df = price_df.fillna(method='ffill')
    
    # Calculate portfolio value for each day for each investment
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total'] = 0
    
    # Process all investments
    for investment_id, investment in portfolio.items():
        symbol = investment.get("symbol")
        shares = investment.get("shares", 0)
        
        if not symbol or not shares:
            print(f"Investment {investment_id} missing symbol or shares")
            continue
            
        if symbol not in price_df.columns:
            print(f"Symbol {symbol} not in price data")
            continue
        
        try:
            # Skip investments before their purchase date
            purchase_date_str = investment.get("purchase_date")
            if not purchase_date_str:
                print(f"Investment {investment_id} missing purchase date")
                continue
                
            purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
            
            # Calculate value for each day
            investment_value = price_df[symbol] * shares
            
            # Zero out values before purchase date
            investment_value.loc[investment_value.index < purchase_date] = 0
            
            # Add to portfolio total and store individual value
            column_name = f"{symbol}_{investment_id[-6:]}"
            portfolio_values[column_name] = investment_value
            portfolio_values['Total'] += investment_value
            
            print(f"Added investment {investment_id} ({symbol}) to portfolio values")
        except Exception as e:
            print(f"Error processing investment {investment_id}: {e}")
    
    if 'Total' not in portfolio_values.columns or portfolio_values['Total'].sum() == 0:
        print("Portfolio values missing Total column or Total is all zeros")
        return pd.DataFrame()
        
    print(f"Portfolio values shape: {portfolio_values.shape}")
    
    # Resample data to the chosen frequency
    if resample_freq != 'D':
        print(f"Resampling to {resample_freq} frequency")
        portfolio_values = portfolio_values.resample(resample_freq).last()
    
    # Get benchmark data - S&P/TSX Composite
    try:
        # Get TSX data
        print("Fetching TSX benchmark data...")
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date)
        
        if not tsx_hist.empty:
            # Resample TSX data if needed
            if resample_freq != 'D' and 'Close' in tsx_hist.columns:
                tsx_hist = tsx_hist.resample(resample_freq).last()
                
            # Normalize to match starting portfolio value
            if not portfolio_values.empty and 'Total' in portfolio_values.columns and len(portfolio_values) > 0:
                initial_portfolio = portfolio_values['Total'].iloc[0]
                if initial_portfolio > 0 and 'Close' in tsx_hist.columns and len(tsx_hist) > 0:
                    initial_tsx = tsx_hist['Close'].iloc[0]
                    if initial_tsx > 0:
                        tsx_normalized = tsx_hist['Close'] / initial_tsx * initial_portfolio
                        portfolio_values['TSX'] = tsx_normalized
                        print("Added TSX benchmark data")
                    else:
                        print("Initial TSX value is zero or negative")
                else:
                    print("Initial portfolio value is zero or negative, or missing TSX Close column")
            else:
                print("Couldn't normalize TSX data - empty portfolio values")
        else:
            print("TSX benchmark data is empty")
    except Exception as e:
        print(f"Error getting TSX data: {e}")
    
    print(f"Final portfolio values shape: {portfolio_values.shape}")
    if not portfolio_values.empty:
        print(f"Final columns: {portfolio_values.columns.tolist()}")
        
    return portfolio_values

def create_performance_graph(portfolio, period="3mo"):
    """
    Create a performance graph for the portfolio with enhanced error handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    print(f"Creating performance graph for period: {period}")
    
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
    
    if historical_data.empty:
        print("Historical data is empty")
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            template="plotly_white"
        )
        return fig
        
    if 'Total' not in historical_data.columns:
        print("'Total' column missing from historical data")
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (Missing Total Column)",
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
    investment_columns = [col for col in historical_data.columns if col not in ['Total', 'TSX']]
    
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
                
            print(f"Added {len(top_investments)} individual investment lines")
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
        print("Added TSX benchmark line")
    
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
                print(f"Added performance annotation: {performance_pct:.2f}%")
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
    
    # Format date on x-axis based on period
    dtick = None
    if period == "1m":
        dtick = "7D"  # Weekly ticks for 1 month view
    elif period == "3m":
        dtick = "14D"  # Bi-weekly ticks for 3 month view
    elif period == "6m":
        dtick = "1M"  # Monthly ticks for 6 month view
    elif period == "1y":
        dtick = "2M"  # Bi-monthly ticks for 1 year view
    
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
    
    print("Performance graph created successfully")
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


def get_portfolio_historical_data(portfolio, period="3m"):
    """
    Get historical performance data for the portfolio with proper currency handling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for
        
    Returns:
        DataFrame: Historical portfolio data
    """
    # Determine start date based on period
    end_date = datetime.now()
    if period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    else:  # "all"
        # Find earliest purchase date
        earliest_date = end_date
        for investment in portfolio.values():
            purchase_date = datetime.strptime(investment.get("purchase_date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
            if purchase_date < earliest_date:
                earliest_date = purchase_date
        
        # Go back at least 1 day before earliest purchase
        start_date = earliest_date - timedelta(days=1)
    
    # Get historical data for each symbol with proper currency correction
    symbol_data = {}
    symbols = set(inv["symbol"] for inv in portfolio.values())
    
    for symbol in symbols:
        try:
            # Get corrected historical data for the symbol
            hist_data = get_corrected_historical_data(symbol, period=period)
            
            if not hist_data.empty:
                symbol_data[symbol] = hist_data['Close']
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
    
    if not symbol_data:
        return pd.DataFrame()
    
    # Combine all price data
    price_df = pd.DataFrame(symbol_data)
    
    # Calculate portfolio value for each day
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total_CAD'] = 0
    portfolio_values['Total_USD'] = 0
    
    # Calculate total value based on shares
    for investment_id, investment in portfolio.items():
        symbol = investment["symbol"]
        shares = investment["shares"]
        currency = investment.get("currency", "USD")
        
        if symbol in price_df.columns:
            # Calculate value in the original currency
            value_series = price_df[symbol] * shares
            
            # Add to the appropriate currency total
            if currency == "CAD":
                portfolio_values[f"{symbol}_CAD"] = value_series
                portfolio_values['Total_CAD'] += value_series
            else:  # USD
                portfolio_values[f"{symbol}_USD"] = value_series
                portfolio_values['Total_USD'] += value_series
    
    # Get historical USD to CAD exchange rates for converting totals
    from modules.historical_price_converter import get_historical_usd_to_cad_rates
    exchange_rates = get_historical_usd_to_cad_rates(portfolio_values.index.min(), portfolio_values.index.max())
    
    # Convert USD to CAD and calculate total
    if 'Total_USD' in portfolio_values.columns and not portfolio_values['Total_USD'].empty:
        # For each date, convert USD to CAD using the historical exchange rate
        portfolio_values['USD_in_CAD'] = 0
        
        for date in portfolio_values.index:
            if date in exchange_rates.index:
                rate = exchange_rates.loc[date]
                portfolio_values.loc[date, 'USD_in_CAD'] = portfolio_values.loc[date, 'Total_USD'] * rate
            else:
                # Find closest available rate
                closest_idx = exchange_rates.index.get_indexer([date], method='nearest')[0]
                rate = exchange_rates.iloc[closest_idx]
                portfolio_values.loc[date, 'USD_in_CAD'] = portfolio_values.loc[date, 'Total_USD'] * rate
        
        # Calculate total in CAD
        portfolio_values['Total'] = portfolio_values['Total_CAD'] + portfolio_values['USD_in_CAD']
    else:
        # If no USD values, total is just the CAD total
        portfolio_values['Total'] = portfolio_values['Total_CAD']
    
    # Get benchmark data - S&P/TSX Composite (already in CAD)
    try:
        # Get TSX data
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date)
        
        if not tsx_hist.empty:
            # Normalize to match starting portfolio value
            initial_value = portfolio_values['Total'].iloc[0] if not portfolio_values.empty and 'Total' in portfolio_values.columns else 1000
            tsx_normalized = tsx_hist['Close'] / tsx_hist['Close'].iloc[0] * initial_value
            portfolio_values['TSX'] = tsx_normalized
    except Exception as e:
        print(f"Error getting TSX data: {e}")
    
    return portfolio_values