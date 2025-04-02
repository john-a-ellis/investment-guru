# components/portfolio_visualizer.py
"""
Optimized and simplified portfolio visualization component with timezone handling fix.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Import from consolidated utilities
from modules.portfolio_utils import (
    get_usd_to_cad_rate, get_historical_usd_to_cad_rates
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
                            {"label": "Normalized (100)", "value": "normalized"},
                            {"label": "Relative (%)", "value": "relative"},
                            {"label": "Actual Value ($)", "value": "value"}
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
        
        # Go back at least 1 day before earliest purchase to get a baseline
        start_date = earliest_date - timedelta(days=1)
        
        # Use weekly frequency for "all" time to avoid overcrowding the chart for long periods
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
        # Use yfinance download for better performance with multiple symbols
        if len(regular_symbols) > 1:
            try:
                all_data = yf.download(regular_symbols, start=start_date, end=end_date + timedelta(days=1), progress=False)
                if not all_data.empty and 'Close' in all_data.columns:
                    if len(regular_symbols) == 1:
                        # Handle single symbol case where yfinance returns 1D DataFrame
                        closes = pd.DataFrame(all_data['Close'], columns=regular_symbols)
                    else:
                        closes = all_data['Close']
                        
                    for symbol in regular_symbols:
                        if symbol in closes.columns:
                            regular_symbol_data[symbol] = closes[symbol]
            except Exception as e:
                print(f"Error in bulk download: {e}, falling back to individual downloads")
                # Fall back to individual downloads if bulk download fails
                for symbol in regular_symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                        if not hist.empty and 'Close' in hist.columns:
                            regular_symbol_data[symbol] = hist['Close']
                    except Exception as e:
                        print(f"Error getting historical data for {symbol}: {e}")
        else:
            # Just use individual download for a single symbol
            for symbol in regular_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                    if not hist.empty and 'Close' in hist.columns:
                        regular_symbol_data[symbol] = hist['Close']
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
                    mutual_fund_data[symbol] = fund_hist['Close']
            except Exception as e:
                print(f"Error getting mutual fund data for {symbol}: {e}")
    
    # Combine all price data
    all_symbols_data = {**regular_symbol_data, **mutual_fund_data}
    
    if not all_symbols_data:
        return pd.DataFrame()
    
    # Create DataFrame with all price data
    price_df = pd.DataFrame(all_symbols_data)
    
    # Fill missing days with forward fill method
    price_df = price_df.ffill()
    
    # Create a new dataframe for portfolio values
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
                
            # Convert purchase date to datetime
            purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
            
            # Calculate value for each day
            investment_value = price_df[symbol] * shares
            
            # Column name for this investment
            column_name = f"{symbol}_{investment_id[-6:]}"
            
            # Initialize column with zeros
            portfolio_values[column_name] = 0
            
            # Create a mask for dates after or equal to purchase date
            mask = portfolio_values.index >= purchase_date
            
            # Set values only for dates after purchase
            portfolio_values.loc[mask, column_name] = investment_value.loc[mask]
            
            # Add to appropriate currency total
            if currency == "CAD":
                portfolio_values.loc[mask, 'Total_CAD'] += investment_value.loc[mask]
            else:  # USD
                portfolio_values.loc[mask, 'Total_USD'] += investment_value.loc[mask]
        except Exception as e:
            print(f"Error processing investment {investment_id}: {e}")
    
    # Get historical USD to CAD exchange rates
    try:
        from modules.portfolio_utils import get_historical_usd_to_cad_rates
        exchange_rates = get_historical_usd_to_cad_rates(start_date, end_date)
        
        # Convert USD to CAD
        if 'Total_USD' in portfolio_values.columns and (portfolio_values['Total_USD'] > 0).any():
            portfolio_values['USD_in_CAD'] = 0
            
            # Convert USD to CAD for each date
            for date in portfolio_values.index:
                # Find the exchange rate for this date
                closest_rate_date = exchange_rates.index[exchange_rates.index <= date][-1] if len(exchange_rates.index) > 0 else None
                
                if closest_rate_date is not None:
                    rate = exchange_rates[closest_rate_date]
                else:
                    # Use a default rate if no exchange rate data available
                    rate = 1.33
                
                # Convert USD to CAD
                portfolio_values.loc[date, 'USD_in_CAD'] = portfolio_values.loc[date, 'Total_USD'] * rate
            
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
            portfolio_values['TSX'] = tsx_hist['Close']
    except Exception as e:
        print(f"Error adding benchmark data: {e}")
    
    # Resample data to the chosen frequency
    if resample_freq != 'D':
        portfolio_values = portfolio_values.resample(resample_freq).last().ffill()
    
    return portfolio_values

def create_performance_graph(portfolio, period="3m"):
    """
    Create a performance graph for the portfolio with actual values
    
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
    
    # Add portfolio total line (bold)
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Total'],
        mode='lines',
        name='Portfolio Total',
        line=dict(color='#2C3E50', width=3)
    ))
    
    # Get individual investment columns
    investment_columns = [col for col in historical_data.columns 
                        if col not in ['Total', 'TSX', 'Total_CAD', 'Total_USD', 'USD_in_CAD']]
    
    # Find top investments by final value to reduce clutter
    if investment_columns:
        try:
            # Calculate final values
            final_values = {}
            for col in investment_columns:
                if col in historical_data.columns:
                    # Get last non-zero, non-NaN value
                    non_zero_values = historical_data[col].replace(0, np.nan).dropna()
                    if not non_zero_values.empty:
                        final_values[col] = non_zero_values.iloc[-1]
            
            # Sort by final value and take top 5
            if final_values:
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
            print(f"Error processing top investments: {e}")
    
    # Add benchmark line (dashed) if available
    if 'TSX' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['TSX'],
            mode='lines',
            name='S&P/TSX Composite',
            line=dict(color='#3498DB', width=2, dash='dash')
        ))
    
    # Calculate performance metrics for display
    try:
        total_series = historical_data['Total'].dropna()
        
        if len(total_series) > 1:
            start_value = total_series.iloc[0]
            end_value = total_series.iloc[-1]
            
            if start_value > 0:
                performance_pct = ((end_value / start_value) - 1) * 100
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Performance: {performance_pct:.2f}%",
                    showarrow=False,
                    font=dict(size=14, color="white"),
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
    import dash_bootstrap_components as dbc
    from dash import html
    from modules.portfolio_utils import get_usd_to_cad_rate
    
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
    
    # Find best and worst performers
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
                    html.H5("Total Investment", className="card-title"),
                    html.H3(f"${total_investment:.2f}", className="text-secondary")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Gain/Loss", className="card-title"),
                    html.H3(
                        f"${total_gain_loss:.2f} ({total_gain_loss_pct:.2f}%)", 
                        className="text-success" if total_gain_loss >= 0 else "text-danger"
                    )
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Best & Worst Performers", className="card-title"),
                    html.Div([
                        html.Span(f"Best: {best_investment['symbol']} ", className="font-weight-bold"),
                        html.Span(
                            f"+{best_investment['gain_loss_pct']:.2f}%" if best_investment['gain_loss_pct'] >= 0 else f"{best_investment['gain_loss_pct']:.2f}%",
                            className="text-success" if best_investment['gain_loss_pct'] >= 0 else "text-danger"
                        ),
                        html.Br(),
                        html.Span(f"Worst: {worst_investment['symbol']} ", className="font-weight-bold"),
                        html.Span(
                            f"+{worst_investment['gain_loss_pct']:.2f}%" if worst_investment['gain_loss_pct'] >= 0 else f"{worst_investment['gain_loss_pct']:.2f}%",
                            className="text-success" if worst_investment['gain_loss_pct'] >= 0 else "text-danger"
                        )
                    ])
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
    
    # Get all relevant columns for normalization
    relevant_columns = []
    
    # Always include the portfolio total if available
    if 'Total' in historical_data.columns:
        relevant_columns.append('Total')
    
    # Add TSX benchmark if available
    if 'TSX' in historical_data.columns:
        relevant_columns.append('TSX')
    
    # Add individual investment columns
    investment_columns = [col for col in historical_data.columns 
                         if col not in ['Total', 'TSX', 'Total_CAD', 'Total_USD', 'USD_in_CAD']]
    
    # Find top investments by final value to reduce chart clutter
    if investment_columns:
        try:
            # Calculate final values more reliably by taking the last non-zero value for each investment
            final_values = {}
            for col in investment_columns:
                if col in historical_data.columns:
                    # Get last non-zero, non-NaN value
                    non_zero_values = historical_data[col].replace(0, np.nan).dropna()
                    if not non_zero_values.empty:
                        final_values[col] = non_zero_values.iloc[-1]
            
            # Sort by final value and take top 5
            if final_values:
                top_investments = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:5]
                investment_columns = [col for col, _ in top_investments]
        except Exception as e:
            print(f"Error finding top investments: {e}")
    
    # Add selected investment columns to the relevant columns list
    relevant_columns.extend(investment_columns)
    
    # Normalize each column to start at 100
    for col in relevant_columns:
        if col not in historical_data.columns:
            continue
            
        column_data = historical_data[col]
        
        # Skip columns with all zeros/NaNs
        if column_data.replace(0, np.nan).dropna().empty:
            continue
        
        # Get the first non-zero, non-NaN value
        non_zero_values = column_data.replace(0, np.nan).dropna()
        
        if not non_zero_values.empty:
            first_val = non_zero_values.iloc[0]
            
            if first_val > 0:
                # Calculate normalized values (starting at 100)
                normalized_values = (column_data / first_val) * 100
                
                # Replace NaN with None to avoid displaying before start date
                normalized_data[col] = normalized_values
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio total line (bold)
    if 'Total' in normalized_data.columns:
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data['Total'],
            mode='lines',
            name='Portfolio Total',
            line=dict(color='#2C3E50', width=3)
        ))
    
    # Add benchmark line (dashed)
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
        if col in normalized_data.columns and not normalized_data[col].dropna().empty:
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
    
    # Calculate performance for display (if data available)
    if 'Total' in normalized_data.columns:
        try:
            total_series = normalized_data['Total'].dropna()
            
            if len(total_series) > 1:
                start_value = 100  # Normalized start value
                end_value = total_series.iloc[-1]
                performance_pct = end_value - start_value
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Portfolio Performance: {performance_pct:.2f}%",
                    showarrow=False,
                    font=dict(size=14, color="white"),
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
    
    # Add reference line at 100 (starting value)
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
        y1=200,  # Upper bound for positive zone
        xref="paper",
        fillcolor="rgba(0, 255, 0, 0.04)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,  # Lower bound for negative zone
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

def create_comparison_graph(portfolio, benchmark_symbols=None, period="3m"):
    """
    Create a performance comparison graph between portfolio and specified benchmarks
    
    Args:
        portfolio (dict): Portfolio data
        benchmark_symbols (list): List of benchmark symbols to compare against
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with comparison graph
    """
    import plotly.graph_objects as go
    
    # Default benchmark symbols if none provided
    if benchmark_symbols is None:
        benchmark_symbols = ['^GSPTSE', 'XIU.TO']  # TSX Composite and iShares S&P/TSX 60 ETF
    
    # Get historical data for portfolio
    try:
        portfolio_data = get_portfolio_historical_data(portfolio, period)
    except Exception as e:
        print(f"Error getting portfolio data: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Performance Comparison (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Normalized Value (Starting at 100)",
            template="plotly_white"
        )
        return fig
    
    if portfolio_data.empty or 'Total' not in portfolio_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Performance Comparison (No Portfolio Data Available)",
            xaxis_title="Date",
            yaxis_title="Normalized Value (Starting at 100)",
            template="plotly_white"
        )
        return fig
    
    # Define date range based on portfolio data
    start_date = portfolio_data.index.min()
    end_date = portfolio_data.index.max()
    
    # Create figure
    fig = go.Figure()
    
    # Normalize and add portfolio data
    portfolio_series = portfolio_data['Total'].dropna()
    if not portfolio_series.empty:
        first_val = portfolio_series.iloc[0]
        if first_val > 0:
            normalized_portfolio = (portfolio_series / first_val) * 100
            fig.add_trace(go.Scatter(
                x=normalized_portfolio.index,
                y=normalized_portfolio,
                mode='lines',
                name='Portfolio',
                line=dict(color='#2C3E50', width=3)
            ))
    
    # Get and normalize data for each benchmark
    for symbol in benchmark_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
            
            if not hist.empty and 'Close' in hist.columns:
                benchmark_series = hist['Close']
                first_val = benchmark_series.iloc[0]
                
                if first_val > 0:
                    normalized_benchmark = (benchmark_series / first_val) * 100
                    
                    # Get a friendly name for the benchmark
                    if symbol == '^GSPTSE':
                        name = 'S&P/TSX Composite'
                    elif symbol == 'XIU.TO':
                        name = 'iShares S&P/TSX 60 ETF'
                    else:
                        name = symbol
                    
                    fig.add_trace(go.Scatter(
                        x=normalized_benchmark.index,
                        y=normalized_benchmark,
                        mode='lines',
                        name=name,
                        line=dict(dash='dash')
                    ))
        except Exception as e:
            print(f"Error getting data for benchmark {symbol}: {e}")
    
    # Calculate relative performance
    try:
        # Get portfolio performance
        portfolio_series = portfolio_data['Total'].dropna()
        if len(portfolio_series) > 1:
            first_val = portfolio_series.iloc[0]
            last_val = portfolio_series.iloc[-1]
            if first_val > 0:
                portfolio_perf = ((last_val / first_val) - 1) * 100
                
                # Add portfolio performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Portfolio: {portfolio_perf:.2f}%",
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    align="left",
                    bgcolor="#2C3E50",
                    borderpad=4,
                    opacity=0.8
                )
    except Exception as e:
        print(f"Error calculating portfolio performance: {e}")
    
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
    title_text = f"Performance Comparison - {period_text.get(period, period)}"
    
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
    
    # Add reference line at 100 (starting value)
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
    
    # Apply date formatting if specified
    if dtick:
        fig.update_xaxes(dtick=dtick)
    
    return fig
def create_adaptive_scale_graph(portfolio, period="3m", relative_view=False):
    """
    Create a performance graph for the portfolio with adaptive scaling
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        relative_view (bool): Whether to show relative percentage changes instead of absolute values
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
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
            yaxis_title="Value (%)" if relative_view else "Value ($)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value (%)" if relative_view else "Value ($)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Process the data based on view type
    if relative_view:
        # For relative view, convert to percentage change from first value
        total_series = historical_data['Total'].dropna()
        if not total_series.empty:
            first_value = total_series.iloc[0]
            if first_value > 0:
                # Calculate percentage change
                relative_values = ((total_series / first_value) - 1) * 100
                
                # Add portfolio total line (bold)
                fig.add_trace(go.Scatter(
                    x=relative_values.index,
                    y=relative_values,
                    mode='lines',
                    name='Portfolio Total',
                    line=dict(color='#2C3E50', width=3)
                ))
                
                # Add zero line for reference
                fig.add_shape(
                    type="line",
                    x0=relative_values.index.min(),
                    y0=0,
                    x1=relative_values.index.max(),
                    y1=0,
                    line=dict(
                        color="gray",
                        width=1,
                        dash="dot",
                    )
                )
        
        # Process benchmark (TSX) if available
        if 'TSX' in historical_data.columns:
            tsx_series = historical_data['TSX'].dropna()
            if not tsx_series.empty:
                first_value = tsx_series.iloc[0]
                if first_value > 0:
                    # Calculate percentage change
                    relative_values = ((tsx_series / first_value) - 1) * 100
                    
                    # Add benchmark line (dashed)
                    fig.add_trace(go.Scatter(
                        x=relative_values.index,
                        y=relative_values,
                        mode='lines',
                        name='S&P/TSX Composite',
                        line=dict(color='#3498DB', width=2, dash='dash')
                    ))
    else:
        # Regular value view
        # Add portfolio total line (bold)
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Total'],
            mode='lines',
            name='Portfolio Total',
            line=dict(color='#2C3E50', width=3)
        ))
        
        # Add benchmark line (dashed) if available
        if 'TSX' in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['TSX'],
                mode='lines',
                name='S&P/TSX Composite',
                line=dict(color='#3498DB', width=2, dash='dash')
            ))
    
    # Calculate performance metrics for display
    try:
        total_series = historical_data['Total'].dropna()
        
        if len(total_series) > 1:
            start_value = total_series.iloc[0]
            end_value = total_series.iloc[-1]
            
            if start_value > 0:
                performance_pct = ((end_value / start_value) - 1) * 100
                
                # Add performance annotation
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"Performance: {performance_pct:.2f}%",
                    showarrow=False,
                    font=dict(size=14, color="white"),
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
    view_type = "Relative Performance" if relative_view else "Performance"
    title_text = f"Portfolio {view_type} - {period_text.get(period, period)}"
    
    y_title = "Relative Change (%)" if relative_view else "Value ($)"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title=y_title,
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