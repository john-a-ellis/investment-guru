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
import traceback
import plotly.graph_objects as go

from modules.data_provider import data_provider
from modules.portfolio_utils import get_money_weighted_return
from modules.portfolio_utils import load_transactions
from modules.portfolio_utils import get_usd_to_cad_rate
from modules.mutual_fund_provider import MutualFundProvider
from modules.yf_utils import download_yf_data, get_ticker_history
from modules.portfolio_utils import calculate_total_return



def get_portfolio_historical_data(portfolio, period="3m"):
    """
    Get historical performance data for the portfolio with robust timezone handling
    using the DataProvider.

    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for

    Returns:
        DataFrame: Historical portfolio data
    """

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
                purchase_date_str = investment.get("purchase_date", end_date.strftime("%Y-%m-%d"))
                if purchase_date_str: # Ensure purchase_date exists
                    purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
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

    # --- Use DataProvider for all data fetching ---
    all_symbols_data = {}
    all_symbols = list(set(inv["symbol"] for _, inv in portfolio.items() if inv.get("symbol")))

    for symbol in all_symbols:
        try:
            # Determine asset type to call the correct DataProvider method
            asset_type = next((inv.get("asset_type", "stock") for _, inv in portfolio.items() if inv.get("symbol") == symbol), "stock")

            if asset_type == "mutual_fund":
                # Use the dedicated mutual fund method in DataProvider
                hist = data_provider.get_mutual_fund_data(symbol, start_date=start_date, end_date=end_date)
            else:
                # Use the standard historical price method
                hist = data_provider.get_historical_price(symbol, start_date=start_date, end_date=end_date)

            # DataProvider already standardizes timezone and column names (lowercase)
            # --- FIX: Check for lowercase 'close' ---
            if not hist.empty and 'close' in hist.columns:
                # --- FIX: Use lowercase 'close' ---
                all_symbols_data[symbol] = hist['close']
            else:
                print(f"Warning: No historical data with 'close' column for {symbol}")
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    if not all_symbols_data:
        print("Warning: No historical data fetched for any symbol in the portfolio.")
        return pd.DataFrame()

    # Create DataFrame with all price data
    price_df = pd.DataFrame(all_symbols_data)

    # Fill missing days with forward fill method
    price_df = price_df.ffill().bfill() # Fill NaNs first

    # Create a new dataframe for portfolio values
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total_CAD'] = 0.0 # Initialize with float
    portfolio_values['Total_USD'] = 0.0 # Initialize with float

    # Process all investments
    for investment_id, investment in portfolio.items():
        symbol = investment.get("symbol")
        # Convert any Decimal types to float
        shares = float(investment.get("shares", 0))
        currency = investment.get("currency", "USD")

        if not symbol or shares <= 0 or symbol not in price_df.columns:
            continue

        try:
            # Skip investments before their purchase date
            purchase_date_str = investment.get("purchase_date")
            if not purchase_date_str:
                continue

            # Convert purchase date to datetime (timezone-naive)
            purchase_date = pd.to_datetime(purchase_date_str).tz_localize(None)

            # Calculate value for each day using the price_df
            investment_value = price_df[symbol] * shares

            # Column name for this investment
            column_name = f"{symbol}_{investment_id[-6:]}"

            # Initialize column with zeros
            portfolio_values[column_name] = 0.0 # Initialize with float

            # Create a mask for dates after or equal to purchase date
            mask = portfolio_values.index >= purchase_date

            # Set values only for dates after purchase
            # Ensure alignment and handle potential missing dates in investment_value
            valid_dates = investment_value.index.intersection(portfolio_values.loc[mask].index)
            portfolio_values.loc[valid_dates, column_name] = investment_value.loc[valid_dates]

            # Add to appropriate currency total
            if currency == "CAD":
                portfolio_values.loc[valid_dates, 'Total_CAD'] = portfolio_values.loc[valid_dates, 'Total_CAD'].fillna(0) + investment_value.loc[valid_dates].fillna(0)
            else:  # USD
                portfolio_values.loc[valid_dates, 'Total_USD'] = portfolio_values.loc[valid_dates, 'Total_USD'].fillna(0) + investment_value.loc[valid_dates].fillna(0)
        except Exception as e:
            print(f"Error processing investment {investment_id} ({symbol}): {e}")
            import traceback
            traceback.print_exc()

    # Get historical USD to CAD exchange rates using DataProvider
    try:
        # Use DataProvider's method
        exchange_rates_df = data_provider.get_historical_exchange_rates("USD", "CAD", start_date=start_date, end_date=end_date)

        if not exchange_rates_df.empty and 'rate' in exchange_rates_df.columns:
            exchange_rates = exchange_rates_df['rate']
            # Reindex and forward fill to match portfolio_values index
            exchange_rates = exchange_rates.reindex(portfolio_values.index, method='ffill').fillna(method='bfill')

            # Convert USD to CAD
            if 'Total_USD' in portfolio_values.columns and (portfolio_values['Total_USD'] > 0).any():
                portfolio_values['USD_in_CAD'] = portfolio_values['Total_USD'] * exchange_rates
                portfolio_values['Total'] = portfolio_values['Total_CAD'].fillna(0) + portfolio_values['USD_in_CAD'].fillna(0)
            else:
                # If no USD values, total is just the CAD total
                portfolio_values['Total'] = portfolio_values['Total_CAD'].fillna(0)
        else:
             # Fallback if exchange rates fail
            print("Warning: Failed to get historical exchange rates. Using default rate.")
            default_rate = data_provider.get_exchange_rate("USD", "CAD") # Get current rate as fallback
            portfolio_values['USD_in_CAD'] = portfolio_values['Total_USD'] * default_rate
            portfolio_values['Total'] = portfolio_values['Total_CAD'].fillna(0) + portfolio_values['USD_in_CAD'].fillna(0)

    except Exception as e:
        print(f"Error converting currencies: {e}")
        import traceback
        traceback.print_exc()
        # Fallback - just use CAD values
        portfolio_values['Total'] = portfolio_values['Total_CAD'].fillna(0)

    # Get benchmark data - S&P/TSX Composite (already in CAD) using DataProvider
    try:
        # Get TSX data from DataProvider
        tsx_hist = data_provider.get_historical_price("^GSPTSE", start_date=start_date, end_date=end_date)

        # DataProvider handles standardization
        # --- FIX: Check for lowercase 'close' ---
        if not tsx_hist.empty and 'close' in tsx_hist.columns:
            # --- FIX: Use lowercase 'close' ---
            # Reindex and forward fill to match portfolio_values index
            portfolio_values['TSX'] = tsx_hist['close'].reindex(portfolio_values.index, method='ffill').fillna(method='bfill')
        else:
            print("Warning: No benchmark data ('^GSPTSE') found.")
    except Exception as e:
        print(f"Error adding benchmark data: {e}")
        import traceback
        traceback.print_exc()

    # Resample data to the chosen frequency
    if resample_freq != 'D':
        portfolio_values = portfolio_values.resample(resample_freq).last().ffill()

    # Final check for NaNs in 'Total' column
    if 'Total' in portfolio_values:
        portfolio_values['Total'] = portfolio_values['Total'].fillna(0) # Fill any remaining NaNs with 0
        print(f"Final portfolio_values['Total'] length: {len(portfolio_values['Total'])}, NaNs: {portfolio_values['Total'].isna().sum()}")
    else:
        print("Warning: 'Total' column missing from final portfolio_values DataFrame.")
        return pd.DataFrame() # Return empty if 'Total' is somehow missing

    return portfolio_values


# Note: Define get_portfolio_historical_data first to avoid circular imports
from modules.portfolio_utils import calculate_twrr

def create_portfolio_visualizer_component():
    """
    Creates a component for visualizing portfolio performance with
    options for different calculation methods, including TWRR
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
                ], width=4),
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
                ], width=4),
                dbc.Col([
                    dbc.Label("Calculation Method"),
                    dbc.RadioItems(
                        id="performance-calculation-method",
                        options=[
                            {"label": "Simple Returns", "value": "simple"},
                            {"label": "Time-Weighted (TWRR)", "value": "twrr"},
                            {"label": "Money-Weighted (IRR)", "value": "mwr"}
                        ],
                        value="twrr",  # Default to TWRR
                        inline=True
                    )
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="portfolio-summary-stats", className="mt-3")
                ], width=12)
            ])
        ])
    ])

# Add this new function to create a TWRR chart
def create_twrr_performance_graph(portfolio, period="3m"):
    """
    Create a Time-Weighted Rate of Return (TWRR) performance graph that
    eliminates the impact of cash flows (deposits/withdrawals)
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with TWRR performance graph
    """
    
    try:
        # Calculate TWRR
        twrr_data = calculate_twrr(portfolio, period=period)
        
        if twrr_data['normalized_series'].empty:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Portfolio TWRR Performance (No Data Available)",
                xaxis_title="Date",
                yaxis_title="Time-Weighted Return (%)",
                template="plotly_white"
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Add TWRR series
        fig.add_trace(go.Scatter(
            x=twrr_data['normalized_series'].index,
            y=twrr_data['normalized_series'],
            mode='lines',
            name='Portfolio TWRR',
            line=dict(color='#2C3E50', width=3)
        ))
        
        # Add TSX benchmark for comparison (if available)
        
        historical_data = get_portfolio_historical_data(portfolio, period)
        
        if not historical_data.empty and 'TSX' in historical_data.columns:
            # Normalize TSX to 100 at start
            first_tsx = historical_data['TSX'].iloc[0]
            if first_tsx > 0:
                normalized_tsx = (historical_data['TSX'] / first_tsx) * 100
                
                fig.add_trace(go.Scatter(
                    x=normalized_tsx.index,
                    y=normalized_tsx,
                    mode='lines',
                    name='S&P/TSX Composite',
                    line=dict(color='#3498DB', width=2, dash='dash')
                ))
        
        # Add overall TWRR percentage to chart title
        twrr_pct = twrr_data['twrr']
        period_text = {"1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year", "all": "All Time"}
        title_text = f"Portfolio Performance (TWRR: {twrr_pct:.2f}%) - {period_text.get(period, period)}"
        
        # Add zone coloring for positive/negative performance
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
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="Time-Weighted Return (Base 100)",
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
        
    except Exception as e:
        print(f"Error creating TWRR graph: {e}")
        
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio TWRR Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Time-Weighted Return (%)",
            template="plotly_white"
        )
        return fig

# Add this new function to create an IRR/Money-Weighted Return summary
def create_mwr_summary(portfolio, period="3m"):
    """
    Create a summary of Money-Weighted Return (IRR) with deposit/withdrawal impact details
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        html.Div: Dash component with MWR summary and cash flow details
    """
    
    try:
        # Calculate Money-Weighted Return
        transactions = load_transactions()
        mwr = get_money_weighted_return(portfolio, transactions, period)
        
        # Format the period name for display
        period_text = {
            "1m": "1 Month", 
            "3m": "3 Months", 
            "6m": "6 Months", 
            "1y": "1 Year", 
            "all": "All Time"
        }.get(period, period)
        
        # Define date range based on period
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
            # Find earliest transaction date
            earliest_date = end_date
            for transaction_id, transaction in transactions.items():
                try:
                    transaction_date = datetime.strptime(transaction.get("date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                    if transaction_date < earliest_date:
                        earliest_date = transaction_date
                except Exception as e:
                    continue
            
            start_date = earliest_date - timedelta(days=1)
        
        # Get all transactions within the period
        period_transactions = []
        for transaction_id, transaction in transactions.items():
            try:
                transaction_date = datetime.strptime(transaction.get("date", ""), "%Y-%m-%d")
                if start_date <= transaction_date <= end_date:
                    period_transactions.append({
                        'date': transaction_date.strftime("%Y-%m-%d"),
                        'type': transaction.get("type", "").capitalize(),
                        'symbol': transaction.get("symbol", ""),
                        'amount': transaction.get("amount", 0)
                    })
            except Exception as e:
                continue
        
        # Sort transactions by date (newest first)
        period_transactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate total deposits and withdrawals
        total_deposits = sum(tx['amount'] for tx in period_transactions if tx['type'].lower() == 'buy')
        total_withdrawals = sum(tx['amount'] for tx in period_transactions if tx['type'].lower() == 'sell')
        
        # Create the summary component
        return html.Div([
            dbc.Alert([
                html.H4(f"Money-Weighted Return ({period_text}): {mwr:.2f}%", className="alert-heading"),
                html.P(f"This accounts for the timing and size of all deposits and withdrawals during this period."),
                html.Hr(),
                html.P([
                    f"Total Deposits: ${total_deposits:.2f}",
                    html.Br(),
                    f"Total Withdrawals: ${total_withdrawals:.2f}"
                ], className="mb-0")
            ], color="info"),
            
            html.H5("Cash Flow Details", className="mt-3"),
            
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Type"),
                        html.Th("Symbol"),
                        html.Th("Amount")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(tx['date']),
                        html.Td(tx['type']),
                        html.Td(tx['symbol']),
                        html.Td(f"${tx['amount']:.2f}")
                    ]) for tx in period_transactions[:10]  # Only show most recent 10 transactions
                ])
            ], bordered=True, striped=True, hover=True, size="sm")
        ])
        
    except Exception as e:
        print(f"Error creating MWR summary: {e}")
        
        # Return simple error message
        return html.Div([
            dbc.Alert(f"Error calculating Money-Weighted Return: {str(e)}", color="danger")
        ])

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
    
    
    if not portfolio:
        return html.Div("No investments to display.")
    
    # Group investments by currency
    cad_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "CAD"}
    usd_investments = {k: v for k, v in portfolio.items() if v.get("currency", "USD") == "USD"}
    
    # Calculate total portfolio value and investment in CAD
    total_value_cad = sum(float(inv.get("current_value", 0)) for inv in cad_investments.values())
    total_investment_cad = sum(float(inv.get("shares", 0)) * float(inv.get("purchase_price", 0)) for inv in cad_investments.values())
    
    # Calculate total portfolio value and investment in USD
    total_value_usd = sum(float(inv.get("current_value", 0)) for inv in usd_investments.values())
    total_investment_usd = sum(float(inv.get("shares", 0)) * float(inv.get("purchase_price", 0)) for inv in usd_investments.values())
    
    # Get current exchange rate
    usd_to_cad_rate = get_usd_to_cad_rate()
    
    # Convert USD to CAD
    total_value_usd_in_cad = total_value_usd * usd_to_cad_rate
    total_investment_usd_in_cad = total_investment_usd * usd_to_cad_rate
    
    # Calculate totals in CAD - explicitly convert all values to float to avoid decimal/float mismatch
    total_value = float(total_value_cad) + float(total_value_usd_in_cad)
    total_investment = float(total_investment_cad) + float(total_investment_usd_in_cad)
    
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
        
        # Add shares and values to the consolidated investment - explicitly convert to float
        consolidated_investments[symbol]["total_shares"] += float(inv.get("shares", 0))
        consolidated_investments[symbol]["total_investment"] += float(inv.get("shares", 0)) * float(inv.get("purchase_price", 0))
        consolidated_investments[symbol]["total_current_value"] += float(inv.get("current_value", 0))
    
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
    total_return_data = calculate_total_return(portfolio, period="1y")
    total_return_card = dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H5("Total Return (1yr)", className="card-title"),
                html.Div([
                    html.H3(
                        f"{total_return_data['total_return_pct']:.2f}%", 
                        className="text-success" if total_return_data['total_return_pct'] >= 0 else "text-danger"
                    ),
                    html.Div([
                        html.Span("Price Return: ", className="fw-bold"),
                        html.Span(
                            f"{total_return_data['price_return_pct']:.2f}%",
                            className="text-success" if total_return_data['price_return_pct'] >= 0 else "text-danger"
                        )
                    ]),
                    html.Div([
                        html.Span("Dividend Return: ", className="fw-bold"),
                        html.Span(
                            f"{total_return_data['dividend_return_pct']:.2f}%",
                            className="text-success"
                        )
                    ])
                ])
            ])
        ])
    ], width=3)
    new_row = dbc.Row([
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
        total_return_card  # Add the new card
    ], className="mb-4")

    # Return the updated row
    return new_row
    # return dbc.Row([
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H5("Total Value (CAD)", className="card-title"),
    #                 html.H3(f"${total_value:.2f}", className="text-primary")
    #             ])
    #         ])
    #     ], width=3),
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H5("Total Investment", className="card-title"),
    #                 html.H3(f"${total_investment:.2f}", className="text-secondary")
    #             ])
    #         ])
    #     ], width=3),
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H5("Total Gain/Loss", className="card-title"),
    #                 html.H3(
    #                     f"${total_gain_loss:.2f} ({total_gain_loss_pct:.2f}%)", 
    #                     className="text-success" if total_gain_loss >= 0 else "text-danger"
    #                 )
    #             ])
    #         ])
    #     ], width=3),
    #     dbc.Col([
    #         dbc.Card([
    #             dbc.CardBody([
    #                 html.H5("Best & Worst Performers", className="card-title"),
    #                 html.Div([
    #                     html.Span(f"Best: {best_investment['symbol']} ", className="font-weight-bold"),
    #                     html.Span(
    #                         f"+{best_investment['gain_loss_pct']:.2f}%" if best_investment['gain_loss_pct'] >= 0 else f"{best_investment['gain_loss_pct']:.2f}%",
    #                         className="text-success" if best_investment['gain_loss_pct'] >= 0 else "text-danger"
    #                     ),
    #                     html.Br(),
    #                     html.Span(f"Worst: {worst_investment['symbol']} ", className="font-weight-bold"),
    #                     html.Span(
    #                         f"+{worst_investment['gain_loss_pct']:.2f}%" if worst_investment['gain_loss_pct'] >= 0 else f"{worst_investment['gain_loss_pct']:.2f}%",
    #                         className="text-success" if worst_investment['gain_loss_pct'] >= 0 else "text-danger"
    #                     )
    #                 ])
    #             ])
    #         ])
    #     ], width=3)
    # ])

def create_normalized_performance_graph(portfolio, period="3m"):
    """
    Create a normalized performance graph where all assets start at 100
    
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
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Normalized Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Value (Normalized to 100)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Normalized Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value (Normalized to 100)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Normalize portfolio total to 100 at start
    if len(historical_data['Total']) > 0:
        first_value = historical_data['Total'].iloc[0]
        if first_value > 0:  # Avoid division by zero
            normalized_total = (historical_data['Total'] / first_value) * 100
            
            # Add normalized portfolio total line (bold)
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=normalized_total,
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
                    
                    # Normalize to 100 at start
                    first_inv_value = historical_data[col].replace(0, np.nan).dropna().iloc[0] if not historical_data[col].replace(0, np.nan).dropna().empty else 1
                    if first_inv_value > 0:  # Avoid division by zero
                        normalized_inv = (historical_data[col] / first_inv_value) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=normalized_inv,
                            mode='lines',
                            name=f"{symbol} (Normalized)",
                            line=dict(width=1.5),
                            visible='legendonly'  # Hidden by default
                        ))
        except Exception as e:
            print(f"Error processing top investments: {e}")
    
    # Add benchmark line (dashed) if available
    if 'TSX' in historical_data.columns:
        # Normalize TSX to 100 at start
        first_tsx = historical_data['TSX'].iloc[0]
        if first_tsx > 0:  # Avoid division by zero
            normalized_tsx = (historical_data['TSX'] / first_tsx) * 100
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=normalized_tsx,
                mode='lines',
                name='S&P/TSX Composite (Normalized)',
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
    title_text = f"Portfolio Performance (Normalized to 100) - {period_text.get(period, period)}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value (Normalized to 100)",
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


def create_adaptive_scale_graph(portfolio, period="3m", relative_view=False):
    """
    Create a performance graph with adaptive scaling, either showing
    relative percentage changes or absolute values
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        relative_view (bool): Whether to show relative percentage changes
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    # Get historical data
    try:
        historical_data = get_portfolio_historical_data(portfolio, period)
    except Exception as e:
        print(f"Error getting historical data: {e}")
        traceback.print_exc()
        
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Portfolio Performance (Error: {str(e)})",
            xaxis_title="Date",
            yaxis_title="Value" if not relative_view else "Change (%)",
            template="plotly_white"
        )
        return fig
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Portfolio Performance (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Value" if not relative_view else "Change (%)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Calculate relative or absolute values for portfolio total
    if relative_view:
        # Calculate percentage change from start
        if len(historical_data['Total']) > 0:
            first_value = historical_data['Total'].iloc[0]
            if first_value > 0:  # Avoid division by zero
                portfolio_values = ((historical_data['Total'] / first_value) - 1) * 100
            else:
                portfolio_values = historical_data['Total'] * 0  # All zeros if can't calculate
        else:
            portfolio_values = historical_data['Total'] * 0  # All zeros if empty
    else:
        # Use actual values
        portfolio_values = historical_data['Total']
    
    # Add portfolio total line (bold)
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=portfolio_values,
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
                    
                    if relative_view:
                        # Calculate percentage change from start
                        first_inv_value = historical_data[col].replace(0, np.nan).dropna().iloc[0] if not historical_data[col].replace(0, np.nan).dropna().empty else 1
                        if first_inv_value > 0:  # Avoid division by zero
                            inv_values = ((historical_data[col] / first_inv_value) - 1) * 100
                        else:
                            inv_values = historical_data[col] * 0  # All zeros if can't calculate
                    else:
                        # Use actual values
                        inv_values = historical_data[col]
                    
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=inv_values,
                        mode='lines',
                        name=f"{symbol}",
                        line=dict(width=1.5),
                        visible='legendonly'  # Hidden by default
                    ))
        except Exception as e:
            print(f"Error processing top investments: {e}")
    
    # Add benchmark line (dashed) if available
    if 'TSX' in historical_data.columns:
        if relative_view:
            # Calculate percentage change from start
            first_tsx = historical_data['TSX'].iloc[0]
            if first_tsx > 0:  # Avoid division by zero
                tsx_values = ((historical_data['TSX'] / first_tsx) - 1) * 100
            else:
                tsx_values = historical_data['TSX'] * 0  # All zeros if can't calculate
        else:
            # Use actual values
            tsx_values = historical_data['TSX']
            
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=tsx_values,
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
    
    # Add reference line at 0% for relative view
    if relative_view:
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=0,
            xref="paper",
            line=dict(
                color="gray",
                width=1,
                dash="dot",
            )
        )
    
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
    title_text = f"Portfolio Performance ({period_text.get(period, period)})"
    if relative_view:
        title_text = f"Relative Portfolio Performance ({period_text.get(period, period)})"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value ($)" if not relative_view else "Change (%)",
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