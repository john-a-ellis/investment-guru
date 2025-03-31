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
                            {"label": "1 Month", "value": "1m"},
                            {"label": "3 Months", "value": "3m"},
                            {"label": "6 Months", "value": "6m"},
                            {"label": "1 Year", "value": "1y"},
                            {"label": "All Time", "value": "all"}
                        ],
                        value="3m",
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
    Get historical performance data for the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to retrieve data for
        
    Returns:
        DataFrame: Historical portfolio data
    """
    if not portfolio:
        return pd.DataFrame()
        
    # Determine start date based on period
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
            purchase_date = datetime.strptime(investment.get("purchase_date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
            if purchase_date < earliest_date:
                earliest_date = purchase_date
        
        # Go back at least 1 day before earliest purchase
        start_date = earliest_date - timedelta(days=1)
        
        # Use weekly frequency for "all" time to avoid overcrowding the chart
        if (end_date - start_date).days > 365:
            resample_freq = 'W'  # Weekly
        else:
            resample_freq = 'D'  # Daily
    
    # Get historical data for each symbol
    symbol_data = {}
    symbols = [inv["symbol"] for inv in portfolio.values()]
    
    # Use a batch request if possible to minimize API calls
    try:
        ticker_data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
        
        # Process each symbol
        for symbol in symbols:
            if symbol in ticker_data.columns:
                hist = ticker_data[symbol]
                symbol_data[symbol] = hist['Close']
    except:
        # Fallback to individual requests if batch fails
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    symbol_data[symbol] = hist['Close']
            except Exception as e:
                print(f"Error getting historical data for {symbol}: {e}")
    
    if not symbol_data:
        return pd.DataFrame()
    
    # Combine all price data
    price_df = pd.DataFrame(symbol_data)
    
    # Fill missing days with forward fill method
    price_df = price_df.fillna(method='ffill')
    
    # Calculate portfolio value for each day for each investment
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total'] = 0
    
    # Create a separate column for each investment to show individual performance
    for investment_id, investment in portfolio.items():
        symbol = investment["symbol"]
        shares = investment["shares"]
        
        if symbol in price_df.columns:
            # Skip investments before their purchase date
            purchase_date = datetime.strptime(investment.get("purchase_date", start_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
            
            # Calculate value for each day
            investment_value = price_df[symbol] * shares
            
            # Zero out values before purchase date
            investment_value.loc[investment_value.index < purchase_date] = 0
            
            # Add to portfolio total and store individual value
            portfolio_values[f"{symbol}_{investment_id[-6:]}"] = investment_value
            portfolio_values['Total'] += investment_value
    
    # Resample data to the chosen frequency
    if resample_freq != 'D':
        portfolio_values = portfolio_values.resample(resample_freq).last()
    
    # Get benchmark data - S&P/TSX Composite
    try:
        # Get TSX data
        tsx = yf.Ticker("^GSPTSE")
        tsx_hist = tsx.history(start=start_date, end=end_date)
        
        if not tsx_hist.empty:
            # Resample TSX data if needed
            if resample_freq != 'D':
                tsx_hist = tsx_hist.resample(resample_freq).last()
                
            # Normalize to match starting portfolio value
            if not portfolio_values.empty and portfolio_values['Total'].iloc[0] > 0:
                initial_value = portfolio_values['Total'].iloc[0]
                tsx_normalized = tsx_hist['Close'] / tsx_hist['Close'].iloc[0] * initial_value
                portfolio_values['TSX'] = tsx_normalized
    except Exception as e:
        print(f"Error getting TSX data: {e}")
    
    return portfolio_values

def create_performance_graph(portfolio, period="3m"):
    """
    Create a performance graph for the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to display
        
    Returns:
        Figure: Plotly figure with performance graph
    """
    # Get historical data
    historical_data = get_portfolio_historical_data(portfolio, period)
    
    if historical_data.empty or 'Total' not in historical_data.columns:
        # Return empty figure if no data
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
    investment_columns = [col for col in historical_data.columns if col not in ['Total', 'TSX']]
    
    # Limit to top 5 investments by final value to avoid cluttering the chart
    if investment_columns:
        final_values = {col: historical_data[col].iloc[-1] for col in investment_columns if not historical_data[col].empty}
        top_investments = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for col, _ in top_investments:
            # Skip if all values are zero
            if historical_data[col].sum() == 0:
                continue
                
            # Extract the symbol from the column name
            symbol = col.split('_')[0]
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[col],
                mode='lines',
                name=f"{symbol}",
                line=dict(width=1.5),
                visible='legendonly'  # Hidden by default
            ))
    
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
    if len(historical_data) > 1:
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
    fig.update_layout(
        title=f"Portfolio Performance - {period.upper() if period != 'all' else 'All Time'}",
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