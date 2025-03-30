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
    
    # Get historical data for each symbol
    symbol_data = {}
    symbols = set(inv["symbol"] for inv in portfolio.values())
    
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
    
    # Get historical USD to CAD exchange rates
    usd_to_cad_rates = get_historical_usd_to_cad_rates(start_date, end_date)
    
    # Combine all price data
    price_df = pd.DataFrame(symbol_data)
    
    # Calculate portfolio value for each day
    portfolio_values = pd.DataFrame(index=price_df.index)
    portfolio_values['Total_CAD'] = 0
    portfolio_values['Total_USD'] = 0
    
    # Calculate total value based on shares and currency
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
    
    # Convert USD to CAD and calculate total
    if 'Total_USD' in portfolio_values.columns and not portfolio_values['Total_USD'].empty:
        # Convert USD values to CAD
        portfolio_values['USD_in_CAD'] = portfolio_values['Total_USD'] * usd_to_cad_rates
        
        # Fill any NaN values with the last valid rate
        if 'USD_in_CAD' in portfolio_values.columns:
            portfolio_values['USD_in_CAD'] = portfolio_values['USD_in_CAD'].fillna(method='ffill')
        
        # Calculate total in CAD
        portfolio_values['Total'] = portfolio_values['Total_CAD'] + portfolio_values.get('USD_in_CAD', 0)
    else:
        # If no USD values, total is just the CAD total
        portfolio_values['Total'] = portfolio_values['Total_CAD']
    
    # Get benchmark data - S&P/TSX Composite
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
        name='Portfolio (CAD)',
        line=dict(color='#2C3E50', width=3)
    ))
    
    # Add benchmark line if available
    if 'TSX' in historical_data.columns:
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
    
    # Update layout
    fig.update_layout(
        title=f"Portfolio Performance - {period.upper() if period != 'all' else 'All Time'} (CAD)",
        xaxis_title="Date",
        yaxis_title="Value (CAD $)",
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
    
    # Find best and worst performing investments
    investments_list = []
    for inv_id, inv in portfolio.items():
        gain_loss_pct = inv.get("gain_loss_percent", 0)
        investments_list.append({
            "symbol": inv.get("symbol", ""),
            "gain_loss_pct": gain_loss_pct,
            "currency": inv.get("currency", "USD")
        })
    
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