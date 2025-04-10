# components/portfolio_analysis.py
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from modules.yf_utils import get_ticker_info, download_yf_data
from modules.data_provider import data_provider
def create_portfolio_analysis_component():
    """
    Creates a component for analyzing portfolio allocation and diversification
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Analysis"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Graph(id="portfolio-allocation-chart"),
                    html.Div(id="allocation-details", className="mt-3")
                ], label="Asset Allocation"),
                
                dbc.Tab([
                    dcc.Graph(id="portfolio-sector-chart"),
                    html.Div(id="sector-details", className="mt-3")
                ], label="Sector Breakdown"),
                
                dbc.Tab([
                    dcc.Graph(id="portfolio-correlation-chart"),
                    html.Div(id="correlation-analysis", className="mt-3")
                ], label="Correlation Analysis")
            ]),
            dcc.Interval(
                id="analysis-update-interval",
                interval=3600000,  # 1 hour in milliseconds
                n_intervals=0
            )
        ])
    ])

def get_portfolio_allocation(portfolio):
    """
    Calculate the asset allocation of the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        DataFrame: Asset allocation data
    """
    if not portfolio:
        return pd.DataFrame()
    
    # Calculate total portfolio value
    total_value = sum(inv.get("current_value", 0) for inv in portfolio.values())
    
    if total_value == 0:
        return pd.DataFrame()
    
    # Get asset type for each symbol
    asset_types = {}
    for inv in portfolio.values():
        symbol = inv.get("symbol", "")
        value = inv.get("current_value", 0)
        
        # Determine asset type from symbol (this is a simplified approach)
        if symbol.endswith(".TO"):
            asset_type = "Canadian Equity"
        elif "." not in symbol:
            asset_type = "US Equity"
        elif symbol.endswith(".V"):
            asset_type = "Venture"
        elif "-CAD" in symbol:
            asset_type = "Cryptocurrency"
        elif symbol.startswith("XB") or symbol.endswith("BOND"):
            asset_type = "Bonds"
        else:
            asset_type = "Other"
            
        # Add to asset types
        if asset_type not in asset_types:
            asset_types[asset_type] = 0
        
        asset_types[asset_type] += value
    
    # Calculate percentages
    allocation_data = []
    for asset_type, value in asset_types.items():
        percentage = (value / total_value) * 100
        allocation_data.append({
            "Asset Type": asset_type,
            "Value": value,
            "Percentage": percentage
        })
    
    return pd.DataFrame(allocation_data)

def get_sector_breakdown(portfolio):
    """
    Get sector breakdown for the portfolio
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        DataFrame: Sector breakdown data
    """
    if not portfolio:
        return pd.DataFrame()
    
    # Calculate total portfolio value
    total_value = sum(inv.get("current_value", 0) for inv in portfolio.values())
    
    if total_value == 0:
        return pd.DataFrame()
    
    # Get sector data for each symbol
    sectors = {}
    for inv in portfolio.values():
        symbol = inv.get("symbol", "")
        value = inv.get("current_value", 0)
        
        try:
            # Get sector information from yfinance
            ticker_info = get_ticker_info(symbol)
            
            sector = ticker_info.get("sector", "Unknown")
            if not sector or sector == "":
                sector = "Other"
                
            # Add to sectors
            if sector not in sectors:
                sectors[sector] = 0
            
            sectors[sector] += value
        except Exception as e:
            print(f"Error getting sector data for {symbol}: {e}")
            # Use "Unknown" if can't get sector data
            if "Unknown" not in sectors:
                sectors["Unknown"] = 0
            
            sectors["Unknown"] += value
    
    # Calculate percentages
    sector_data = []
    for sector, value in sectors.items():
        percentage = (value / total_value) * 100
        sector_data.append({
            "Sector": sector,
            "Value": value,
            "Percentage": percentage
        })
    
    return pd.DataFrame(sector_data)

def calculate_correlation_matrix(portfolio, period="1y"):
    """
    Calculate correlation matrix for portfolio investments using FMP API.
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to calculate for ("1m", "3m", "6m", "1y", "all")
        
    Returns:
        DataFrame: Correlation matrix
    """
    if not portfolio:
        return pd.DataFrame()
    
    # Get symbols from portfolio
    symbols = []
    for inv_id, inv in portfolio.items():
        symbol = inv.get("symbol", "")
        if symbol:
            symbols.append(symbol)
    
    if not symbols:
        return pd.DataFrame()
    
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
        # Find earliest purchase date in portfolio
        earliest_date = end_date
        for inv in portfolio.values():
            try:
                purchase_date = datetime.strptime(inv.get("purchase_date", end_date.strftime("%Y-%m-%d")), "%Y-%m-%d")
                if purchase_date < earliest_date:
                    earliest_date = purchase_date
            except Exception as e:
                print(f"Error parsing purchase date: {e}")
                continue
        
        # Go back at least 1 day before earliest purchase to get a baseline
        start_date = earliest_date - timedelta(days=1)
    
    try:
        # Import FMP API
        # from modules.fmp_api import fmp_api
        
        # Get historical price data for each symbol using FMP API
        historical_data = {}
        for symbol in symbols:
            try:
                hist = data_provider.get_historical_price(symbol, start_date=start_date, end_date=end_date)
                
                if not hist.empty and 'close' in hist.columns:
                    historical_data[symbol] = hist['close']
            except Exception as e:
                print(f"Error getting historical data for {symbol} from FMP: {e}")
        
        # If we have at least 2 symbols with data, calculate correlation
        if len(historical_data) >= 2:
            # Create DataFrame with all price data
            prices_df = pd.DataFrame(historical_data)
            
            # Fill missing values with forward fill then backward fill
            prices_df = prices_df.ffill().bfill()
            
            # Calculate daily returns
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
        else:
            print("Not enough symbols with historical data to calculate correlation matrix")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()



def create_allocation_chart(portfolio):
    """
    Create a chart showing portfolio asset allocation
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Figure: Plotly figure
    """
    # Get allocation data
    allocation_df = get_portfolio_allocation(portfolio)
    
    if allocation_df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Asset Allocation (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Create pie chart
    fig = px.pie(
        allocation_df,
        values="Percentage",
        names="Asset Type",
        title="Portfolio Asset Allocation",
        hover_data=["Value"],
        labels={"Value": "Value ($)"}
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update hover template to show dollar values
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Value: $%{customdata[0]:.2f}<br>Percentage: %{percent:.1%}<extra></extra>"
    )
    
    return fig

def create_sector_chart(portfolio):
    """
    Create a chart showing portfolio sector breakdown
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Figure: Plotly figure
    """
    # Get sector data
    sector_df = get_sector_breakdown(portfolio)
    
    if sector_df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Sector Breakdown (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Sort by percentage
    sector_df = sector_df.sort_values("Percentage", ascending=False)
    
    # Create bar chart
    fig = px.bar(
        sector_df,
        x="Sector",
        y="Percentage",
        title="Portfolio Sector Breakdown",
        color="Sector",
        hover_data=["Value"],
        labels={"Value": "Value ($)"}
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Allocation (%)",
        legend_title="Sector",
        margin=dict(t=50, b=100)
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: $%{customdata[0]:.2f}<br>Percentage: %{y:.1f}%<extra></extra>"
    )
    
    return fig

def create_correlation_chart(portfolio):
    """
    Create a heatmap showing correlation between portfolio investments using FMP API.
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Figure: Plotly figure with correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(portfolio)
    
    if corr_matrix.empty:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Correlation Analysis (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Investment", y="Investment", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Investment Correlation Matrix"
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=50, b=50, l=50, r=50),
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1 (Negative)", "-0.5", "0 (None)", "0.5", "1 (Positive)"]
        )
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{y}</b> and <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>"
    )
    
    return fig


def create_allocation_details(portfolio):
    """
    Create details about portfolio allocation
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with allocation details
    """
    # Get allocation data
    allocation_df = get_portfolio_allocation(portfolio)
    
    if allocation_df.empty:
        return html.P("No allocation data available.")
    
    total_value = allocation_df["Value"].sum()
    
    # Format allocation data for display
    allocation_rows = []
    for _, row in allocation_df.iterrows():
        allocation_rows.append(
            html.Tr([
                html.Td(row["Asset Type"]),
                html.Td(f"${row['Value']:.2f}"),
                html.Td(f"{row['Percentage']:.2f}%")
            ])
        )
    
    # Create details component
    return html.Div([
        html.H5(f"Total Portfolio Value: ${total_value:.2f}"),
        html.P("Asset allocation breakdown:"),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Asset Type"),
                    html.Th("Value"),
                    html.Th("Percentage")
                ])
            ),
            html.Tbody(allocation_rows)
        ], bordered=True, striped=True, size="sm")
    ])

def create_sector_details(portfolio):
    """
    Create details about sector breakdown
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with sector details
    """
    # Get sector data
    sector_df = get_sector_breakdown(portfolio)
    
    if sector_df.empty:
        return html.P("No sector data available.")
    
    # Sort by percentage
    sector_df = sector_df.sort_values("Percentage", ascending=False)
    
    # Format sector data for display
    sector_rows = []
    for _, row in sector_df.iterrows():
        sector_rows.append(
            html.Tr([
                html.Td(row["Sector"]),
                html.Td(f"${row['Value']:.2f}"),
                html.Td(f"{row['Percentage']:.2f}%")
            ])
        )
    
    # Create details component
    return html.Div([
        html.P("Sector breakdown:"),
        dbc.Table([
            html.Thead(
                html.Tr([
                    html.Th("Sector"),
                    html.Th("Value"),
                    html.Th("Percentage")
                ])
            ),
            html.Tbody(sector_rows)
        ], bordered=True, striped=True, size="sm")
    ])

def create_correlation_analysis(portfolio):
    """
    Create analysis of portfolio correlation using FMP data.
    
    Args:
        portfolio (dict): Portfolio data
        
    Returns:
        Component: Dash component with correlation analysis
    """
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(portfolio)
    
    if corr_matrix.empty:
        return html.P("No correlation data available. Make sure you have at least two investments with sufficient historical data.")
    
    # Find highest and lowest correlations
    highest_corr = pd.DataFrame()
    lowest_corr = pd.DataFrame()
    
    if len(corr_matrix) > 1:
        # Create a dataframe of all pairs
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                symbol1 = corr_matrix.columns[i]
                symbol2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                pairs.append({
                    "Symbol1": symbol1,
                    "Symbol2": symbol2,
                    "Correlation": correlation
                })
        
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            
            # Get highest and lowest correlations
            highest_corr = pairs_df.sort_values("Correlation", ascending=False).head(3)
            lowest_corr = pairs_df.sort_values("Correlation").head(3)
    
    # Create analysis component
    content = [html.H5("Correlation Analysis")]
    
    if not highest_corr.empty and not lowest_corr.empty:
        # Add highest correlations
        content.append(html.P("Highest correlated investments (move together):"))
        highest_rows = []
        for _, row in highest_corr.iterrows():
            highest_rows.append(
                html.Tr([
                    html.Td(f"{row['Symbol1']} & {row['Symbol2']}"),
                    html.Td(f"{row['Correlation']:.3f}", className="text-danger" if row['Correlation'] > 0.7 else "")
                ])
            )
        
        content.append(
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Pair"),
                        html.Th("Correlation")
                    ])
                ),
                html.Tbody(highest_rows)
            ], bordered=True, striped=True, size="sm")
        )
        
        # Add lowest correlations
        content.append(html.P("Lowest correlated investments (good for diversification):"))
        lowest_rows = []
        for _, row in lowest_corr.iterrows():
            lowest_rows.append(
                html.Tr([
                    html.Td(f"{row['Symbol1']} & {row['Symbol2']}"),
                    html.Td(f"{row['Correlation']:.3f}", className="text-success" if row['Correlation'] < 0.3 else "")
                ])
            )
        
        content.append(
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Pair"),
                        html.Th("Correlation")
                    ])
                ),
                html.Tbody(lowest_rows)
            ], bordered=True, striped=True, size="sm")
        )
        
        # Add diversification guidance
        avg_correlation = np.mean([pair["Correlation"] for pair in pairs])
        
        if avg_correlation > 0.7:
            diversification_message = "Your portfolio has high correlation between investments, suggesting limited diversification. Consider adding investments from different sectors or asset classes."
            diversification_color = "danger"
        elif avg_correlation > 0.5:
            diversification_message = "Your portfolio has moderate correlation between investments. There may be opportunities to improve diversification."
            diversification_color = "warning"
        else:
            diversification_message = "Your portfolio has low correlation between investments, suggesting good diversification."
            diversification_color = "success"
        
        content.append(
            dbc.Alert(diversification_message, color=diversification_color, className="mt-3")
        )
    else:
        content.append(html.P("Add more investments to see correlation analysis."))
    
    return html.Div(content)
