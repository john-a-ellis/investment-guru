# modules/portfolio_risk_metrics.py
"""
Enhanced risk metrics calculation module for the Investment Recommendation System.
Calculates Sharpe ratio, Sortino ratio, maximum drawdown, and other risk metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from modules.db_utils import execute_query
# from modules.portfolio_utils import get_historical_usd_to_cad_rates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_risk_metrics(portfolio, period="1y", risk_free_rate=0.05):
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period to calculate for ("1m", "3m", "6m", "1y", "all")
        risk_free_rate (float): Annual risk-free rate (default: 5%)
        
    Returns:
        dict: Dictionary containing all risk metrics
    """
    try:
        # Get historical portfolio data
        from components.portfolio_visualizer import get_portfolio_historical_data
        historical_data = get_portfolio_historical_data(portfolio, period)
        
        if historical_data.empty or 'Total' not in historical_data.columns:
            logger.warning("No historical data available for risk metrics calculation")
            return create_empty_metrics()
        
        # Get daily returns from portfolio values
        portfolio_values = historical_data['Total'].dropna()
        if len(portfolio_values) < 5:  # Need at least 5 data points for meaningful metrics
            logger.warning("Insufficient data points for risk metrics calculation")
            return create_empty_metrics()
        
        # Calculate daily and annualized returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Calculate basic statistics
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # Convert daily risk-free rate from annual
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate Sharpe Ratio
        excess_returns = daily_returns - daily_risk_free_rate
        sharpe_ratio = (mean_daily_return - daily_risk_free_rate) / std_daily_return if std_daily_return > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Annualize
        
        # Calculate Sortino Ratio (only considers downside deviation)
        # Filter returns that are below the target (risk-free rate)
        downside_returns = daily_returns[daily_returns < daily_risk_free_rate]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_daily_return - daily_risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        annualized_sortino = sortino_ratio * np.sqrt(252)  # Annualize
        
        # Calculate Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        max_drawdown = drawdowns.min()
        max_drawdown_pct = max_drawdown * 100  # Convert to percentage
        
        # Calculate Calmar Ratio (annualized return divided by max drawdown)
        # Estimate annual return based on available data
        total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        if total_days > 0:
            annualized_return = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (365 / total_days)) - 1
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        else:
            annualized_return = 0
            calmar_ratio = 0
        
        # Calculate Value at Risk (VaR) - 95% and 99% confidence
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()  # Conditional VaR (Expected Shortfall)
        
        # Calculate Beta if benchmark data is available ('TSX' in this case)
        beta = 0
        r_squared = 0
        if 'TSX' in historical_data.columns:
            benchmark_values = historical_data['TSX'].dropna()
            if len(benchmark_values) >= 5:
                benchmark_returns = benchmark_values.pct_change().dropna()
                # Align the series
                common_idx = daily_returns.index.intersection(benchmark_returns.index)
                if len(common_idx) >= 5:
                    portfolio_aligned = daily_returns.loc[common_idx]
                    benchmark_aligned = benchmark_returns.loc[common_idx]
                    # Calculate covariance and beta
                    covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
                    benchmark_variance = np.var(benchmark_aligned)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    # Calculate R-squared (correlation squared)
                    correlation = np.corrcoef(portfolio_aligned, benchmark_aligned)[0, 1]
                    r_squared = correlation ** 2
        
        # Calculate Information Ratio if benchmark data is available
        information_ratio = 0
        if 'TSX' in historical_data.columns:
            benchmark_values = historical_data['TSX'].dropna()
            if len(benchmark_values) >= 5:
                benchmark_returns = benchmark_values.pct_change().dropna()
                # Align the series
                common_idx = daily_returns.index.intersection(benchmark_returns.index)
                if len(common_idx) >= 5:
                    portfolio_aligned = daily_returns.loc[common_idx]
                    benchmark_aligned = benchmark_returns.loc[common_idx]
                    # Calculate excess returns over benchmark
                    excess_return_over_benchmark = portfolio_aligned - benchmark_aligned
                    # Information ratio is the mean of excess returns divided by their standard deviation
                    mean_excess = excess_return_over_benchmark.mean()
                    std_excess = excess_return_over_benchmark.std()
                    information_ratio = mean_excess / std_excess if std_excess > 0 else 0
                    # Annualize
                    information_ratio = information_ratio * np.sqrt(252)
        
        # Calculate Upside/Downside Capture Ratio
        upside_capture = 0
        downside_capture = 0
        if 'TSX' in historical_data.columns:
            benchmark_values = historical_data['TSX'].dropna()
            if len(benchmark_values) >= 5:
                benchmark_returns = benchmark_values.pct_change().dropna()
                # Align the series
                common_idx = daily_returns.index.intersection(benchmark_returns.index)
                if len(common_idx) >= 5:
                    portfolio_aligned = daily_returns.loc[common_idx]
                    benchmark_aligned = benchmark_returns.loc[common_idx]
                    
                    # Upside: when benchmark is positive
                    upside_index = benchmark_aligned > 0
                    if upside_index.any():
                        portfolio_upside = portfolio_aligned[upside_index].mean()
                        benchmark_upside = benchmark_aligned[upside_index].mean()
                        upside_capture = (portfolio_upside / benchmark_upside) * 100 if benchmark_upside > 0 else 0
                    
                    # Downside: when benchmark is negative
                    downside_index = benchmark_aligned < 0
                    if downside_index.any():
                        portfolio_downside = portfolio_aligned[downside_index].mean()
                        benchmark_downside = benchmark_aligned[downside_index].mean()
                        # For downside, lower (less negative) is better, so invert the ratio
                        downside_capture = (portfolio_downside / benchmark_downside) * 100 if benchmark_downside < 0 else 0
        
        # Compile all metrics into a dictionary
        return {
            'sharpe_ratio': annualized_sharpe,
            'sortino_ratio': annualized_sortino,
            'max_drawdown': max_drawdown_pct,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95 * 100,  # Convert to percentage
            'var_99': var_99 * 100,  # Convert to percentage
            'cvar_95': cvar_95 * 100 if not pd.isna(cvar_95) else 0,  # Convert to percentage
            'beta': beta,
            'r_squared': r_squared,
            'information_ratio': information_ratio,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'volatility': std_daily_return * np.sqrt(252) * 100,  # Annualized volatility in percentage
            'annualized_return': annualized_return * 100,  # Convert to percentage
            'daily_returns': daily_returns,
            'drawdowns': drawdowns,
            'has_data': True
        }
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        import traceback
        traceback.print_exc()
        return create_empty_metrics()

def create_empty_metrics():
    """
    Create a dictionary with empty risk metrics.
    
    Returns:
        dict: Empty risk metrics
    """
    return {
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'max_drawdown': 0,
        'calmar_ratio': 0,
        'var_95': 0,
        'var_99': 0,
        'cvar_95': 0,
        'beta': 0,
        'r_squared': 0,
        'information_ratio': 0,
        'upside_capture': 0,
        'downside_capture': 0,
        'volatility': 0,
        'annualized_return': 0,
        'daily_returns': pd.Series(),
        'drawdowns': pd.Series(),
        'has_data': False
    }

def get_risk_rating(metrics):
    """
    Calculate a risk rating (1-10) based on the risk metrics.
    
    Args:
        metrics (dict): Risk metrics dictionary
        
    Returns:
        int: Risk rating from 1 (low risk) to 10 (high risk)
    """
    # If no data, return moderate risk (5)
    if not metrics['has_data']:
        return 5
    
    # Calculate risk score based on multiple factors
    # 1. Volatility (annualized standard deviation) - higher means more risk
    volatility_score = min(10, max(1, metrics['volatility'] / 4))
    
    # 2. Max drawdown - more negative means more risk (absolute value)
    drawdown_score = min(10, max(1, abs(metrics['max_drawdown']) / 5))
    
    # 3. Sharpe ratio - higher means less risk (inverse relationship)
    sharpe_score = 10 - min(9, max(1, metrics['sharpe_ratio'] * 2)) if metrics['sharpe_ratio'] > 0 else 9
    
    # 4. VaR 95% - more negative means more risk (absolute value)
    var_score = min(10, max(1, abs(metrics['var_95']) * 5))
    
    # 5. Beta - higher absolute value means more risk
    beta_score = min(10, max(1, abs(metrics['beta']) * 5)) if metrics['beta'] != 0 else 5
    
    # Calculate weighted average (with more weight on volatility and drawdown)
    weights = [0.3, 0.3, 0.15, 0.15, 0.1]  # Sum to 1
    risk_score = (
        volatility_score * weights[0] +
        drawdown_score * weights[1] +
        sharpe_score * weights[2] +
        var_score * weights[3] +
        beta_score * weights[4]
    )
    
    # Round to nearest integer and ensure it's between 1 and 10
    return max(1, min(10, round(risk_score)))

def get_risk_description(risk_rating):
    """
    Get a description of the risk rating.
    
    Args:
        risk_rating (int): Risk rating from 1 to 10
        
    Returns:
        dict: Risk description with level, color, and text
    """
    if risk_rating <= 2:
        return {
            'level': 'Very Low',
            'color': 'success',
            'text': 'This portfolio has demonstrated very low volatility and drawdowns, with excellent risk-adjusted returns. Suitable for conservative investors with short investment horizons.'
        }
    elif risk_rating <= 4:
        return {
            'level': 'Low',
            'color': 'info',
            'text': 'This portfolio has shown low volatility and moderate drawdowns, with good risk-adjusted returns. Suitable for moderately conservative investors.'
        }
    elif risk_rating <= 6:
        return {
            'level': 'Moderate',
            'color': 'warning',
            'text': 'This portfolio has demonstrated moderate volatility and drawdowns, with reasonable risk-adjusted returns. Suitable for balanced investors with medium-term investment horizons.'
        }
    elif risk_rating <= 8:
        return {
            'level': 'High',
            'color': 'orange',
            'text': 'This portfolio has shown high volatility and significant drawdowns, with potential for high returns but also substantial losses. Suitable for growth-oriented investors with longer investment horizons.'
        }
    else:
        return {
            'level': 'Very High',
            'color': 'danger',
            'text': 'This portfolio has demonstrated very high volatility and major drawdowns, with potential for very high returns but also severe losses. Only suitable for aggressive investors with long investment horizons who can tolerate significant market fluctuations.'
        }

def create_risk_metrics_component(portfolio, period="1y"):
    """
    Create a dashboard component to display portfolio risk metrics.
    
    Args:
        portfolio (dict): Portfolio data
        period (str): Time period for metrics calculation
        
    Returns:
        Component: Dash component with risk metrics visualization
    """
    import dash_bootstrap_components as dbc
    from dash import html, dcc
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Calculate risk metrics
    metrics = calculate_risk_metrics(portfolio, period)
    
    # Calculate risk rating
    risk_rating = get_risk_rating(metrics)
    risk_description = get_risk_description(risk_rating)
    
    # Create risk gauge figure
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_rating,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio Risk Rating"},
        gauge={
            'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 3], 'color': 'green'},
                {'range': [3, 5], 'color': 'lightgreen'},
                {'range': [5, 7], 'color': 'yellow'},
                {'range': [7, 9], 'color': 'orange'},
                {'range': [9, 10], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_rating
            }
        }
    ))
    
    gauge_fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=10)
    )
    
    # Create drawdown chart
    drawdown_fig = go.Figure()
    
    if not metrics['drawdowns'].empty and len(metrics['drawdowns']) > 1:
        drawdown_fig.add_trace(go.Scatter(
            x=metrics['drawdowns'].index,
            y=metrics['drawdowns'] * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Update layout
        drawdown_fig.update_layout(
            title="Portfolio Drawdowns",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(tickformat='.1f')
        )
    else:
        drawdown_fig.update_layout(
            title="Portfolio Drawdowns (No Data Available)",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
    
    # Create metrics cards
    metrics_cards = [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sharpe Ratio", className="card-title"),
                    html.H3(f"{metrics['sharpe_ratio']:.2f}", className="text-primary"),
                    html.P("Higher is better. Measures return per unit of risk.", className="small text-muted")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sortino Ratio", className="card-title"),
                    html.H3(f"{metrics['sortino_ratio']:.2f}", className="text-primary"),
                    html.P("Focuses on downside risk. Higher is better.", className="small text-muted")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Max Drawdown", className="card-title"),
                    html.H3(f"{metrics['max_drawdown']:.2f}%", className="text-danger"),
                    html.P("Maximum decline from peak. Lower is better.", className="small text-muted")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Volatility (Ann.)", className="card-title"),
                    html.H3(f"{metrics['volatility']:.2f}%", className="text-primary"),
                    html.P("Standard deviation of returns. Lower indicates less risk.", className="small text-muted")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Beta", className="card-title"),
                    html.H3(f"{metrics['beta']:.2f}", className="text-primary"),
                    html.P("Sensitivity to market. <1 is less volatile than market.", className="small text-muted")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Value at Risk (95%)", className="card-title"),
                    html.H3(f"{metrics['var_95']:.2f}%", className="text-danger"),
                    html.P("Maximum daily loss with 95% confidence.", className="small text-muted")
                ])
            ])
        ], width=4)
    ]
    
    # Create component
    component = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Portfolio Risk Assessment", className="card-title")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=gauge_fig, config={'displayModeBar': False})
                            ], width=4),
                            
                            dbc.Col([
                                html.H5(f"Risk Level: {risk_description['level']}", className=f"text-{risk_description['color']}"),
                                html.P(risk_description['text']),
                                dbc.Alert([
                                    html.Strong("Note: "),
                                    "Risk metrics are calculated based on historical data and may not predict future performance."
                                ], color="info", className="mt-3 mb-0")
                            ], width=8)
                        ]),
                        
                        html.Hr(),
                        
                        dbc.Row(metrics_cards, className="g-2 mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=drawdown_fig, config={'displayModeBar': False})
                            ], width=12)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ])
    
    return component