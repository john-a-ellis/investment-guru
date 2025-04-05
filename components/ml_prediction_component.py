# components/ml_prediction_component.py
"""
ML prediction component for the Investment Recommendation System.
Provides visualization and interaction with ML models for price prediction and trend analysis.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import custom modules for ML predictions and analysis
from modules.model_integration import ModelIntegration
from modules.trend_analysis import TrendAnalyzer
from modules.price_prediction import get_price_predictions
from modules.portfolio_utils import load_tracked_assets, load_portfolio

def create_ml_prediction_component():
    """
    Creates a component for ML-based investment prediction visualization and analysis
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H4("AI Investment Analysis", className="card-title"),
            html.P("Machine learning predictions and technical analysis", className="card-subtitle")
        ]),
        dbc.CardBody([
            dbc.Tabs([
                # Tab 1: Price Prediction for selected asset
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Asset"),
                            dbc.InputGroup([
                                dbc.Select(
                                    id="ml-asset-selector",
                                    placeholder="Select an asset to analyze",
                                ),
                                dbc.Button("Analyze", id="ml-analyze-button", color="primary")
                            ]),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Prediction Horizon"),
                            dbc.RadioItems(
                                id="ml-horizon-selector",
                                options=[
                                    {"label": "30 Days", "value": 30},
                                    {"label": "60 Days", "value": 60},
                                    {"label": "90 Days", "value": 90}
                                ],
                                value=30,
                                inline=True
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Spinner([
                        dcc.Graph(id="ml-prediction-chart"),
                    ], color="primary", type="border", fullscreen=False),
                    html.Div(id="ml-prediction-details", className="mt-3"),
                    dcc.Store(id="ml-prediction-data")
                ], label="Price Prediction"),
                
                # Tab 2: Technical Analysis
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner([
                                html.Div(id="trend-analysis-content")
                            ], color="primary", type="border")
                        ], width=12)
                    ])
                ], label="Technical Analysis"),
                
                # Tab 3: Portfolio Insights
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Generate Portfolio Insights", 
                                     id="ml-portfolio-insights-button", 
                                     color="success", 
                                     className="mb-3"),
                            dbc.Spinner([
                                html.Div(id="ml-portfolio-insights")
                            ], color="primary", type="border")
                        ], width=12)
                    ])
                ], label="Portfolio Insights"),
                
                # Tab 4: Model Training
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Asset to Train"),
                            dbc.InputGroup([
                                dbc.Select(
                                    id="ml-train-asset-selector",
                                    placeholder="Select an asset for model training",
                                ),
                                dbc.Button("Train Model", id="ml-train-button", color="warning")
                            ]),
                            html.Div(id="ml-training-status", className="mt-3"),
                            html.Hr(),
                            html.H5("Model Training Status"),
                            html.Div(id="ml-training-status-table")
                        ], width=12)
                    ])
                ], label="Model Training")
            ], id="ml-prediction-tabs")
        ]),
        dcc.Interval(
            id="ml-update-interval",
            interval=60000,  # Update every minute
            n_intervals=0
        )
    ])

def create_prediction_chart(prediction_data, historical_data=None):
    """
    Create a prediction chart based on ML model predictions
    
    Args:
        prediction_data (dict): Prediction data from ML models
        historical_data (DataFrame, optional): Historical price data for context
    
    Returns:
        Figure: Plotly figure with prediction chart
    """
    if not prediction_data or 'values' not in prediction_data:
        # Return empty figure if no prediction data
        fig = go.Figure()
        fig.update_layout(
            title="Price Prediction (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data if available
    if historical_data is not None and not historical_data.empty:
        # Use only the last 90 days of historical data for context
        recent_history = historical_data.tail(90)
        
        fig.add_trace(go.Scatter(
            x=recent_history.index,
            y=recent_history['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2C3E50', width=2)
        ))
    
    # Add predicted values
    predicted_dates = pd.to_datetime(prediction_data['dates'])
    predicted_values = prediction_data['values']
    
    fig.add_trace(go.Scatter(
        x=predicted_dates,
        y=predicted_values,
        mode='lines',
        name='Predicted Price',
        line=dict(color='#2980B9', width=3, dash='dash')
    ))
    
    # Add confidence intervals if available
    if 'confidence' in prediction_data:
        confidence = prediction_data['confidence']
        
        if confidence['upper'] is not None and confidence['lower'] is not None:
            upper_values = confidence['upper']
            lower_values = confidence['lower']
            
            fig.add_trace(go.Scatter(
                x=predicted_dates,
                y=upper_values,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=predicted_dates,
                y=lower_values,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(41, 128, 185, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
    
    # Current price and prediction annotation
    if historical_data is not None and not historical_data.empty:
        current_price = historical_data['Close'].iloc[-1]
        future_price = predicted_values[-1]
        expected_return = ((future_price / current_price) - 1) * 100
        
        # Add current price and expected return to title
        title_text = f"Price Prediction - <b>{prediction_data['symbol']}</b>"
        title_text += f" (Current: ${current_price:.2f}, Expected: ${future_price:.2f}, Return: {expected_return:.2f}%)"
        
        # Show expected return as positive or negative
        if expected_return > 0:
            title_color = "green"
        else:
            title_color = "red"
    else:
        title_text = f"Price Prediction - {prediction_data['symbol']}"
        title_color = "black"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color=title_color)
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_prediction_details(prediction_data, historical_data=None):
    """
    Create detailed analysis of price predictions
    
    Args:
        prediction_data (dict): Prediction data from ML models
        historical_data (DataFrame, optional): Historical price data for context
    
    Returns:
        Component: Dash component with prediction details
    """
    if not prediction_data:
        return html.Div("No prediction data available.")
    
    symbol = prediction_data.get('symbol', 'Unknown')
    model_type = prediction_data.get('model', 'Unknown')
    
    # Calculate expected return
    if historical_data is not None and not historical_data.empty:
        current_price = historical_data['Close'].iloc[-1]
        predicted_values = prediction_data.get('values', [])
        
        if predicted_values:
            future_price = predicted_values[-1]
            expected_return = ((future_price / current_price) - 1) * 100
            expected_return_str = f"{expected_return:.2f}%"
            expected_return_color = "success" if expected_return > 0 else "danger"
        else:
            expected_return_str = "N/A"
            expected_return_color = "secondary"
    else:
        current_price = "N/A"
        future_price = "N/A"
        expected_return_str = "N/A"
        expected_return_color = "secondary"
    
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H5("Prediction Summary", className="card-title"),
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("Asset: "), 
                            symbol
                        ]),
                        html.P([
                            html.Strong("Model: "), 
                            model_type.capitalize()
                        ]),
                        html.P([
                            html.Strong("Prediction Horizon: "), 
                            f"{len(prediction_data.get('dates', []))} days"
                        ])
                    ], width=6),
                    dbc.Col([
                        html.P([
                            html.Strong("Current Price: "), 
                            f"${current_price}" if current_price != "N/A" else current_price
                        ]),
                        html.P([
                            html.Strong("Predicted Price: "), 
                            f"${future_price}" if future_price != "N/A" else future_price
                        ]),
                        html.P([
                            html.Strong("Expected Return: "), 
                            html.Span(expected_return_str, className=f"text-{expected_return_color}")
                        ])
                    ], width=6)
                ]),
                dbc.Alert([
                    html.Strong("Investment Recommendation: "),
                    html.Span(
                        "Consider Buying" if expected_return_str != "N/A" and float(expected_return_str[:-1]) > 5 else
                        "Consider Selling" if expected_return_str != "N/A" and float(expected_return_str[:-1]) < -5 else
                        "Hold/Neutral",
                        className=f"text-{'success' if expected_return_str != 'N/A' and float(expected_return_str[:-1]) > 5 else 'danger' if expected_return_str != 'N/A' and float(expected_return_str[:-1]) < -5 else 'warning'}"
                    )
                ], color="info"),
                html.P([
                    html.Small("Disclaimer: These predictions are based on historical data and machine learning models. Actual market performance may vary. Always conduct your own research before making investment decisions.")
                ], className="text-muted mt-3")
            ])
        ])
    ])

def create_trend_analysis_display(analysis_data):
    """
    Create trend analysis visualization for a specific asset
    
    Args:
        analysis_data (dict): Trend analysis data
    
    Returns:
        Component: Dash component with trend analysis
    """
    if not analysis_data:
        return html.Div("No trend analysis data available.")
    
    # Extract data from analysis
    trend = analysis_data.get('trend', {})
    overall_trend = trend.get('overall_trend', 'unknown')
    trend_strength = trend.get('trend_strength', 0)
    
    # Get support and resistance levels
    support_resistance = analysis_data.get('support_resistance', {})
    support_levels = support_resistance.get('support', [])
    resistance_levels = support_resistance.get('resistance', [])
    
    # Get pattern information
    patterns = analysis_data.get('patterns', {}).get('patterns', [])
    
    # Get breakout prediction
    breakout = analysis_data.get('breakout', {})
    breakout_prediction = breakout.get('prediction', 'neutral')
    breakout_confidence = breakout.get('confidence', 0)
    
    # Get market regime
    market_regime = analysis_data.get('market_regime', {})
    regime = market_regime.get('regime', 'unknown')
    
    # Determine colors based on trend
    trend_color = "success" if "bull" in overall_trend else "danger" if "bear" in overall_trend else "warning"
    breakout_color = "success" if breakout_prediction == "bullish" else "danger" if breakout_prediction == "bearish" else "warning"
    regime_color = "success" if "bull" in regime else "danger" if "bear" in regime else "warning"
    
    # Create cards for each analysis section
    trend_card = dbc.Card([
        dbc.CardHeader("Trend Analysis"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        "Overall Trend: ",
                        html.Span(
                            overall_trend.replace("_", " ").title(),
                            className=f"text-{trend_color}"
                        )
                    ]),
                    html.P(f"Trend Strength: {trend_strength:.1f}%"),
                    html.P([
                        "RSI: ",
                        html.Span(
                            f"{trend.get('details', {}).get('rsi_value', 0):.1f}",
                            className=f"{'text-danger' if trend.get('details', {}).get('rsi_value', 0) > 70 else 'text-success' if trend.get('details', {}).get('rsi_value', 0) < 30 else ''}"
                        ),
                        " ",
                        html.Small(
                            f"({trend.get('details', {}).get('rsi_trend', 'neutral')})",
                            className="text-muted"
                        )
                    ]),
                    html.P([
                        "MACD Trend: ",
                        html.Span(
                            trend.get('details', {}).get('macd_trend', 'neutral'),
                            className=f"{'text-success' if trend.get('details', {}).get('macd_trend', 'neutral') == 'bullish' else 'text-danger' if trend.get('details', {}).get('macd_trend', 'neutral') == 'bearish' else ''}"
                        )
                    ])
                ], width=6),
                dbc.Col([
                    # Trend strength gauge
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=trend_strength,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Trend Strength"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#18BC9C" if "bull" in overall_trend else "#E74C3C" if "bear" in overall_trend else "#F39C12"},
                                    'steps': [
                                        {'range': [0, 33], 'color': "#F5F5F5"},
                                        {'range': [33, 66], 'color': "#EEEEEE"},
                                        {'range': [66, 100], 'color': "#E8E8E8"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 2},
                                        'thickness': 0.75,
                                        'value': trend_strength
                                    }
                                }
                            )),
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Support and resistance table
    levels_card = dbc.Card([
        dbc.CardHeader("Support & Resistance Levels"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Support Levels"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Price"),
                            html.Th("Strength"),
                            html.Th("Distance")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"${level['price']:.2f}"),
                                html.Td(f"{level['strength']:.1f}%"),
                                html.Td(f"{level['distance_pct']:.2f}%")
                            ]) for level in support_levels[:3]  # Show top 3 support levels
                        ])
                    ], size="sm", bordered=True, striped=True)
                ], width=6),
                dbc.Col([
                    html.H6("Resistance Levels"),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Price"),
                            html.Th("Strength"),
                            html.Th("Distance")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"${level['price']:.2f}"),
                                html.Td(f"{level['strength']:.1f}%"),
                                html.Td(f"{level['distance_pct']:.2f}%")
                            ]) for level in resistance_levels[:3]  # Show top 3 resistance levels
                        ])
                    ], size="sm", bordered=True, striped=True)
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Chart patterns card
    patterns_card = dbc.Card([
        dbc.CardHeader("Chart Patterns"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Pattern"),
                    html.Th("Type"),
                    html.Th("Strength"),
                    html.Th("Date")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(pattern['name']),
                        html.Td(
                            html.Span(
                                pattern['type'].capitalize(),
                                className=f"text-{'success' if pattern['type'] == 'bullish' else 'danger'}"
                            )
                        ),
                        html.Td(f"{pattern['strength']:.1f}%"),
                        html.Td(f"{pattern['date']} ({pattern['days_ago']} days ago)")
                    ]) for pattern in patterns[:3]  # Show top 3 patterns
                ] if patterns else [html.Tr([html.Td("No patterns detected", colSpan=4)])])
            ], size="sm", bordered=True, striped=True)
        ])
    ], className="mb-3")
    
    # Breakout prediction card
    breakout_card = dbc.Card([
        dbc.CardHeader("Breakout Prediction"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        "Prediction: ",
                        html.Span(
                            breakout_prediction.capitalize(),
                            className=f"text-{breakout_color}"
                        )
                    ]),
                    html.P(f"Confidence: {breakout_confidence:.1f}%"),
                    html.P(f"Market Regime: {regime.replace('_', ' ').title()}", className=f"text-{regime_color}")
                ], width=6),
                dbc.Col([
                    # Breakout confidence gauge
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=breakout_confidence,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Breakout Confidence"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#18BC9C" if breakout_prediction == "bullish" else "#E74C3C" if breakout_prediction == "bearish" else "#F39C12"},
                                    'steps': [
                                        {'range': [0, 33], 'color': "#F5F5F5"},
                                        {'range': [33, 66], 'color': "#EEEEEE"},
                                        {'range': [66, 100], 'color': "#E8E8E8"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 2},
                                        'thickness': 0.75,
                                        'value': breakout_confidence
                                    }
                                }
                            )),
                            config={'displayModeBar': False},
                            style={'height': '200px'}
                        )
                    ])
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Combine all cards
    return html.Div([
        trend_card,
        breakout_card,
        levels_card,
        patterns_card
    ])

def create_portfolio_insights(recommendations):
    """
    Create portfolio insights based on ML recommendations
    
    Args:
        recommendations (dict): Portfolio recommendations from ML models
    
    Returns:
        Component: Dash component with portfolio insights
    """
    if not recommendations:
        return html.Div("No portfolio insights available.")
    
    # Extract recommendations
    buy_recs = recommendations.get('buy', [])
    sell_recs = recommendations.get('sell', [])
    portfolio_score = recommendations.get('portfolio_score', 0)
    
    # Portfolio score card
    score_card = dbc.Card([
        dbc.CardHeader("Portfolio Health Score"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2(f"{portfolio_score:.1f}/100", className=f"text-{'success' if portfolio_score >= 70 else 'warning' if portfolio_score >= 50 else 'danger'}"),
                    html.P("Based on ML analysis of current holdings and market conditions")
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        figure=go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=portfolio_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Portfolio Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': 
                                      "#18BC9C" if portfolio_score >= 70 
                                      else "#F39C12" if portfolio_score >= 50 
                                      else "#E74C3C"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#FFEEEE"},
                                    {'range': [50, 70], 'color': "#FFFFEE"},
                                    {'range': [70, 100], 'color': "#EEFFEE"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 2},
                                    'thickness': 0.75,
                                    'value': portfolio_score
                                }
                            }
                        )),
                        config={'displayModeBar': False},
                        style={'height': '200px'}
                    )
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Buy recommendations card
    buy_card = dbc.Card([
        dbc.CardHeader("Buy Recommendations"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Confidence"),
                    html.Th("Expected Return"),
                    html.Th("Action")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(rec['symbol']),
                        html.Td(f"{rec['confidence']:.1f}%"),
                        html.Td(f"{rec['expected_return']:.2f}%", className=f"text-{'success' if rec['expected_return'] > 0 else 'danger'}"),
                        html.Td(
                            dbc.Button("Buy", color="success", size="sm", id={"type": "buy-rec-button", "symbol": rec['symbol']})
                        )
                    ]) for rec in buy_recs[:5]  # Show top 5 buy recommendations
                ] if buy_recs else [html.Tr([html.Td("No buy recommendations", colSpan=4)])])
            ], size="sm", bordered=True, striped=True, hover=True)
        ])
    ], className="mb-3")
    
    # Sell recommendations card
    sell_card = dbc.Card([
        dbc.CardHeader("Sell Recommendations"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Confidence"),
                    html.Th("Expected Return"),
                    html.Th("Action")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(rec['symbol']),
                        html.Td(f"{rec['confidence']:.1f}%"),
                        html.Td(f"{rec['expected_return']:.2f}%", className=f"text-{'success' if rec['expected_return'] > 0 else 'danger'}"),
                        html.Td(
                            dbc.Button("Sell", color="danger", size="sm", id={"type": "sell-rec-button", "symbol": rec['symbol']})
                        )
                    ]) for rec in sell_recs[:5]  # Show top 5 sell recommendations
                ] if sell_recs else [html.Tr([html.Td("No sell recommendations", colSpan=4)])])
            ], size="sm", bordered=True, striped=True, hover=True)
        ])
    ], className="mb-3")
    
    # Combine all cards
    return html.Div([
        score_card,
        buy_card,
        sell_card,
        dbc.Alert([
            html.Strong("Note: "), 
            "These recommendations are generated using machine learning models and technical analysis. Always conduct your own research before making investment decisions."
        ], color="info", className="mt-3")
    ])

def create_training_status_table(status_dict):
    """
    Create a table showing model training status
    
    Args:
        status_dict (dict): Dictionary of model training status
    
    Returns:
        Component: Dash component with training status table
    """
    if not status_dict:
        return html.Div("No model training status available.")
    
    # Convert status dictionary to list for table display
    status_list = []
    for symbol, data in status_dict.items():
        status_list.append({
            'symbol': symbol,
            'status': data.get('status', 'unknown'),
            'last_updated': data.get('last_updated', 'never')
        })
    
    # Sort by symbol
    status_list.sort(key=lambda x: x['symbol'])
    
    # Create table
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Symbol"),
            html.Th("Status"),
            html.Th("Last Updated")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(status['symbol']),
                html.Td(
                    html.Span(
                        status['status'].replace('_', ' ').title(),
                        className=f"text-{'success' if status['status'] == 'completed' else 'warning' if status['status'] == 'in_progress' else 'danger' if status['status'] == 'failed' else 'secondary'}"
                    )
                ),
                html.Td(status['last_updated'])
            ]) for status in status_list
        ])
    ], bordered=True, striped=True, hover=True)

# Define callbacks for main.py

def register_ml_prediction_callbacks(app):
    """
    Register callbacks for ML prediction component
    
    Args:
        app: Dash app instance
    """
    # Initialize model integration
    model_integration = ModelIntegration()
    
    # Populate asset selector with portfolio and tracked assets
    @app.callback(
        [Output("ml-asset-selector", "options"),
         Output("ml-train-asset-selector", "options")],
        [Input("ml-update-interval", "n_intervals")]
    )
    def update_asset_options(n_intervals):
        # Get tracked assets
        tracked_assets = load_tracked_assets()
        
        # Get portfolio assets
        portfolio = load_portfolio()
        portfolio_symbols = set()
        for investment_id, details in portfolio.items():
            symbol = details.get("symbol", "")
            if symbol:
                portfolio_symbols.add(symbol)
        
        # Combine all symbols
        all_symbols = set(tracked_assets.keys()) | portfolio_symbols
        
        # Create options for dropdown
        options = []
        for symbol in sorted(all_symbols):
            # Try to get a name from tracked assets
            name = tracked_assets.get(symbol, {}).get("name", symbol)
            options.append({"label": f"{symbol} - {name}", "value": symbol})
        
        return options, options
    
    # Generate price prediction when analyze button is clicked
    @app.callback(
        [Output("ml-prediction-chart", "figure"),
         Output("ml-prediction-details", "children"),
         Output("ml-prediction-data", "data")],
        [Input("ml-analyze-button", "n_clicks")],
        [State("ml-asset-selector", "value"),
         State("ml-horizon-selector", "value")]
    )
    def update_prediction(n_clicks, symbol, days):
        # Check if button click is triggered
        if n_clicks is None or not symbol:
            # Return empty components on initial load
            return go.Figure(), None, None
        
        try:
            # Import FMP API
            from modules.fmp_api import fmp_api
            
            # Get historical data for context
            historical_data = fmp_api.get_historical_price(symbol, period="1y")
            
            # Get price predictions
            prediction_data = get_price_predictions(symbol, days=days)
            
            if prediction_data:
                # Create prediction chart
                chart = create_prediction_chart(prediction_data, historical_data)
                
                # Create prediction details
                details = create_prediction_details(prediction_data, historical_data)
                
                return chart, details, prediction_data
            else:
                # Return empty figure with error message
                fig = go.Figure()
                fig.update_layout(
                    title=f"Error: Could not generate predictions for {symbol}",
                    template="plotly_white"
                )
                
                return fig, html.Div(f"Could not generate predictions for {symbol}. Try training a model first."), None
        
        except Exception as e:
            # Return error message
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: {str(e)}",
                template="plotly_white"
            )
            
            return fig, html.Div(f"Error: {str(e)}"), None
    
    # Generate trend analysis when asset selected
    @app.callback(
        Output("trend-analysis-content", "children"),
        [Input("ml-analyze-button", "n_clicks")],
        [State("ml-asset-selector", "value")]
    )
    def update_trend_analysis(n_clicks, symbol):
        # Check if button click is triggered
        if n_clicks is None or not symbol:
            # Return empty component on initial load
            return None
        
        try:
            # Get comprehensive asset analysis
            analysis = model_integration.get_asset_analysis(symbol)
            
            if analysis:
                # Create trend analysis display
                return create_trend_analysis_display(analysis)
            else:
                return html.Div(f"Could not generate analysis for {symbol}.")
        
        except Exception as e:
            # Return error message
            return html.Div(f"Error generating trend analysis: {str(e)}")
    
    # Generate portfolio insights
    @app.callback(
        Output("ml-portfolio-insights", "children"),
        Input("ml-portfolio-insights-button", "n_clicks")
    )
    def update_portfolio_insights(n_clicks):
        # Check if button click is triggered
        if n_clicks is None:
            # Return empty component on initial load
            return None
        
        try:
            # Get portfolio recommendations
            recommendations = model_integration.get_portfolio_recommendations()
            
            # Create portfolio insights
            return create_portfolio_insights(recommendations)
        
        except Exception as e:
            # Return error message
            return html.Div(f"Error generating portfolio insights: {str(e)}")
    
    # Train model for selected asset
    @app.callback(
        Output("ml-training-status", "children"),
        [Input("ml-train-button", "n_clicks")],
        [State("ml-train-asset-selector", "value")]
    )
    def train_model(n_clicks, symbol):
        # Check if button click is triggered
        if n_clicks is None or not symbol:
            # Return empty component on initial load
            return None
        
        try:
            # Start model training
            status = model_integration.train_models_for_symbol(symbol, lookback_period="2y")
            
            # Return status message
            return dbc.Alert(f"Training status: {status}", color="info")
        
        except Exception as e:
            # Return error message
            return dbc.Alert(f"Error: {str(e)}", color="danger")
    
    # Update training status table
    @app.callback(
        Output("ml-training-status-table", "children"),
        [Input("ml-update-interval", "n_intervals"),
         Input("ml-training-status", "children")]
    )
    def update_training_status(n_intervals, training_status):
        try:
            # Get model training status for all symbols
            status_dict = model_integration.get_model_training_status()
            
            # Create status table
            return create_training_status_table(status_dict)
        
        except Exception as e:
            # Return error message
            return html.Div(f"Error retrieving training status: {str(e)}")
    
    # Handle buy recommendation clicks
    @app.callback(
        Output("ml-portfolio-insights", "children", allow_duplicate=True),
        Input({"type": "buy-rec-button", "symbol": dash.ALL}, "n_clicks"),
        State({"type": "buy-rec-button", "symbol": dash.ALL}, "id"),
        prevent_initial_call='initial_duplicate'
    )
    def handle_buy_recommendation(n_clicks_list, button_ids):
        # Check if any button was clicked
        if not any(n for n in n_clicks_list if n):
            raise dash.exceptions.PreventUpdate
        
        # Find which button was clicked
        clicked_idx = next((i for i, n in enumerate(n_clicks_list) if n), None)
        if clicked_idx is None:
            raise dash.exceptions.PreventUpdate
        
        # Get the symbol
        symbol = button_ids[clicked_idx]["symbol"]
        
        # Record transaction (placeholder - you should integrate with your transaction system)
        # This is where you would integrate with your existing transaction_recorder functionality
        try:
            # Import FMP API to get current price
            from modules.fmp_api import fmp_api
            quote = fmp_api.get_quote(symbol)
            
            if quote and 'price' in quote:
                current_price = quote['price']
                
                # Use the record_transaction function from your portfolio_utils
                from modules.portfolio_utils import record_transaction
                
                # Default to buying 1 share - in a real app, you'd want a quantity input
                success = record_transaction(symbol, "buy", current_price, 1)
                
                if success:
                    return dbc.Alert(f"Successfully added buy transaction for {symbol}", color="success")
                else:
                    return dbc.Alert(f"Failed to add transaction for {symbol}", color="danger")
            else:
                return dbc.Alert(f"Could not get current price for {symbol}", color="warning")
        
        except Exception as e:
            return dbc.Alert(f"Error creating transaction: {str(e)}", color="danger")
    
    # Handle sell recommendation clicks
    @app.callback(
        Output("ml-portfolio-insights", "children", allow_duplicate=True),
        Input({"type": "sell-rec-button", "symbol": dash.ALL}, "n_clicks"),
        State({"type": "sell-rec-button", "symbol": dash.ALL}, "id"),
        prevent_initial_call='initial_duplicate'
    )
    def handle_sell_recommendation(n_clicks_list, button_ids):
        # Check if any button was clicked
        if not any(n for n in n_clicks_list if n):
            raise dash.exceptions.PreventUpdate
        
        # Find which button was clicked
        clicked_idx = next((i for i, n in enumerate(n_clicks_list) if n), None)
        if clicked_idx is None:
            raise dash.exceptions.PreventUpdate
        
        # Get the symbol
        symbol = button_ids[clicked_idx]["symbol"]
        
        # Record transaction (placeholder - you should integrate with your transaction system)
        try:
            # Import FMP API to get current price
            from modules.fmp_api import fmp_api
            quote = fmp_api.get_quote(symbol)
            
            if quote and 'price' in quote:
                current_price = quote['price']
                
                # Use the record_transaction function from your portfolio_utils
                from modules.portfolio_utils import record_transaction
                
                # Default to selling 1 share - in a real app, you'd want a quantity input
                success = record_transaction(symbol, "sell", current_price, 1)
                
                if success:
                    return dbc.Alert(f"Successfully added sell transaction for {symbol}", color="success")
                else:
                    return dbc.Alert(f"Failed to add transaction for {symbol}", color="danger")
            else:
                return dbc.Alert(f"Could not get current price for {symbol}", color="warning")
        
        except Exception as e:
            return dbc.Alert(f"Error creating transaction: {str(e)}", color="danger")