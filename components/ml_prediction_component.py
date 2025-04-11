# components/ml_prediction_component.py
"""
ML prediction component for the Investment Recommendation System.
Provides visualization and interaction with ML models for price prediction, trend analysis,
portfolio insights, and model management.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx # Added ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging # Added
import traceback # Added for error logging
import json # Added
from dash.exceptions import PreventUpdate

# Import custom modules for ML predictions and analysis
from modules.model_integration import ModelIntegration
from modules.trend_analysis import TrendAnalyzer
from modules.price_prediction import get_price_predictions # Keep original prediction function
from modules.portfolio_utils import load_tracked_assets, load_portfolio, record_transaction # Added record_transaction
from modules.data_provider import data_provider

# Import the function to get trained model data
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent dir
try:
    from modules.db_utils import get_trained_models_data
except ImportError:
    print("ERROR: Could not import get_trained_models_data from modules.db_utils")
    # Define a dummy function to avoid crashing the app layout
    def get_trained_models_data():
        print("WARNING: Using dummy get_trained_models_data function.")
        return pd.DataFrame()

# Configure logging
logger = logging.getLogger(__name__)

# --- HELPER FUNCTION TO CREATE THE TRAINED MODELS TABLE ---
def create_trained_models_table(models_df):
    """Creates a dbc.Table from the trained models DataFrame."""
    if not isinstance(models_df, pd.DataFrame) or models_df.empty:
        return dbc.Alert("No trained model data found in the database.", color="info")

    metrics_to_display = ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'r2_score', 'rmse']

    header = [
        html.Thead(html.Tr([
            html.Th("Filename"), html.Th("Symbol"), html.Th("Type"),
            html.Th("Training Date"), html.Th("Key Metrics"),
            html.Th("Notes", style={'maxWidth': '200px'})
        ]))
    ]
    rows = []
    for index, row in models_df.iterrows():
        metrics_dict = row.get('metrics', {})
        metrics_display_items = []
        if isinstance(metrics_dict, dict):
            for key in metrics_to_display:
                if key in metrics_dict:
                    value = metrics_dict[key]
                    if isinstance(value, (float, int)):
                         metrics_display_items.append(html.Li(f"{key.replace('_',' ').title()}: {value:.4f}"))
                    else:
                         metrics_display_items.append(html.Li(f"{key.replace('_',' ').title()}: {value}"))
            if not metrics_display_items and metrics_dict and 'error' not in metrics_dict:
                 metrics_display_items.append(html.Li(f"Other metrics available ({len(metrics_dict)} total)"))
            elif not metrics_dict or 'error' in metrics_dict:
                 metrics_display_items.append(html.Li("N/A"))
            elif len(metrics_dict) > len(metrics_display_items):
                 metrics_display_items.append(html.Li(f"... ({len(metrics_dict) - len(metrics_display_items)} more)"))
        elif metrics_dict:
             metrics_display_items.append(html.Li(f"Error: {metrics_dict}"))
        else:
             metrics_display_items.append(html.Li("N/A"))

        metrics_cell = html.Td(html.Ul(metrics_display_items, style={'paddingLeft': '15px', 'marginBottom': '0', 'listStyleType': 'none'}))
        notes = row.get('notes', '')
        truncated_notes = (notes[:100] + '...') if notes and len(notes) > 100 else notes
        rows.append(html.Tr([
            html.Td(row.get('model_filename', 'N/A')), html.Td(row.get('symbol', 'N/A')),
            html.Td(row.get('model_type', 'N/A')), html.Td(row.get('training_date', 'N/A')),
            metrics_cell, html.Td(truncated_notes, title=notes if truncated_notes != notes else '')
        ]))
    body = [html.Tbody(rows)]
    return dbc.Table(header + body, bordered=True, striped=True, hover=True, responsive=True, size="sm")

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
    Create a prediction chart based on ML model predictions with improved error handling.
    (Copied from original, assuming it's correct)
    """
    if not prediction_data:
        fig = go.Figure()
        fig.update_layout(title="Price Prediction (No Prediction Data Available)", template="plotly_white")
        return fig

    if 'values' not in prediction_data or not prediction_data.get('values'):
        fig = go.Figure()
        fig.update_layout(title=f"Price Prediction (No Valid Values for {prediction_data.get('symbol', 'Unknown')})", template="plotly_white")
        return fig

    fig = go.Figure()
    if historical_data is not None and not historical_data.empty:
        recent_history = historical_data.tail(90)
        if 'close' in recent_history.columns: # Use lowercase
            fig.add_trace(go.Scatter(x=recent_history.index, y=recent_history['close'], mode='lines', name='Historical Price', line=dict(color='#2C3E50', width=2)))
        else: print("Warning: 'close' column not found in historical data for prediction chart.")

    try:
        if 'dates' in prediction_data and prediction_data['dates']:
            predicted_dates = pd.to_datetime(prediction_data['dates']) if isinstance(prediction_data['dates'][0], str) else prediction_data['dates']
            values = prediction_data['values']
            predicted_values = [float(val) for val in values if pd.notna(val) and val is not None]

            if predicted_values and len(predicted_dates) >= len(predicted_values):
                valid_dates = predicted_dates[:len(predicted_values)]
                fig.add_trace(go.Scatter(x=valid_dates, y=predicted_values, mode='lines', name='Predicted Price', line=dict(color='#2980B9', width=3, dash='dash')))

                if 'confidence' in prediction_data and prediction_data['confidence']:
                    confidence = prediction_data['confidence']
                    if ('upper' in confidence and confidence['upper'] and 'lower' in confidence and confidence['lower']):
                        upper_values = [float(val) for val in confidence['upper'] if pd.notna(val) and val is not None]
                        lower_values = [float(val) for val in confidence['lower'] if pd.notna(val) and val is not None]
                        if upper_values and lower_values:
                            min_len = min(len(valid_dates), len(upper_values), len(lower_values))
                            valid_dates_ci, upper_values, lower_values = valid_dates[:min_len], upper_values[:min_len], lower_values[:min_len]
                            fig.add_trace(go.Scatter(x=valid_dates_ci, y=upper_values, mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
                            fig.add_trace(go.Scatter(x=valid_dates_ci, y=lower_values, mode='lines', name='Lower Bound', line=dict(width=0), fillcolor='rgba(41, 128, 185, 0.2)', fill='tonexty', showlegend=False))
    except Exception as e:
        print(f"Error adding predictions to chart: {e}")
        traceback.print_exc()

    title_color = "black"
    title_text = f"Price Prediction"
    current_price = None
    if historical_data is not None and not historical_data.empty and 'close' in historical_data.columns:
        current_price = historical_data['close'].iloc[-1]
        prediction_horizon = len(prediction_data.get('values', []))
        title_text = f"Price Prediction ({prediction_horizon}-Day Horizon) - <b>{prediction_data.get('symbol', '')}</b>"
        predicted_values_list = [v for v in prediction_data.get('values', []) if pd.notna(v) and v is not None]
        if predicted_values_list and current_price is not None and current_price > 0:
            future_price = predicted_values_list[-1]
            expected_return = ((future_price / current_price) - 1) * 100
            title_text += f" (Current: ${current_price:.2f}, Expected: ${future_price:.2f}, Return: {expected_return:.2f}%)"
            title_color = "green" if expected_return > 0 else "red"
        else:
            print(f"Warning: Cannot calculate expected return. predicted_values empty: {not predicted_values_list}, current_price: {current_price}")
    else:
        print("Warning: 'close' column not found or historical data empty for title generation.")
        prediction_horizon = len(prediction_data.get('values', []))
        title_text = f"Price Prediction ({prediction_horizon}-Day Horizon) - <b>{prediction_data.get('symbol', '')}</b> (Current Price Unavailable)"

    fig.update_layout(
        title=dict(text=title_text, font=dict(color=title_color)),
        xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig



def create_prediction_details(prediction_data, historical_data=None):
    """
    Create detailed analysis of price predictions with improved error handling.
    (Copied from original, assuming it's correct)
    """
    if not prediction_data: return html.Div("No prediction data available.")
    symbol = prediction_data.get('symbol', 'Unknown')
    model_type = prediction_data.get('model', 'Unknown')
    current_price_val = "N/A"
    future_price_val = "N/A"
    expected_return_str = "N/A"
    expected_return_color = "secondary"
    expected_return = 0
    recommendation = "Hold/Neutral"
    rec_color = "warning"

    if historical_data is not None and not historical_data.empty and 'close' in historical_data.columns:
        current_price_val = historical_data['close'].iloc[-1]
        predicted_values = [float(val) for val in prediction_data.get('values', []) if pd.notna(val) and val is not None]
        if predicted_values and current_price_val > 0:
            future_price_val = predicted_values[-1]
            expected_return = ((future_price_val / current_price_val) - 1) * 100
            expected_return_str = f"{expected_return:.2f}%"
            expected_return_color = "success" if expected_return > 0 else "danger"
            if expected_return > 5: recommendation, rec_color = "Consider Buying", "success"
            elif expected_return < -5: recommendation, rec_color = "Consider Selling", "danger"
        else:
            print("Warning: Cannot calculate expected return in details.")
            current_price_val = "N/A" # Reset if calculation failed
    else:
        print("Warning: 'close' column not found or historical data empty for prediction details.")

    prediction_horizon = len([v for v in prediction_data.get('values', []) if pd.notna(v) and v is not None])

    return html.Div([dbc.Card([dbc.CardBody([
        html.H5("Prediction Summary", className="card-title"),
        dbc.Row([
            dbc.Col([
                html.P([html.Strong("Asset: "), symbol]),
                html.P([html.Strong("Model: "), model_type.capitalize() if model_type else "Unknown"]),
                html.P([html.Strong("Prediction Horizon: "), f"{prediction_horizon} days"])
            ], width=6),
            dbc.Col([
                html.P([html.Strong("Current Price: "), f"${current_price_val:.2f}" if isinstance(current_price_val, (int, float)) else current_price_val]),
                html.P([html.Strong("Predicted Price: "), f"${future_price_val:.2f}" if isinstance(future_price_val, (int, float)) else future_price_val]),
                html.P([html.Strong("Expected Return: "), html.Span(expected_return_str, className=f"text-{expected_return_color}")])
            ], width=6)
        ]),
        dbc.Alert([html.Strong("Investment Recommendation: "), html.Span(recommendation, className=f"text-{rec_color}")], color="info"),
        html.P([html.Small("Disclaimer: These predictions are based on historical data and machine learning models. Actual market performance may vary. Always conduct your own research before making investment decisions.")], className="text-muted mt-3")
    ])])])


def create_trend_analysis_display(analysis_data):
    """
    Create trend analysis visualization for a specific asset.
    (Copied from original, assuming it's correct and TrendAnalyzer provides the expected dict structure)
    """
    if not analysis_data: return html.Div("No trend analysis data available.")
    trend = analysis_data.get('trend', {})
    overall_trend = trend.get('overall_trend', 'unknown')
    trend_strength = trend.get('trend_strength', 0)
    support_resistance = analysis_data.get('support_resistance', {})
    support_levels = support_resistance.get('support', [])
    resistance_levels = support_resistance.get('resistance', [])
    patterns = analysis_data.get('patterns', {}).get('patterns', [])
    breakout = analysis_data.get('breakout', {})
    breakout_prediction = breakout.get('prediction', 'neutral')
    breakout_confidence = breakout.get('confidence', 0)
    market_regime = analysis_data.get('market_regime', {})
    regime = market_regime.get('regime', 'unknown')
    trend_color = "success" if "bull" in overall_trend else "danger" if "bear" in overall_trend else "warning"
    breakout_color = "success" if breakout_prediction == "bullish" else "danger" if breakout_prediction == "bearish" else "warning"
    regime_color = "success" if "bull" in regime else "danger" if "bear" in regime else "warning"

    trend_card = dbc.Card([dbc.CardHeader("Trend Analysis"), dbc.CardBody([dbc.Row([
        dbc.Col([
            html.H5(["Overall Trend: ", html.Span(overall_trend.replace("_", " ").title(), className=f"text-{trend_color}")]),
            html.P(f"Trend Strength: {trend_strength:.1f}%"),
            html.P(["RSI: ", html.Span(f"{trend.get('details', {}).get('rsi_value', 0):.1f}", className=f"{'text-danger' if trend.get('details', {}).get('rsi_value', 0) > 70 else 'text-success' if trend.get('details', {}).get('rsi_value', 0) < 30 else ''}"), " ", html.Small(f"({trend.get('details', {}).get('rsi_trend', 'neutral')})", className="text-muted")]),
            html.P(["MACD Trend: ", html.Span(trend.get('details', {}).get('macd_trend', 'neutral'), className=f"{'text-success' if trend.get('details', {}).get('macd_trend', 'neutral') == 'bullish' else 'text-danger' if trend.get('details', {}).get('macd_trend', 'neutral') == 'bearish' else ''}")])
        ], width=6),
        dbc.Col([html.Div([dcc.Graph(figure=go.Figure(go.Indicator(mode="gauge+number", value=trend_strength, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Trend Strength"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#18BC9C" if "bull" in overall_trend else "#E74C3C" if "bear" in overall_trend else "#F39C12"}, 'steps': [{'range': [0, 33], 'color': "#F5F5F5"}, {'range': [33, 66], 'color': "#EEEEEE"}, {'range': [66, 100], 'color': "#E8E8E8"}], 'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': trend_strength}})), config={'displayModeBar': False})])], width=6)
    ])])], className="mb-3")

    levels_card = dbc.Card([dbc.CardHeader("Support & Resistance Levels"), dbc.CardBody([dbc.Row([
        dbc.Col([html.H6("Support Levels"), dbc.Table([html.Thead(html.Tr([html.Th("Price"), html.Th("Strength"), html.Th("Distance")])), html.Tbody([html.Tr([html.Td(f"${level['price']:.2f}"), html.Td(f"{level['strength']:.1f}%"), html.Td(f"{level['distance_pct']:.2f}%")]) for level in support_levels[:3]])], size="sm", bordered=True, striped=True)], width=6),
        dbc.Col([html.H6("Resistance Levels"), dbc.Table([html.Thead(html.Tr([html.Th("Price"), html.Th("Strength"), html.Th("Distance")])), html.Tbody([html.Tr([html.Td(f"${level['price']:.2f}"), html.Td(f"{level['strength']:.1f}%"), html.Td(f"{level['distance_pct']:.2f}%")]) for level in resistance_levels[:3]])], size="sm", bordered=True, striped=True)], width=6)
    ])])], className="mb-3")

    patterns_card = dbc.Card([dbc.CardHeader("Chart Patterns"), dbc.CardBody([dbc.Table([
        html.Thead(html.Tr([html.Th("Pattern"), html.Th("Type"), html.Th("Strength"), html.Th("Date")])),
        html.Tbody([html.Tr([html.Td(pattern['name']), html.Td(html.Span(pattern['type'].capitalize(), className=f"text-{'success' if pattern['type'] == 'bullish' else 'danger'}")), html.Td(f"{pattern['strength']:.1f}%"), html.Td(f"{pattern['date']} ({pattern['days_ago']} days ago)")]) for pattern in patterns[:3]] if patterns else [html.Tr([html.Td("No patterns detected", colSpan=4)])])
    ], size="sm", bordered=True, striped=True)])], className="mb-3")

    breakout_card = dbc.Card([dbc.CardHeader("Breakout Prediction"), dbc.CardBody([dbc.Row([
        dbc.Col([
            html.H5(["Prediction: ", html.Span(breakout_prediction.capitalize(), className=f"text-{breakout_color}")]),
            html.P(f"Confidence: {breakout_confidence:.1f}%"),
            html.P(f"Market Regime: {regime.replace('_', ' ').title()}", className=f"text-{regime_color}")
        ], width=6),
        dbc.Col([html.Div([dcc.Graph(figure=go.Figure(go.Indicator(mode="gauge+number", value=breakout_confidence, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Breakout Confidence"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#18BC9C" if breakout_prediction == "bullish" else "#E74C3C" if breakout_prediction == "bearish" else "#F39C12"}, 'steps': [{'range': [0, 33], 'color': "#F5F5F5"}, {'range': [33, 66], 'color': "#EEEEEE"}, {'range': [66, 100], 'color': "#E8E8E8"}], 'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': breakout_confidence}})), config={'displayModeBar': False})])], width=6)
    ])])], className="mb-3")

    return html.Div([trend_card, breakout_card, levels_card, patterns_card])

def create_portfolio_insights(recommendations):
    """
    Create portfolio insights based on ML recommendations.
    (Copied from original, assuming it's correct and ModelIntegration provides the expected dict structure)
    """
    if not recommendations: return html.Div("No portfolio insights available.")
    buy_recs = recommendations.get('buy', [])
    sell_recs = recommendations.get('sell', [])
    portfolio_score = recommendations.get('portfolio_score', 0)

    score_card = dbc.Card([dbc.CardHeader("Portfolio Health Score"), dbc.CardBody([dbc.Row([
        dbc.Col([html.H2(f"{portfolio_score:.1f}/100", className=f"text-{'success' if portfolio_score >= 70 else 'warning' if portfolio_score >= 50 else 'danger'}"), html.P("Based on ML analysis of current holdings and market conditions")], width=6),
        dbc.Col([dcc.Graph(figure=go.Figure(go.Indicator(mode="gauge+number", value=portfolio_score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Portfolio Score"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#18BC9C" if portfolio_score >= 70 else "#F39C12" if portfolio_score >= 50 else "#E74C3C"}, 'steps': [{'range': [0, 50], 'color': "#FFEEEE"}, {'range': [50, 70], 'color': "#FFFFEE"}, {'range': [70, 100], 'color': "#EEFFEE"}], 'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': portfolio_score}})), config={'displayModeBar': False})], width=6)
    ])])], className="mb-3")

    buy_card = dbc.Card([dbc.CardHeader("Buy Recommendations"), dbc.CardBody([dbc.Table([
        html.Thead(html.Tr([html.Th("Symbol"), html.Th("Confidence"), html.Th("Expected Return"), html.Th("Action")])),
        html.Tbody([html.Tr([html.Td(rec['symbol']), html.Td(f"{rec['confidence']:.1f}%"), html.Td(f"{rec['expected_return']:.2f}%", className=f"text-{'success' if rec['expected_return'] > 0 else 'danger'}"), html.Td(dbc.Button("Buy", color="success", size="sm", id={"type": "buy-rec-button", "symbol": rec['symbol']}))]) for rec in buy_recs[:5]] if buy_recs else [html.Tr([html.Td("No buy recommendations", colSpan=4)])])
    ], size="sm", bordered=True, striped=True, hover=True)])], className="mb-3")

    sell_card = dbc.Card([dbc.CardHeader("Sell Recommendations"), dbc.CardBody([dbc.Table([
        html.Thead(html.Tr([html.Th("Symbol"), html.Th("Confidence"), html.Th("Expected Return"), html.Th("Action")])),
        html.Tbody([html.Tr([html.Td(rec['symbol']), html.Td(f"{rec['confidence']:.1f}%"), html.Td(f"{rec['expected_return']:.2f}%", className=f"text-{'success' if rec['expected_return'] > 0 else 'danger'}"), html.Td(dbc.Button("Sell", color="danger", size="sm", id={"type": "sell-rec-button", "symbol": rec['symbol']}))]) for rec in sell_recs[:5]] if sell_recs else [html.Tr([html.Td("No sell recommendations", colSpan=4)])])
    ], size="sm", bordered=True, striped=True, hover=True)])], className="mb-3")

    return html.Div([score_card, buy_card, sell_card, dbc.Alert([html.Strong("Note: "), "These recommendations are generated using machine learning models and technical analysis. Always conduct your own research before making investment decisions."], color="info", className="mt-3")])


def create_training_status_table(status_dict):
    """
    Create a table showing model training status.
    (Copied from original, assuming it's correct and ModelIntegration provides the expected dict structure)
    """
    if not status_dict: return html.Div("No model training status available.")
    status_list = []
    for symbol, data in status_dict.items():
        if isinstance(data, dict): status_list.append({'symbol': symbol, 'status': data.get('status', 'unknown'), 'last_updated': data.get('last_updated', 'never'), 'metrics': data.get('metrics', {}), 'error': data.get('error', '')})
        elif isinstance(data, str): status_list.append({'symbol': symbol, 'status': data, 'last_updated': 'unknown', 'metrics': {}, 'error': ''}) # Handle simple string status
    status_list.sort(key=lambda x: x['symbol'])

    rows = []
    for status in status_list:
        status_val = status['status']
        status_color = {"completed": "success", "in_progress": "warning", "pending": "info", "failed": "danger", "unknown": "secondary", "not_started": "secondary"}.get(status_val, "secondary")
        metrics = status.get('metrics', {})
        metrics_text = ""
        if metrics and isinstance(metrics, dict): metrics_text = f"MAE: {metrics.get('mae', 0):.2f}, RMSE: {metrics.get('rmse', 0):.2f}"
        error_message = status.get('error', '')
        rows.append(html.Tr([
            html.Td(status['symbol']),
            html.Td(html.Span(status_val.replace('_', ' ').title(), className=f"text-{status_color}")),
            html.Td(status['last_updated']),
            html.Td(metrics_text),
            html.Td(error_message, className="text-danger" if error_message else "")
        ]))

    return dbc.Table([
        html.Thead(html.Tr([html.Th("Symbol"), html.Th("Status"), html.Th("Last Updated"), html.Th("Metrics"), html.Th("Error (if any)")])),
        html.Tbody(rows)
    ], bordered=True, striped=True, hover=True)

# Define callbacks for main.py

# --- MERGED COMPONENT LAYOUT FUNCTION ---
def create_ml_prediction_component():
    """
    Creates the merged ML Prediction component layout including all original tabs
    and the new trained model overview.
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H4("AI Investment Analysis & Model Management", className="card-title"), # Updated title
            html.P("Machine learning predictions, technical analysis, and model overview", className="card-subtitle") # Updated subtitle
        ]),
        dbc.CardBody([
            dbc.Tabs(id="ml-prediction-tabs", active_tab="price-prediction-tab", children=[ # Default to price prediction tab
                # Tab 1: Price Prediction (Original Structure)
                dbc.Tab(label="Price Prediction", tab_id="price-prediction-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Asset"),
                            dbc.InputGroup([
                                dbc.Select(id="ml-asset-selector", placeholder="Select an asset to analyze"),
                                dbc.Button("Analyze", id="ml-analyze-button", color="primary")
                            ]),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Prediction Horizon"),
                            dbc.RadioItems(
                                id="ml-horizon-selector",
                                options=[{"label": "30 Days", "value": 30}, {"label": "60 Days", "value": 60}, {"label": "90 Days", "value": 90}],
                                value=30, inline=True
                            )
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Spinner([dcc.Graph(id="ml-prediction-chart")], color="primary", type="border", fullscreen=False),
                    html.Div(id="ml-prediction-details", className="mt-3"),
                    dcc.Store(id="ml-prediction-data") # Keep store if needed by callbacks
                ]),

                # Tab 2: Technical Analysis (Original Structure)
                dbc.Tab(label="Technical Analysis", tab_id="technical-analysis-tab", children=[
                    dbc.Row([dbc.Col([dbc.Spinner([html.Div(id="trend-analysis-content")], color="primary", type="border")], width=12)])
                ]),

                # Tab 3: Portfolio Insights (Original Structure)
                dbc.Tab(label="Portfolio Insights", tab_id="portfolio-insights-tab", children=[
                    dbc.Row([dbc.Col([
                        dbc.Button("Generate Portfolio Insights", id="ml-portfolio-insights-button", color="success", className="mb-3"),
                        dbc.Spinner([html.Div(id="ml-portfolio-insights")], color="primary", type="border")
                    ], width=12)])
                ]),

                # Tab 4: Model Training (Merged Structure)
                dbc.Tab(label="Model Training", tab_id="model-training-tab", children=[
                    # --- New: Trained Model Overview ---
                    html.Div([
                        html.H5("Trained Model Overview"),
                        dbc.Button(
                            "Refresh Model List", id="refresh-trained-models",
                            color="secondary", size="sm", className="mb-3", n_clicks=0
                        ),
                        html.Div(id="trained-models-table-container", children=[
                            dbc.Spinner(html.Div(id="trained-models-table"))
                        ]),
                    ]),
                    html.Hr(), # Separator
                    # --- Original: Training Controls & Status ---
                    html.Div([
                        html.H5("Train New Model / View Status"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Select Asset to Train"),
                                dbc.InputGroup([
                                    dbc.Select(id="ml-train-asset-selector", placeholder="Select an asset for model training"),
                                    dbc.Button("Train Model", id="ml-train-button", color="warning")
                                ]),
                                html.Div(id="ml-training-status", className="mt-3"), # Immediate feedback
                                html.Hr(),
                                html.H5("Model Training Status (Live)"),
                                dbc.Spinner(html.Div(id="ml-training-status-table")) # Table for all statuses
                            ], width=12)
                        ])
                    ])
                ]),
            ]),
            dcc.Interval(
                id="ml-update-interval", # Keep original interval ID if used by original callbacks
                interval=60000,  # Update every minute (adjust as needed)
                n_intervals=0
            )
        ])
    ])

# --- MERGED CALLBACK REGISTRATION FUNCTION ---
def register_ml_prediction_callbacks(app):
    """
    Register callbacks for the merged ML prediction component.
    Includes callbacks from the original component and the new overview table.
    """
    # Initialize model integration (from original)
    model_integration = ModelIntegration()

    # --- Callbacks from Original Component ---

    # Populate asset selectors (original)
    @app.callback(
        [Output("ml-asset-selector", "options"), Output("ml-train-asset-selector", "options")],
        Input("ml-update-interval", "n_intervals") # Use the correct interval ID
    )
    def update_asset_options(n_intervals):
        tracked_assets = load_tracked_assets()
        portfolio = load_portfolio()
        portfolio_symbols = {details.get("symbol", "") for _, details in portfolio.items() if details.get("symbol")}
        all_symbols = sorted(set(tracked_assets.keys()) | portfolio_symbols)
        options = [{"label": f"{symbol} - {tracked_assets.get(symbol, {}).get('name', symbol)}", "value": symbol} for symbol in all_symbols if symbol]
        return options, options

    # Generate price prediction (original)
    @app.callback(
        [Output("ml-prediction-chart", "figure"), Output("ml-prediction-details", "children"), Output("ml-prediction-data", "data")],
        Input("ml-analyze-button", "n_clicks"),
        [State("ml-asset-selector", "value"), State("ml-horizon-selector", "value")],
        prevent_initial_call=True # Prevent initial call
    )
    def update_prediction(n_clicks, symbol, days):
        if n_clicks is None or not symbol: raise PreventUpdate
        try:
            historical_data_df = data_provider.get_historical_price(symbol, period="1y") # Use data_provider
            if historical_data_df.empty:
                fig = go.Figure().update_layout(title=f"Error: No historical data available for {symbol}", template="plotly_white")
                return fig, html.Div(f"No historical data available for {symbol}."), None

            # Use the get_price_predictions function (which now has fallbacks)
            prediction_data = get_price_predictions(symbol=symbol, days=days)

            if prediction_data:
                # Additional validation
                if (not prediction_data.get('values') or len(prediction_data['values']) == 0 or all(pd.isna(v) for v in prediction_data['values'])):
                    logger.warning(f"All prediction values for {symbol} are invalid after get_price_predictions.")
                    from modules.price_prediction import create_simple_fallback # Import fallback
                    prediction_data = create_simple_fallback(symbol, days) # Use simplest fallback

                chart = create_prediction_chart(prediction_data, historical_data_df)
                details = create_prediction_details(prediction_data, historical_data_df)
                return chart, details, prediction_data
            else:
                fig = go.Figure().update_layout(title=f"Error: Could not generate predictions for {symbol}", template="plotly_white")
                return fig, dbc.Alert(f"Could not generate predictions for {symbol}. Try training a model first.", color="warning"), None
        except Exception as e:
            logger.error(f"Error in update_prediction for {symbol}: {e}")
            traceback.print_exc()
            fig = go.Figure().update_layout(title=f"Error: {str(e)}", template="plotly_white")
            return fig, dbc.Alert(f"Error generating prediction: {str(e)}", color="danger"), None

    # Generate trend analysis (original)
    @app.callback(
        Output("trend-analysis-content", "children"),
        Input("ml-analyze-button", "n_clicks"), # Triggered when prediction is run
        State("ml-asset-selector", "value"),
        prevent_initial_call=True # Prevent initial call
    )
    def update_trend_analysis(n_clicks, symbol):
        if n_clicks is None or not symbol: raise PreventUpdate
        try:
            # Use the ModelIntegration instance initialized above
            analysis = model_integration.get_asset_analysis(symbol) # This gets all analysis types
            if analysis:
                return create_trend_analysis_display(analysis) # Pass the full analysis dict
            else:
                return dbc.Alert(f"Could not generate analysis for {symbol}.", color="warning")
        except Exception as e:
            logger.error(f"Error generating trend analysis for {symbol}: {e}")
            return dbc.Alert(f"Error generating trend analysis: {str(e)}", color="danger")

    # Generate portfolio insights (original)
    @app.callback(
        Output("ml-portfolio-insights", "children"),
        Input("ml-portfolio-insights-button", "n_clicks"),
        prevent_initial_call=True # Prevent initial call
    )
    def update_portfolio_insights(n_clicks):
        if n_clicks is None: raise PreventUpdate
        try:
            # Use the ModelIntegration instance
            recommendations = model_integration.get_portfolio_recommendations()
            return create_portfolio_insights(recommendations)
        except Exception as e:
            logger.error(f"Error generating portfolio insights: {e}")
            return dbc.Alert(f"Error generating portfolio insights: {str(e)}", color="danger")

    # Train model (original) - Outputs immediate feedback
    @app.callback(
        Output("ml-training-status", "children"),
        Input("ml-train-button", "n_clicks"),
        State("ml-train-asset-selector", "value"),
        prevent_initial_call=True # Prevent initial call
    )
    def train_model(n_clicks, symbol):
        if n_clicks is None or not symbol: raise PreventUpdate
        try:
            # Use the ModelIntegration instance
            status = model_integration.train_models_for_symbol(symbol, lookback_period="2y", async_training=True) # Run async
            if status == "pending":
                return dbc.Alert(f"Training started for {symbol}. Check status table below.", color="info")
            else: # Should not happen with async=True, but handle just in case
                return dbc.Alert(f"Training status for {symbol}: {status}", color="warning")
        except Exception as e:
            logger.error(f"Error initiating training for {symbol}: {e}")
            return dbc.Alert(f"Error starting training: {str(e)}", color="danger")

    # Update training status table (original) - Periodically updates the table
    @app.callback(
        Output("ml-training-status-table", "children"),
        [Input("ml-update-interval", "n_intervals"), # Triggered by interval
         Input("ml-training-status", "children")], # Also trigger when immediate feedback changes
        prevent_initial_call=False # Allow initial load
    )
    def update_training_status_table(n_intervals, training_status_feedback): # Renamed input arg
        try:
            # Use the ModelIntegration instance
            status_dict = model_integration.get_model_training_status()
            if not status_dict: return html.P("No model training data available.")
            return create_training_status_table(status_dict) # Use the helper
        except Exception as e:
            logger.error(f"Error retrieving training status table: {e}")
            traceback.print_exc()
            return dbc.Alert(f"Error retrieving training status: {str(e)}", color="danger")

    # Handle sell recommendation clicks (original)
    @app.callback(
        Output("ml-portfolio-insights", "children", allow_duplicate=True),
        Input({"type": "sell-rec-button", "symbol": dash.ALL}, "n_clicks"),
        State({"type": "sell-rec-button", "symbol": dash.ALL}, "id"),
        prevent_initial_call=True # Use True here
    )
    def handle_sell_recommendation(n_clicks_list, button_ids):
        if not ctx.triggered or not any(n for n in n_clicks_list if n): raise PreventUpdate
        clicked_idx = next((i for i, n in enumerate(n_clicks_list) if n), None)
        if clicked_idx is None: raise PreventUpdate
        symbol = button_ids[clicked_idx]["symbol"]
        try:
            quote = data_provider.get_current_quote(symbol) # Use data_provider
            if quote and 'price' in quote:
                current_price = quote['price']
                # Use record_transaction from portfolio_utils
                # Find shares owned for this symbol
                portfolio = load_portfolio()
                shares_owned = 0
                for inv_id, details in portfolio.items():
                    if details.get("symbol") == symbol:
                        shares_owned += float(details.get("shares", 0))

                if shares_owned <= 0:
                     return dbc.Alert(f"No shares of {symbol} owned to sell.", color="warning")

                # Default to selling all shares for simplicity in this context
                shares_to_sell = shares_owned
                success = record_transaction("sell", symbol, current_price, shares_to_sell, date=datetime.now().strftime("%Y-%m-%d"))
                if success:
                    # Refresh insights after transaction
                    recommendations = model_integration.get_portfolio_recommendations()
                    insights_content = create_portfolio_insights(recommendations)
                    return [dbc.Alert(f"Successfully recorded sell transaction for {shares_to_sell:.2f} shares of {symbol}", color="success"), insights_content]
                else:
                    return dbc.Alert(f"Failed to record sell transaction for {symbol}", color="danger")
            else:
                return dbc.Alert(f"Could not get current price for {symbol} to record sell.", color="warning")
        except Exception as e:
            logger.error(f"Error handling sell recommendation for {symbol}: {e}")
            return dbc.Alert(f"Error creating sell transaction: {str(e)}", color="danger")

    # --- Callback for Trained Models Table (New) ---
    @app.callback(
        Output("trained-models-table", "children"),
        [Input("refresh-trained-models", "n_clicks"),
         Input("ml-prediction-tabs", "active_tab")], # Trigger when tab becomes active
        prevent_initial_call=False # Allow initial load if tab is active by default
    )
    def update_trained_models_display(refresh_clicks, active_tab):
        triggered_id = ctx.triggered_id if ctx.triggered else None
        # Update only if the training tab is active OR the refresh button was the trigger
        if active_tab == 'model-training-tab' or triggered_id == 'refresh-trained-models':
            logger.info(f"Updating trained models table. Trigger: {triggered_id}, Active Tab: {active_tab}")
            try:
                models_df = get_trained_models_data() # Fetch data from db_utils
                return create_trained_models_table(models_df) # Use the helper
            except Exception as e:
                 logger.error(f"Error updating trained models display: {e}")
                 logger.error(traceback.format_exc()) # Log full traceback
                 return dbc.Alert(f"Error loading model data: {e}", color="danger")
        else:
            # Don't update if the tab isn't visible and refresh wasn't clicked
            logger.debug(f"Preventing update for trained models table. Trigger: {triggered_id}, Active Tab: {active_tab}")
            raise PreventUpdate

# --- END OF MERGED CALLBACKS ---