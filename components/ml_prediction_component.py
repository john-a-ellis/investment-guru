# components/ml_prediction_component.py
"""
Enhanced ML prediction component that allows users to select different model types for training.
This implementation adds a dropdown to select the model type before training.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx 
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import traceback
import json, sys, os
from dash.exceptions import PreventUpdate

# Import custom modules for ML predictions and analysis
from modules.model_integration import ModelIntegration
from modules.trend_analysis import TrendAnalyzer
from modules.price_prediction import get_price_predictions
from modules.portfolio_utils import load_tracked_assets, load_portfolio, record_transaction
from modules.data_provider import data_provider

# Import the function to get trained model data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
    if not isinstance(models_df, pd.DataFrame) or models_df.empty:
        return dbc.Alert("No trained model data found in the database.", color="info")

    metrics_to_display = ['accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'r2_score', 'rmse']

    header = [
        html.Thead(html.Tr([
            html.Th("Filename"), html.Th("Symbol"), html.Th("Type"),
            html.Th("Training Date"), html.Th("Key Metrics"),
            html.Th("Notes", style={'maxWidth': '200px'}),
            html.Th("Actions", style={'width': '80px'}) # <-- ADDED Actions column
        ]))
    ]

    rows = []
    for index, row in models_df.iterrows():
        metrics_dict = row.get('metrics', {})
        metrics_display_items = []
        # ... (metrics formatting logic remains the same) ...
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
        filename = row.get('model_filename', '') # Get filename for button ID

        # --- ADDED Delete Button ---
        delete_button = dbc.Button(
            # html.I(className="bi bi-trash-fill"), # Use Bootstrap icon
            id={"type": "delete-model-button", "filename": filename},
            color="danger",
            size="sm",
            n_clicks=0,
            disabled=not filename # Disable if filename is missing
        ) if filename else None
        # --- END Delete Button ---

        rows.append(html.Tr([
            html.Td(filename if filename else 'N/A'),
            html.Td(row.get('symbol', 'N/A')),
            html.Td(row.get('model_type', 'N/A')),
            html.Td(row.get('training_date', 'N/A')),
            metrics_cell,
            html.Td(truncated_notes, title=notes if truncated_notes != notes else ''),
            html.Td(delete_button) # <-- ADDED Button cell
        ]))
    body = [html.Tbody(rows)]
    return dbc.Table(header + body, bordered=True, striped=True, hover=True, responsive=True, size="sm")

# Register all callbacks for the ML prediction component
def register_ml_prediction_callbacks(app):
    """
    Register callbacks for the enhanced ML prediction component.
    """
    # Initialize model integration
    model_integration = ModelIntegration()

    @app.callback(
    Output("delete-model-confirm", "displayed"),
    Output("delete-model-confirm", "message"),
    Output("model-delete-feedback", "children"),
    Output("trained-models-table", "children", allow_duplicate=True),
    Input({"type": "delete-model-button", "filename": dash.ALL}, "n_clicks"),
    Input("delete-model-confirm", "submit_n_clicks"),
    State({"type": "delete-model-button", "filename": dash.ALL}, "id"),
    State("delete-model-confirm", "message"),
    prevent_initial_call=True
)
    def handle_delete_model(delete_clicks, confirm_clicks, button_ids, confirm_message):
        # Get the component that triggered the callback
        triggered_id = ctx.triggered_id
        
        # Initialize default outputs
        feedback = None
        confirm_displayed = False
        new_confirm_message = dash.no_update  # Keep message unless delete button clicked
        table_output = dash.no_update  # Don't update table unless confirmed
        
        # Skip processing if neither delete button nor confirm dialog triggered the callback
        if triggered_id is None:
            raise PreventUpdate
            
        # Check if triggered_id is a dict (pattern-matching callback)
        is_delete_button = isinstance(triggered_id, dict) and triggered_id.get("type") == "delete-model-button"
        is_confirm_dialog = triggered_id == "delete-model-confirm"
        
        # If neither a delete button nor the confirm dialog triggered this, do nothing
        if not (is_delete_button or is_confirm_dialog):
            raise PreventUpdate

        # --- Import the delete function ---
        from modules.db_utils import delete_model_metadata

        # --- Define model directory ---
        model_dir = "models"

        # Check if a delete button was clicked
        if is_delete_button:
            filename_to_delete = triggered_id.get("filename")
            if filename_to_delete:
                # Display confirmation dialog
                confirm_displayed = True
                # Store the filename in the message property (simple state management)
                new_confirm_message = f"Are you sure you want to delete model '{filename_to_delete}' and its record?"
                logger.info(f"Delete requested for {filename_to_delete}. Displaying confirmation.")

        # Check if the confirmation dialog was submitted
        elif is_confirm_dialog and confirm_clicks and confirm_clicks > 0:
            # Extract filename from the message
            # Example message: "Are you sure you want to delete model 'arima_CGL.TO.pkl' and its record?"
            try:
                filename_to_delete = confirm_message.split("'")[1]
                logger.info(f"Confirmation received to delete {filename_to_delete}")
            except (IndexError, TypeError):
                filename_to_delete = None
                feedback = dbc.Alert("Error: Could not determine which model to delete from confirmation.", color="danger")
                logger.error("Could not parse filename from confirmation message.")

            if filename_to_delete:
                db_deleted = False
                file_deleted = False
                scaler_deleted = True  # Assume true unless LSTM scaler fails

                # 1. Delete from Database
                try:
                    db_deleted = delete_model_metadata(filename_to_delete)
                    if not db_deleted:
                        logger.error(f"DB deletion failed for {filename_to_delete} (check db_utils logs).")
                except Exception as db_err:
                    logger.error(f"Exception during DB deletion for {filename_to_delete}: {db_err}")
                    feedback = dbc.Alert(f"Error deleting database record for {filename_to_delete}: {db_err}", color="danger")

                # 2. Delete Model File(s)
                if db_deleted:  # Only delete files if DB record was removed
                    model_path = os.path.join(model_dir, filename_to_delete)
                    scaler_path = None
                    if filename_to_delete.startswith("lstm_") and filename_to_delete.endswith(".h5"):
                        scaler_filename = filename_to_delete.replace(".h5", "_scaler.pkl")
                        scaler_path = os.path.join(model_dir, scaler_filename)

                    try:
                        if os.path.exists(model_path):
                            os.remove(model_path)
                            logger.info(f"Deleted model file: {model_path}")
                            file_deleted = True
                        else:
                            logger.warning(f"Model file not found, assuming already deleted: {model_path}")
                            file_deleted = True  # Consider it success if file is gone

                        if scaler_path:
                            if os.path.exists(scaler_path):
                                os.remove(scaler_path)
                                logger.info(f"Deleted scaler file: {scaler_path}")
                                scaler_deleted = True
                            else:
                                logger.warning(f"Scaler file not found, assuming already deleted: {scaler_path}")
                                scaler_deleted = True

                    except OSError as file_err:
                        logger.error(f"Error deleting file(s) for {filename_to_delete}: {file_err}")
                        feedback = dbc.Alert(f"Error deleting file(s) for {filename_to_delete}: {file_err}", color="danger")
                        file_deleted = False
                        scaler_deleted = False  # Mark as failed if error occurs

                # 3. Provide Feedback and Refresh Table
                if db_deleted and file_deleted and scaler_deleted:
                    feedback = dbc.Alert(f"Successfully deleted model {filename_to_delete}.", color="success")
                    # Refresh the table data
                    try:
                        models_df = get_trained_models_data()
                        table_output = create_trained_models_table(models_df)
                    except Exception as refresh_err:
                        logger.error(f"Error refreshing model table after deletion: {refresh_err}")
                        table_output = dbc.Alert("Error refreshing table.", color="warning")
                elif not db_deleted and feedback is None:  # If DB delete failed silently
                    feedback = dbc.Alert(f"Failed to delete database record for {filename_to_delete}. File deletion skipped.", color="danger")
                elif not file_deleted and feedback is None:  # If file delete failed
                    feedback = dbc.Alert(f"Database record deleted, but failed to delete model file for {filename_to_delete}.", color="warning")
                # Keep existing feedback if already set by specific errors

        return confirm_displayed, new_confirm_message, feedback, table_output

    # --- (Keep update_asset_options as is) ---
    @app.callback(
        [Output("ml-asset-selector", "options"), Output("ml-train-asset-selector", "options")],
        Input("ml-update-interval", "n_intervals")
    )
    def update_asset_options(n_intervals):
        # ... (implementation unchanged) ...
        tracked_assets = load_tracked_assets()
        portfolio = load_portfolio()
        portfolio_symbols = {details.get("symbol", "") for _, details in portfolio.items() if details.get("symbol")}
        all_symbols = sorted(set(tracked_assets.keys()) | portfolio_symbols)
        options = [{"label": f"{symbol} - {tracked_assets.get(symbol, {}).get('name', symbol)}", "value": symbol} for symbol in all_symbols if symbol]
        return options, options

    @app.callback(
        Output("ml-specific-model-selector", "options"),
        [Input("refresh-trained-models", "n_clicks"),
         Input("ml-update-interval", "n_intervals")], # Trigger on refresh or interval
        prevent_initial_call=False # Populate on load
    )
    def update_specific_model_options(refresh_clicks, n_intervals):
        try:
            models_df = get_trained_models_data()
            options = [{"label": "Auto-Select Model (Default)", "value": "auto"}] # Default option
            if not models_df.empty:
                # Sort by training date descending
                models_df_sorted = models_df.sort_values(by='training_date', ascending=False)
                for index, row in models_df_sorted.iterrows():
                    filename = row.get('model_filename')
                    symbol = row.get('symbol', 'N/A')
                    model_type = row.get('model_type', 'N/A')
                    date = row.get('training_date', 'N/A')
                    if filename:
                        label = f"{filename} ({symbol} - {model_type.upper()} - {date})"
                        options.append({"label": label, "value": filename})
            return options
        except Exception as e:
            logger.error(f"Error populating specific model selector: {e}")
            return [{"label": "Error loading models", "value": "auto", "disabled": True}]
        
    # Generate price prediction
    @app.callback(
        [Output("ml-prediction-chart", "figure"),
         Output("ml-prediction-details", "children"),
         Output("ml-prediction-data", "data")],
        Input("ml-analyze-button", "n_clicks"),
        [State("ml-asset-selector", "value"),
         State("ml-horizon-selector", "value"),
         State("ml-specific-model-selector", "value")], # <-- ADDED State
        prevent_initial_call=True
    )
    def update_prediction(n_clicks, symbol, days, specific_model_filename): # <-- ADDED specific_model_filename
        if n_clicks is None or not symbol: raise PreventUpdate

        try:
            # Get historical data (needed for chart context regardless of prediction method)
            historical_data_df = data_provider.get_historical_price(symbol, period="1y")
            if historical_data_df.empty:
                fig = go.Figure().update_layout(title=f"Error: No historical data available for {symbol}", template="plotly_white")
                return fig, html.Div(f"No historical data available for {symbol}."), None

            prediction_data = None
            # --- Check if a specific model was selected ---
            if specific_model_filename and specific_model_filename != "auto":
                logger.info(f"Attempting prediction for {symbol} using specific model: {specific_model_filename}")
                # --- Import the new function ---
                from modules.price_prediction import predict_with_specific_model
                prediction_data = predict_with_specific_model(
                    symbol=symbol,
                    filename=specific_model_filename,
                    days=days
                )
                if prediction_data and 'error' in prediction_data:
                     # Show error if specific model prediction failed
                     fig = go.Figure().update_layout(title=f"Error using model {specific_model_filename}", template="plotly_white")
                     return fig, dbc.Alert(f"Error using specific model: {prediction_data['error']}", color="danger"), None
            else:
                # --- Use default prediction logic ---
                logger.info(f"Attempting prediction for {symbol} using default logic (get_price_predictions)")
                prediction_data = get_price_predictions(symbol=symbol, days=days)

            # --- Process prediction results (same logic as before) ---
            if prediction_data and prediction_data.get('values'):
                chart = create_prediction_chart(prediction_data, historical_data_df)
                details = create_prediction_details(prediction_data, historical_data_df)
                return chart, details, prediction_data
            else:
                # Handle case where default prediction also failed
                error_msg = f"Could not generate predictions for {symbol}."
                if specific_model_filename and specific_model_filename != "auto":
                    error_msg += f" Specific model '{specific_model_filename}' might be incompatible or missing."
                else:
                    error_msg += " Try training a model first."
                fig = go.Figure().update_layout(title=f"Error: {error_msg}", template="plotly_white")
                return fig, dbc.Alert(error_msg, color="warning"), None

        except Exception as e:
            logger.error(f"Error in update_prediction for {symbol}: {e}")
            traceback.print_exc()
            fig = go.Figure().update_layout(title=f"Error: {str(e)}", template="plotly_white")
            return fig, dbc.Alert(f"Error generating prediction: {str(e)}", color="danger"), None



    # Generate trend analysis
    @app.callback(
        Output("trend-analysis-content", "children"),
        Input("ml-analyze-button", "n_clicks"),
        State("ml-asset-selector", "value"),
        prevent_initial_call=True
    )
    def update_trend_analysis(n_clicks, symbol):
        if n_clicks is None or not symbol: raise PreventUpdate
        try:
            analysis = model_integration.get_asset_analysis(symbol)
            if analysis:
                return create_trend_analysis_display(analysis)
            else:
                return dbc.Alert(f"Could not generate analysis for {symbol}.", color="warning")
        except Exception as e:
            logger.error(f"Error generating trend analysis for {symbol}: {e}")
            return dbc.Alert(f"Error generating trend analysis: {str(e)}", color="danger")

    # Generate portfolio insights
    @app.callback(
        Output("ml-portfolio-insights", "children"),
        Input("ml-portfolio-insights-button", "n_clicks"),
        prevent_initial_call=True
    )
    def update_portfolio_insights(n_clicks):
        if n_clicks is None: raise PreventUpdate
        try:
            recommendations = model_integration.get_portfolio_recommendations()
            return create_portfolio_insights(recommendations)
        except Exception as e:
            logger.error(f"Error generating portfolio insights: {e}")
            return dbc.Alert(f"Error generating portfolio insights: {str(e)}", color="danger")

    # Train model - Updated to use the selected model type
    @app.callback(
        Output("ml-training-status", "children"),
        Input("ml-train-button", "n_clicks"),
        [State("ml-train-asset-selector", "value"),
         State("ml-train-model-type", "value")],  # Added model type state
        prevent_initial_call=True
    )
    def train_model(n_clicks, symbol, model_type):
        """Train a model for the selected asset using the selected model type"""
        if n_clicks is None or not symbol:
            raise PreventUpdate

        try:
            # Use the selected model type (default to prophet if somehow none is selected)
            selected_model_type = model_type if model_type else "prophet"

            # Pass model_type to the integration function
            status = model_integration.train_models_for_symbol(
                symbol,
                model_type=selected_model_type,  # Use selected model type
                lookback_period="2y",
                async_training=True
            )

            if status == "pending":
                return dbc.Alert(
                    [
                        html.H5(f"Training Started"),
                        html.P(f"Training a {selected_model_type.upper()} model for {symbol}."),
                        html.P("Check the status table below for updates. Training may take several minutes."),
                        html.Div([
                            dbc.Progress(animated=True, value=100, striped=True, className="mb-2")
                        ])
                    ],
                    color="info"
                )
            else:
                return dbc.Alert(f"Training status for {symbol} ({selected_model_type.upper()}): {status}", color="warning")
        except Exception as e:
            logger.error(f"Error initiating training for {symbol} ({model_type}): {e}")
            return dbc.Alert(f"Error starting training: {str(e)}", color="danger")

    # Update training status table
    @app.callback(
        Output("ml-training-status-table", "children"),
        [Input("ml-update-interval", "n_intervals"),
         Input("ml-training-status", "children")],
        prevent_initial_call=False
    )
    def update_training_status_table(n_intervals, training_status_feedback):
        try:
            status_dict = model_integration.get_model_training_status()
            return create_training_status_table(status_dict)
        except Exception as e:
            logger.error(f"Error retrieving training status table: {e}")
            traceback.print_exc()
            return dbc.Alert(f"Error retrieving training status: {str(e)}", color="danger")

    # Handle sell recommendation clicks
    @app.callback(
        Output("ml-portfolio-insights", "children", allow_duplicate=True),
        Input({"type": "sell-rec-button", "symbol": dash.ALL}, "n_clicks"),
        State({"type": "sell-rec-button", "symbol": dash.ALL}, "id"),
        prevent_initial_call=True
    )
    def handle_sell_recommendation(n_clicks_list, button_ids):
        if not ctx.triggered or not any(n for n in n_clicks_list if n): raise PreventUpdate
        clicked_idx = next((i for i, n in enumerate(n_clicks_list) if n), None)
        if clicked_idx is None: raise PreventUpdate
        symbol = button_ids[clicked_idx]["symbol"]
        try:
            quote = data_provider.get_current_quote(symbol)
            if quote and 'price' in quote:
                current_price = quote['price']
                portfolio = load_portfolio()
                shares_owned = sum(float(details.get("shares", 0)) for inv_id, details in portfolio.items() if details.get("symbol") == symbol)

                if shares_owned <= 0: return dbc.Alert(f"No shares of {symbol} owned to sell.", color="warning")

                shares_to_sell = shares_owned # Sell all
                success = record_transaction("sell", symbol, current_price, shares_to_sell, date=datetime.now().strftime("%Y-%m-%d"))
                if success:
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

    # Callback for Trained Models Table
    @app.callback(
        Output("trained-models-table", "children"),
        [Input("refresh-trained-models", "n_clicks"),
         Input("ml-prediction-tabs", "active_tab")],
        prevent_initial_call=False
    )
    def update_trained_models_display(refresh_clicks, active_tab):
        triggered_id = ctx.triggered_id if ctx.triggered else None
        if active_tab == 'model-training-tab' or triggered_id == 'refresh-trained-models':
            logger.info(f"Updating trained models table. Trigger: {triggered_id}, Active Tab: {active_tab}")
            try:
                models_df = get_trained_models_data()  # Get the data here
                return create_trained_models_table(models_df)  # Pass it to the function
            except Exception as e:
                 logger.error(f"Error updating trained models display: {e}")
                 logger.error(traceback.print_exc())
                 return dbc.Alert(f"Error loading model data: {e}", color="danger")
        else:
            logger.debug(f"Preventing update for trained models table. Trigger: {triggered_id}, Active Tab: {active_tab}")
            raise PreventUpdate


# --- Chart and Detail Creation Functions (Keep these as they are) ---

def create_prediction_chart(prediction_data, historical_data=None):
    """
    Create a prediction chart based on ML model predictions with improved error handling.
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
    """Create trend analysis visualization for a specific asset."""
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
    """Create portfolio insights based on ML recommendations."""
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
    """Create a table showing model training status, parsing symbol and type from key."""
    if not status_dict: return html.P("No model training status available.")
    status_list = []
    for key, data in status_dict.items():
        # Try to parse symbol and type from key (e.g., "AAPL_prophet")
        parts = key.split('_')
        symbol = parts[0] if len(parts) > 1 else key # Default to key if no underscore
        model_type = parts[1] if len(parts) > 1 else 'unknown'

        if isinstance(data, dict):
            status_list.append({
                'key': key, # Keep original key if needed
                'symbol': symbol,
                'model_type': model_type,
                'status': data.get('status', 'unknown'),
                'last_updated': data.get('last_updated', 'never'),
                'metrics': data.get('metrics', {}),
                'error': data.get('error', '')
            })
        elif isinstance(data, str): # Handle simple string status
            status_list.append({
                'key': key, 'symbol': symbol, 'model_type': model_type,
                'status': data, 'last_updated': 'unknown', 'metrics': {}, 'error': ''
            })
    status_list.sort(key=lambda x: (x['symbol'], x['model_type'])) # Sort by symbol then type

    rows = []
    for status in status_list:
        status_val = status['status']
        status_color = {"completed": "success", "in_progress": "warning", "pending": "info", "failed": "danger", "unknown": "secondary", "not_started": "secondary"}.get(status_val, "secondary")
        metrics = status.get('metrics', {})
        metrics_text = ""
        if metrics and isinstance(metrics, dict) and 'error' not in metrics:
             # Display key metrics concisely
             key_metrics = {k: v for k, v in metrics.items() if k in ['mae', 'rmse', 'mape']}
             metrics_text = ", ".join([f"{k.upper()}: {v:.2f}" for k,v in key_metrics.items()])
        elif metrics and isinstance(metrics, dict) and 'error' in metrics:
             metrics_text = f"Error: {metrics['error'][:50]}..." # Show part of error

        error_message = status.get('error', '')
        rows.append(html.Tr([
            html.Td(status['symbol']),
            html.Td(status['model_type'].upper()), # Show model type
            html.Td(html.Span(status_val.replace('_', ' ').title(), className=f"text-{status_color}")),
            html.Td(status['last_updated']),
            html.Td(metrics_text),
            html.Td(error_message, className="text-danger small" if error_message else "") # Smaller error text
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Symbol"), html.Th("Model"), html.Th("Status"), # Added Model column
            html.Th("Last Updated"), html.Th("Metrics"), html.Th("Error (if any)")
        ])),
        html.Tbody(rows)
    ], bordered=True, striped=True, hover=True, size="sm") # Use small size


def create_ml_prediction_component():
    """
    Creates the enhanced ML Prediction component with model type selection.
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H4("AI Investment Analysis & Model Management", className="card-title"),
            html.P("Machine learning predictions, technical analysis, and model overview", className="card-subtitle")
        ]),
        dbc.CardBody([
            # --- ADDED Confirmation Dialog and Feedback Div ---
            dcc.ConfirmDialog(
                id='delete-model-confirm',
                message='Are you sure you want to delete this model file and its record?',
            ),
            html.Div(id="model-delete-feedback"),
            # --- END Additions ---
            dbc.Tabs(id="ml-prediction-tabs", active_tab="price-prediction-tab", children=[
                # ... (Tabs 1, 2, 3 remain the same) ...
                # Tab 1: Price Prediction
                dbc.Tab(label="Price Prediction", tab_id="price-prediction-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Asset"),
                            dbc.Select(id="ml-asset-selector", placeholder="Select an asset to analyze"),
                        ], width=4), # Adjusted width
                        dbc.Col([
                            dbc.Label("Use Specific Model (Optional)"),
                            dbc.Select(
                                id="ml-specific-model-selector",
                                placeholder="Auto-Select Model (Default)",
                                value="auto" # Default value
                            ),
                        ], width=3), # Adjusted width
                        dbc.Col([
                            dbc.Label("Prediction Horizon"),
                            dbc.RadioItems(
                                id="ml-horizon-selector",
                                options=[{"label": "30 Days", "value": 30}, {"label": "60 Days", "value": 60}, {"label": "90 Days", "value": 90}],
                                value=30, inline=True
                            )
                        ], width=3), # Adjusted width
                        dbc.Col([
                             dbc.Button("Analyze", id="ml-analyze-button", color="primary", className="w-100", style={"marginTop": "32px"}) # Align button
                        ], width=2) # Adjusted width
                    ], className="mb-3 align-items-end"), # Align items vertically
                    dbc.Spinner([dcc.Graph(id="ml-prediction-chart")], color="primary", type="border", fullscreen=False),
                    html.Div(id="ml-prediction-details", className="mt-3"),
                    dcc.Store(id="ml-prediction-data")
                ]),
                # Tab 2: Technical Analysis
                dbc.Tab(label="Technical Analysis", tab_id="technical-analysis-tab", children=[
                    dbc.Row([dbc.Col([dbc.Spinner([html.Div(id="trend-analysis-content")], color="primary", type="border")], width=12)])
                ]),
                # Tab 3: Portfolio Insights
                dbc.Tab(label="Portfolio Insights", tab_id="portfolio-insights-tab", children=[
                    dbc.Row([dbc.Col([
                        dbc.Button("Generate Portfolio Insights", id="ml-portfolio-insights-button", color="success", className="mb-3"),
                        dbc.Spinner([html.Div(id="ml-portfolio-insights")], color="primary", type="border")
                    ], width=12)])
                ]),
                # Tab 4: Model Training
                dbc.Tab(label="Model Training", tab_id="model-training-tab", children=[
                    # Trained Model Overview
                    html.Div([
                        html.H5("Trained Model Overview"),
                        dbc.Button(
                            "Refresh Model List", id="refresh-trained-models",
                            color="secondary", size="sm", className="mb-3", n_clicks=0
                        ),
                        html.Div(id="trained-models-table-container", children=[
                            dbc.Spinner(html.Div(id="trained-models-table")) # Table is rendered here
                        ]),
                    ]),
                    html.Hr(),
                    # Training Controls & Status
                    html.Div([
                        # ... (Training controls remain the same) ...
                        html.H5("Train New Model / View Status"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Select Asset to Train"),
                                dbc.Select(id="ml-train-asset-selector", placeholder="Select an asset for model training"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Select Model Type"),
                                dbc.Select(
                                    id="ml-train-model-type",
                                    options=[
                                        {"label": "Prophet (Best for Most Cases)", "value": "prophet"},
                                        {"label": "ARIMA (Statistical Model)", "value": "arima"},
                                        {"label": "LSTM (Deep Learning)", "value": "lstm"}
                                    ],
                                    value="prophet",
                                    placeholder="Select model type"
                                ),
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Train Model", id="ml-train-button", color="warning", className="w-100")
                            ], width=12)
                        ]),
                        html.Div([
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader("Model Type Information"),
                                    dbc.PopoverBody([
                                        html.P("Prophet: Facebook's forecasting tool. Fast, handles seasonality well, good for most assets."),
                                        html.P("ARIMA: Statistical time series model. Good for data with clear trends but limited seasonality."),
                                        html.P("LSTM: Deep learning neural network. Can capture complex patterns but requires more data and training time."),
                                    ]),
                                ],
                                target="ml-train-model-type",
                                trigger="hover",
                                placement="bottom"
                            ),
                        ]),
                        html.Div(id="ml-training-status", className="mt-3"),
                        html.Hr(),
                        html.H5("Model Training Status (Live)"),
                        dbc.Spinner(html.Div(id="ml-training-status-table"))
                    ])
                ]),
            ]),
            dcc.Interval(
                id="ml-update-interval",
                interval=60000,  # Update every minute
                n_intervals=0
            )
        ])
    ])