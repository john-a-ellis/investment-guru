# components/rebalancing_component.py
"""
Dashboard component for visualizing portfolio rebalancing recommendations.
"""
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_rebalancing_component():
    """
    Creates a component for visualizing portfolio rebalancing recommendations
    
    Returns:
        dbc.Card: Rebalancing visualization component
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Rebalancing"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab([
                    # Current vs Target Allocation
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Current vs Target Allocation", className="text-center my-3")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="current-vs-target-chart")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="allocation-drift-table")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="rebalance-summary", className="my-3")
                            ], width=12)
                        ])
                    ])
                ], label="Allocation Analysis", tab_id="allocation-analysis-tab"),
                
                dbc.Tab([
                    # Rebalancing Recommendations
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Rebalancing Recommendations", className="text-center my-3")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="rebalance-recommendations")
                            ], width=12)
                        ])
                    ])
                ], label="Rebalancing Plan", tab_id="rebalancing-plan-tab"),
                
                dbc.Tab([
                    # Target Allocation Settings
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Set Target Allocation", className="text-center my-3")
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.P("Adjust the sliders below to set your target asset allocation:"),
                                    html.Div(id="target-allocation-sliders"),
                                    html.Div(id="slider-total-warning", className="my-2"),
                                    dbc.Button("Save Target Allocation", id="save-target-allocation", color="primary", className="mt-3"),
                                    html.Div(id="target-allocation-feedback", className="mt-2")
                                ])
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="target-allocation-chart")
                            ], width=6)
                        ])
                    ])
                ], label="Target Settings", tab_id="target-settings-tab")
            ], id="rebalancing-tabs"),
            
            dcc.Interval(
                id="rebalance-update-interval",
                interval=3600000,  # 1 hour in milliseconds
                n_intervals=0
            )
        ])
    ])

def create_allocation_sliders(current_targets):
    """
    Create sliders for target allocation adjustment.
    
    Args:
        current_targets (dict): Current target allocations
        
    Returns:
        list: List of slider components
    """
    slider_components = []
    
    # Add a slider for each asset type, sorted alphabetically for consistency
    for asset_type, percentage in sorted(current_targets.items()):
        # Format the label with proper capitalization
        label = asset_type.replace('_', ' ').title()
        
        slider_components.append(
            dbc.Row([
                dbc.Col([
                    dbc.Label(f"{label}: {percentage:.1f}%")
                ], width=3),
                dbc.Col([
                    dcc.Slider(
                        id={"type": "target-slider", "asset_type": asset_type},
                        min=0,
                        max=100,
                        step=1,
                        value=percentage,
                        marks={i: str(i) for i in range(0, 101, 20)},
                        className="my-1"
                    )
                ], width=8),
                dbc.Col([
                    html.Div(id={"type": "slider-output", "asset_type": asset_type}, 
                            children=f"{percentage:.1f}%")  # Initialize with current value
                ], width=1)
            ], className="mb-2")
        )
    
    return slider_components

def create_current_vs_target_chart(analysis):
    """
    Create a chart comparing current vs target allocation.
    
    Args:
        analysis (dict): Portfolio allocation analysis
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Check if analysis data is valid
    if not analysis or "current_allocation" not in analysis or "target_allocation" not in analysis:
        fig = go.Figure()
        fig.update_layout(
            title="Current vs Target Allocation (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Prepare data for chart
    asset_types = list(set(list(analysis["current_allocation"].keys()) + list(analysis["target_allocation"].keys())))
    asset_types.sort()
    
    current_values = [analysis["current_allocation"].get(asset_type, {}).get("percentage", 0) for asset_type in asset_types]
    target_values = [analysis["target_allocation"].get(asset_type, 0) for asset_type in asset_types]
    
    # Format asset type labels for display
    formatted_labels = [asset_type.replace('_', ' ').title() for asset_type in asset_types]
    
    # Create figure with two bar series
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=formatted_labels,
        x=current_values,
        name='Current Allocation',
        orientation='h',
        marker=dict(color='rgba(58, 71, 80, 0.8)')
    ))
    
    fig.add_trace(go.Bar(
        y=formatted_labels,
        x=target_values,
        name='Target Allocation',
        orientation='h',
        marker=dict(color='rgba(25, 145, 206, 0.8)')
    ))
    
    # Update layout
    fig.update_layout(
        barmode='group',
        title="Current vs Target Allocation",
        xaxis_title="Percentage (%)",
        yaxis_title="Asset Type",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_target_allocation_chart(target_allocation):
    """
    Create a pie chart visualizing the target allocation.
    
    Args:
        target_allocation (dict): Target allocation percentages
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Check if target_allocation is valid
    if not target_allocation:
        fig = go.Figure()
        fig.update_layout(
            title="Target Allocation (No Data Available)",
            template="plotly_white"
        )
        return fig
    
    # Prepare data for pie chart
    labels = []
    values = []
    
    # Process each asset type and its value
    for asset_type, percentage in target_allocation.items():
        # Format asset type label for display
        display_label = asset_type.replace('_', ' ').title()
        
        # Add to chart data
        labels.append(display_label)
        values.append(percentage)
    
    # Create figure
    fig = px.pie(
        names=labels,
        values=values,
        title="Target Allocation",
        hole=0.4,
        template="plotly_white"
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update hover template to show percentages
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Target: %{value:.1f}%<extra></extra>"
    )
    
    return fig
    
    # Prepare data for pie chart
    labels = [asset_type.replace('_', ' ').title() for asset_type in target_allocation.keys()]
    values = list(target_allocation.values())
    
    # Create figure
    fig = px.pie(
        names=labels,
        values=values,
        title="Target Allocation",
        hole=0.4,
        template="plotly_white"
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_allocation_drift_table(analysis):
    """
    Create a table showing the drift between current and target allocation.
    
    Args:
        analysis (dict): Portfolio allocation analysis
        
    Returns:
        dash_table.DataTable or html.Div: Table component
    """
    # Check if analysis data is valid
    if not analysis or "drift_analysis" not in analysis:
        return html.Div("No allocation data available.")
    
    # Prepare data for table
    table_data = []
    for asset_type, data in analysis["drift_analysis"].items():
        # Format values for display
        label = asset_type.replace('_', ' ').title()
        current_pct = f"{data['current_percentage']:.1f}%"
        target_pct = f"{data['target_percentage']:.1f}%"
        drift_pct = f"{data['drift_percentage']:.1f}%"
        
        # Create a colored cell for drift
        if data["is_actionable"]:
            drift_style = {"color": "red" if data["action"] == "sell" else "green", "font-weight": "bold"}
        else:
            drift_style = {}
        
        # Add to table data
        table_data.append({
            "asset_type": label,
            "current": current_pct,
            "target": target_pct,
            "drift": drift_pct,
            "current_value": f"${data['current_value']:.2f}",
            "target_value": f"${data['target_value']:.2f}",
            "action": data["action"].capitalize() if data["is_actionable"] else "Hold",
            "drift_style": drift_style
        })
    
    # Sort table by absolute drift (highest first)
    table_data.sort(key=lambda x: abs(float(x["drift"].strip("%"))), reverse=True)
    
    # Create the table
    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("Asset Type"),
                html.Th("Current"),
                html.Th("Target"),
                html.Th("Drift"),
                html.Th("Current Value"),
                html.Th("Target Value"),
                html.Th("Action")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row["asset_type"]),
                html.Td(row["current"]),
                html.Td(row["target"]),
                html.Td(row["drift"], style=row["drift_style"]),
                html.Td(row["current_value"]),
                html.Td(row["target_value"]),
                html.Td(
                    row["action"], 
                    style={"color": "red" if row["action"] == "Sell" else "green" if row["action"] == "Buy" else ""}
                )
            ]) for row in table_data
        ])
    ], className="table table-striped table-hover")
    
    return table

def create_rebalance_recommendations(rebalance_plan):
    """
    Create a component displaying rebalance recommendations.
    
    Args:
        rebalance_plan (dict): Portfolio rebalancing plan
        
    Returns:
        html.Div: Component with recommendations
    """
    # Check if rebalance_plan is valid
    if not rebalance_plan or "recommendations" not in rebalance_plan:
        return html.Div("No rebalancing recommendations available.")
    
    recommendations = rebalance_plan.get("recommendations", [])
    if not recommendations:
        return html.Div(
            dbc.Alert("Your portfolio is well-balanced. No rebalancing actions needed at this time.", color="success"),
            className="mt-3"
        )
    
    # Create cards for each recommendation
    recommendation_cards = []
    for rec in recommendations:
        # Determine card color based on action
        card_color = "danger" if rec["action"] == "Sell" else "success"
        
        # Create card with recommendation details
        card = dbc.Card([
            dbc.CardHeader(
                html.H5(f"{rec['action']} {rec['symbol']}", className=f"text-{card_color}")
            ),
            dbc.CardBody([
                html.H5(f"${rec['amount']:.2f}", className="card-title"),
                html.P(rec["description"]),
                html.Small(f"Asset Type: {rec['asset_type'].replace('_', ' ').title()}", className="text-muted")
            ])
        ], className="mb-3", outline=True, color=card_color)
        
        recommendation_cards.append(card)
    
    # Group recommendations into rows with 3 cards per row
    rows = []
    for i in range(0, len(recommendation_cards), 3):
        row_cards = recommendation_cards[i:i+3]
        row = dbc.Row([
            dbc.Col(card, width=4) for card in row_cards
        ], className="mb-2")
        rows.append(row)
    
    # Add summary information
    summary = rebalance_plan.get("summary", {})
    summary_component = html.Div([
        html.H5("Rebalancing Summary", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${summary.get('total_sells', 0):.2f}", className="text-danger"),
                        html.P("Total to Sell")
                    ], className="text-center")
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${summary.get('total_buys', 0):.2f}", className="text-success"),
                        html.P("Total to Buy")
                    ], className="text-center")
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(
                            f"${summary.get('net_cash_impact', 0):.2f}", 
                            className=f"{'text-success' if summary.get('net_cash_impact', 0) >= 0 else 'text-danger'}"
                        ),
                        html.P("Net Cash Impact")
                    ], className="text-center")
                ])
            ], width=4)
        ], className="mb-4")
    ])
    
    return html.Div([
        summary_component,
        *rows,
        dbc.Alert(
            "Note: These recommendations aim to bring your portfolio closer to your target allocation. Consider transaction costs and tax implications before executing these trades.",
            color="info",
            className="mt-3"
        )
    ])

def create_rebalance_summary(analysis):
    """
    Create a summary of the portfolio's current allocation status.
    
    Args:
        analysis (dict): Portfolio allocation analysis
        
    Returns:
        html.Div: Summary component
    """
    # Check if analysis is valid
    if not analysis:
        return html.Div("No allocation data available.")
    
    # Count actionable items
    actionable_items = analysis.get("actionable_items", {})
    action_count = len(actionable_items)
    
    if action_count == 0:
        return dbc.Alert(
            html.Div([
                html.H4("Portfolio Well-Balanced", className="alert-heading"),
                html.P("Your portfolio is currently well-aligned with your target allocation. No rebalancing needed at this time.")
            ]),
            color="success"
        )
    else:
        # Get total drift
        total_drift = sum(abs(data["drift_percentage"]) for data in analysis.get("drift_analysis", {}).values())
        
        # Calculate rebalance urgency based on drift
        if total_drift > 30:
            urgency = "High"
            urgency_color = "danger"
            urgency_text = "Your portfolio has significant deviation from your target allocation. Rebalancing is strongly recommended."
        elif total_drift > 15:
            urgency = "Medium"
            urgency_color = "warning"
            urgency_text = "Your portfolio has moderate deviation from your target allocation. Consider rebalancing soon."
        else:
            urgency = "Low"
            urgency_color = "info"
            urgency_text = "Your portfolio has slight deviation from your target allocation. Minor rebalancing may be beneficial."
        
        return dbc.Alert(
            html.Div([
                html.H4(f"Rebalancing Recommended ({urgency} Urgency)", className="alert-heading"),
                html.P(f"Found {action_count} asset classes that need adjustment. Total allocation drift: {total_drift:.1f}%"),
                html.Hr(),
                html.P(urgency_text, className="mb-0")
            ]),
            color=urgency_color
        )