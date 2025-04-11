# components/help_display.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, ctx
from modules.help_system import get_available_topics # Import topic loader
from dash.exceptions import PreventUpdate

def create_help_display_component():
    """Creates the Offcanvas component for displaying help."""
    topics = get_available_topics()

    topic_links = [
        dbc.ListGroupItem(title, action=True, id={"type": "help-topic-link", "index": topic_id}, n_clicks=0, className="help-nav-link", active=False)
        for topic_id, title in topics.items()
    ]

    offcanvas = dbc.Offcanvas(
        id="help-offcanvas",
        title="Help & Documentation",
        is_open=False,
        placement="end",  # Show on the right side
        scrollable=True,
        close_button=True,
        autoFocus=True,
        children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Topics"),
                    dbc.ListGroup(topic_links, id="help-topic-navigation")
                ], width=4, className="help-nav-col border-end"),
                dbc.Col([
                    html.H5("Content", id="help-content-title"),
                    html.Div(id="help-content-area", className="help-content-div")
                    # Optional Search
                    # dbc.Input(id="help-search-input", placeholder="Search help...", type="search", className="mb-2"),
                    # html.Div(id="help-search-results")
                ], width=8, className="help-content-col")
            ], className="flex-grow-1")  # Make row take available height
        ],
        # Add custom CSS via className if needed for layout/scrolling
        className="help-offcanvas-container d-flex flex-column"
    )

    return offcanvas



    
