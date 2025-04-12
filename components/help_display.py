# components/help_display.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from modules.help_system import get_available_topics

def create_help_display_component():
    """Creates the Offcanvas component for displaying help."""
    topics = get_available_topics()

    # Create a list group item for each available help topic
    topic_links = [
        dbc.ListGroupItem(
            title, 
            action=True, 
            id={"type": "help-topic-link", "index": topic_id}, 
            n_clicks=0, 
            className="help-nav-link", 
            active=(topic_id == "introduction")  # Set introduction as active by default
        )
        for topic_id, title in topics.items()
    ]

    # Create the offcanvas component
    offcanvas = dbc.Offcanvas(
        id="help-offcanvas",
        title="Help & Documentation",
        is_open=False,
        placement="end",  # Show on the right side
        scrollable=True,
        close_button=True,
        backdrop=True,  # Add backdrop for better UX
        children=[
            dbc.Row([
                # Topic navigation sidebar
                dbc.Col([
                    html.H5("Topics"),
                    dbc.ListGroup(topic_links, id="help-topic-navigation")
                ], width=4, className="help-nav-col border-end"),
                
                # Content area
                dbc.Col([
                    html.H5("Content", id="help-content-title"),
                    html.Div(id="help-content-area", className="help-content-div")
                ], width=8, className="help-content-col")
            ], className="g-0 flex-grow-1")  # Remove gutters and make row take available height
        ],
        # Add custom CSS for better styling
        className="help-offcanvas-container d-flex flex-column",
        style={"width": "80%", "max-width": "1000px"}  # Make the offcanvas wider
    )

    return offcanvas


    
