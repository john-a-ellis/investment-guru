# modules/help_system.py
import os
import markdown
import re
from cachetools import cached, TTLCache
import logging

logger = logging.getLogger(__name__)

HELP_DIR = "help_content"
# Cache parsed HTML content for an hour to avoid re-parsing on every request
html_cache = TTLCache(maxsize=50, ttl=3600)

def get_help_topic_path(topic_id):
    """Constructs the path to a help topic Markdown file."""
    filename = f"{topic_id}.md"
    return os.path.join(HELP_DIR, filename)

@cached(TTLCache(maxsize=1, ttl=3600)) # Cache the list of topics
def get_available_topics():
    """
    Scans the help directory and returns a list of available topics
    (based on filenames) with formatted titles.
    """
    topics = {}
    if not os.path.exists(HELP_DIR):
        logger.error(f"Help directory not found: {HELP_DIR}")
        return topics

    for filename in os.listdir(HELP_DIR):
        if filename.endswith(".md"):
            topic_id = filename[:-3] # Remove .md extension
            # Create a more readable title from the ID
            title = topic_id.replace('_', ' ').title()
            topics[topic_id] = title
    # Sort topics alphabetically by title for consistent display
    sorted_topics = dict(sorted(topics.items(), key=lambda item: item[1]))
    return sorted_topics

@cached(html_cache)
def get_help_content_html(topic_id):
    """
    Loads a specific help topic Markdown file, parses it into HTML,
    and returns the HTML string. Returns an error message if not found.
    """
    filepath = get_help_topic_path(topic_id)
    if not os.path.exists(filepath):
        logger.warning(f"Help file not found: {filepath}")
        return f"<p>Error: Help content for '{topic_id}' not found.</p>"

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert Markdown to HTML using the 'markdown' library
        # Use extensions for better formatting (tables, fenced code, etc.)
        html_content = markdown.markdown(
            md_content,
            extensions=['fenced_code', 'tables', 'toc', 'nl2br', 'attr_list']
        )
        return html_content
    except Exception as e:
        logger.error(f"Error reading or parsing help file {filepath}: {e}")
        return f"<p>Error loading help content for '{topic_id}'.</p>"

def search_help_content(search_term):
    """
    (Optional) Basic search functionality across all help topics.
    Returns a list of topics containing the search term.
    """
    results = {}
    if not search_term or len(search_term) < 3:
        return results # Avoid searching for very short terms

    search_term_lower = search_term.lower()
    available_topics = get_available_topics()

    for topic_id, title in available_topics.items():
        filepath = get_help_topic_path(topic_id)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                if search_term_lower in content or search_term_lower in title.lower():
                    results[topic_id] = title
            except Exception as e:
                logger.error(f"Error searching file {filepath}: {e}")
    return results

