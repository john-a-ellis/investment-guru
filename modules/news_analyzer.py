# modules/news_analyzer.py
import re
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer # Keep commented
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging # Optional: for logging model load
import threading # Use threading lock for thread safety in Dash

logger = logging.getLogger(__name__) # Optional

class NewsAnalyzer:
    """
    Analyzes financial news and social media sentiment to identify market-moving events.
    Uses lazy loading for the FinBERT model.
    """

    def __init__(self):
        # Download required NLTK resources (keep this part)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        # Initialize FinBERT model and tokenizer to None initially
        self.finbert_tokenizer = None
        self.finbert_model = None
        # Use a lock to prevent race conditions if multiple requests trigger loading simultaneously
        self._model_loading_lock = threading.Lock()

        # Initialize other components (keep this part)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.company_keywords = {
            "AAPL": ["apple", "iphone", "ipad", "macbook", "tim cook"],
            # ... other keywords ...
        }
        self.sector_keywords = {
            "Technology": ["tech", "software", "hardware", "semiconductor", "cloud"],
            # ... other keywords ...
        }

    def _ensure_model_loaded(self):
        """Loads the FinBERT model and tokenizer if they haven't been loaded yet."""
        # Check without lock first for performance
        if self.finbert_model is not None and self.finbert_tokenizer is not None:
            return True

        # Acquire lock to ensure only one thread loads the model
        with self._model_loading_lock:
            # Double-check inside the lock to prevent redundant loading
            if self.finbert_model is None or self.finbert_tokenizer is None:
                print("--- Loading FinBERT model and tokenizer (lazy load) ---") # Add print statement
                logger.info("Loading FinBERT model and tokenizer...") # Optional logging
                try:
                    self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                    print("--- FinBERT model and tokenizer loaded successfully ---") # Add print statement
                    logger.info("FinBERT model and tokenizer loaded successfully.") # Optional logging
                    return True
                except Exception as e:
                    print(f"--- ERROR: Failed to load FinBERT model: {e} ---") # Add print statement
                    logger.error(f"Failed to load FinBERT model: {e}", exc_info=True) # Optional logging
                    # Keep model/tokenizer as None to indicate failure
                    self.finbert_tokenizer = None
                    self.finbert_model = None
                    return False # Indicate failure
        return True # Model was already loaded by another thread or previous check

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using FinBERT. Loads model on first call.
        """
        # Ensure the model is loaded before proceeding
        if not self._ensure_model_loaded():
             print("ERROR: FinBERT model not available for sentiment analysis.")
             logger.error("FinBERT model not available for sentiment analysis.")
             # Return a default neutral sentiment or raise an error
             return {'category': 'neutral', 'score': 0.0, 'raw_scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}}

        # --- Rest of the analyze_sentiment method remains the same ---
        if not text or not isinstance(text, str):
            return {'category': 'neutral', 'score': 0.0, 'raw_scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}}

        # Tokenize text for FinBERT
        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get model predictions
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities (softmax)
        probabilities = torch.softmax(logits, dim=-1).squeeze()

        # Assuming the model labels are [positive, negative, neutral] for ProsusAI/finbert
        score_positive = probabilities[0].item()
        score_negative = probabilities[1].item()
        score_neutral = probabilities[2].item()

        # Determine category based on highest probability
        max_prob = max(score_positive, score_negative, score_neutral)
        if max_prob == score_positive:
            category = 'positive'
            final_score = score_positive
        elif max_prob == score_negative:
            category = 'negative'
            final_score = -score_negative
        else:
            category = 'neutral'
            # Let's make neutral score 0.0 for consistency with VADER's compound range idea
            final_score = 0.0

        return {
            'category': category,
            'score': final_score,
            'raw_scores': {
                'positive': score_positive,
                'negative': score_negative,
                'neutral': score_neutral
            }
        }

    # --- Other methods (preprocess_text, analyze_news_article, etc.) remain the same ---
    # --- Make sure the standalone analyze_news_sentiment function is DELETED ---

    def preprocess_text(self, text):
        # ... (keep as is, but note it's not used by FinBERT analyze_sentiment) ...
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return tokens

    # This method still exists but needs review if identify_entities is used
    def identify_entities(self, text):
        """
        Identify companies and sectors mentioned in text.
        Args:
            text (str): Text to analyze
        Returns:
            dict: Identified entities
        """
        text_lower = text.lower() # Use lowercased text for keyword matching
        entities = {
            'companies': [],
            'sectors': []
        }
        # Identify companies
        for company, keywords in self.company_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if company not in entities['companies']:
                        entities['companies'].append(company)
                    break
        # Identify sectors
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if sector not in entities['sectors']:
                        entities['sectors'].append(sector)
                    break
        return entities

    def analyze_news_article(self, article):
        """
        Analyze a single news article.
        """
        if 'title' not in article or 'description' not in article:
            return None

        # Combine title and description for analysis - use original case for FinBERT
        text_for_sentiment = article['title'] + ". " + (article['description'] or "")
        # Use lowercased text if needed for entity identification
        text_for_entities = text_for_sentiment.lower()

        # Analyze sentiment using FinBERT (will trigger lazy load on first call)
        sentiment = self.analyze_sentiment(text_for_sentiment)

        # Identify entities (using lowercased text)
        entities = self.identify_entities(text_for_entities) # Pass lowercased text

        # Determine market impact
        impact = 'neutral'
        if sentiment['category'] == 'positive' and (entities['companies'] or entities['sectors']):
            impact = 'positive'
        elif sentiment['category'] == 'negative' and (entities['companies'] or entities['sectors']):
            impact = 'negative'

        # Calculate relevance score (simple example)
        relevance = 0.5
        if entities['companies'] and entities['sectors']:
            relevance = 0.8
        elif entities['companies'] or entities['sectors']:
            relevance = 0.6

        analysis = {
            'title': article['title'],
            'description': article['description'],
            'date': article.get('publishedAt', ''),
            'source': article.get('source', {}).get('name', ''),
            'url': article.get('url', ''),
            'sentiment': sentiment, # Contains category, score, raw_scores
            'entities': entities,
            'impact': impact,
            'relevance': relevance
        }
        return analysis

    # analyze_recent_news and identify_market_events should work with the new analysis structure
    def analyze_recent_news(self, news_articles=None):
        # ... (keep as is) ...
        from modules.data_collector import DataCollector # Consider moving import to top
        if news_articles is None:
            data_collector = DataCollector()
            news_articles = data_collector.get_latest_news()
        analyzed_news = []
        for article in news_articles:
            analysis = self.analyze_news_article(article)
            if analysis:
                analyzed_news.append(analysis)
        analyzed_news.sort(key=lambda x: x['relevance'], reverse=True)
        return analyzed_news

    def identify_market_events(self, analyzed_news):
        # ... (keep as is, it uses sentiment['category'] which is still present) ...
        events = []
        company_news = {}
        sector_news = {}
        # ... (rest of the logic) ...
        return events