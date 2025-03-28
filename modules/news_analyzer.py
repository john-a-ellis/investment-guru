# modules/news_analyzer.py
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NewsAnalyzer:
    """
    Analyzes financial news and social media sentiment to identify market-moving events.
    """
    
    def __init__(self):
        # Download required NLTK resources
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
            
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize company and sector keywords
        self.company_keywords = {
            "AAPL": ["apple", "iphone", "ipad", "macbook", "tim cook"],
            "MSFT": ["microsoft", "azure", "windows", "office", "satya nadella"],
            "AMZN": ["amazon", "aws", "bezos", "prime", "jassy"],
            "GOOG": ["google", "alphabet", "pichai", "android", "youtube"],
            "META": ["facebook", "meta", "instagram", "zuckerberg", "whatsapp"]
        }
        
        self.sector_keywords = {
            "Technology": ["tech", "software", "hardware", "semiconductor", "cloud"],
            "Healthcare": ["health", "pharma", "biotech", "medical", "drug"],
            "Finance": ["bank", "finance", "investment", "loan", "mortgage", "interest rate"],
            "Energy": ["oil", "gas", "energy", "renewable", "solar", "wind"],
            "Consumer": ["retail", "consumer", "e-commerce", "shopping", "goods"]
        }
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of processed tokens
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        sentiment = self.sia.polarity_scores(text)
        
        # Categorize sentiment
        if sentiment['compound'] >= 0.05:
            sentiment['category'] = 'positive'
        elif sentiment['compound'] <= -0.05:
            sentiment['category'] = 'negative'
        else:
            sentiment['category'] = 'neutral'
            
        return sentiment
    
    def identify_entities(self, text):
        """
        Identify companies and sectors mentioned in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Identified entities
        """
        text_lower = text.lower()
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
        
        Args:
            article (dict): News article data
            
        Returns:
            dict: Analysis results
        """
        if 'title' not in article or 'description' not in article:
            return None
            
        # Combine title and description for analysis
        text = article['title'] + ". " + (article['description'] or "")
        
        # Preprocess text
        processed_tokens = self.preprocess_text(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Identify entities
        entities = self.identify_entities(text)
        
        # Determine market impact
        impact = 'neutral'
        if sentiment['category'] == 'positive' and entities['companies'] or entities['sectors']:
            impact = 'positive'
        elif sentiment['category'] == 'negative' and entities['companies'] or entities['sectors']:
            impact = 'negative'
            
        # Calculate relevance score
        relevance = 0.5  # Default
        if entities['companies'] and entities['sectors']:
            relevance = 0.8
        elif entities['companies'] or entities['sectors']:
            relevance = 0.6

            # Add analysis results
        analysis = {
            'title': article['title'],
            'description': article['description'],
            'date': article.get('publishedAt', ''),
            'source': article.get('source', {}).get('name', ''),
            'url': article.get('url', ''),
            'sentiment': sentiment,
            'entities': entities,
            'impact': impact,
            'relevance': relevance
        }
        
        return analysis
    
    def analyze_recent_news(self, news_articles=None):
        """
        Analyze recent financial news articles.
        
        Args:
            news_articles (list): List of news articles to analyze
            
        Returns:
            list: Analyzed news with impact assessment
        """
        from modules.data_collector import DataCollector
        
        if news_articles is None:
            # Get recent news using the DataCollector
            data_collector = DataCollector()
            news_articles = data_collector.get_latest_news()
        
        analyzed_news = []
        for article in news_articles:
            analysis = self.analyze_news_article(article)
            if analysis:
                analyzed_news.append(analysis)
        
        # Sort by relevance
        analyzed_news.sort(key=lambda x: x['relevance'], reverse=True)
        
        return analyzed_news
    
    def identify_market_events(self, analyzed_news):
        """
        Identify significant market events from analyzed news.
        
        Args:
            analyzed_news (list): List of analyzed news articles
            
        Returns:
            list: Significant market events
        """
        events = []
        
        # Group news by entities
        company_news = {}
        sector_news = {}
        
        for news in analyzed_news:
            # Group by companies
            for company in news['entities']['companies']:
                if company not in company_news:
                    company_news[company] = []
                company_news[company].append(news)
            
            # Group by sectors
            for sector in news['entities']['sectors']:
                if sector not in sector_news:
                    sector_news[sector] = []
                sector_news[sector].append(news)
        
        # Identify company-specific events
        for company, news_list in company_news.items():
            # Check if there are multiple high-relevance news items
            if len([n for n in news_list if n['relevance'] > 0.7]) >= 2:
                # Check sentiment consistency
                sentiments = [n['sentiment']['category'] for n in news_list]
                if sentiments.count('positive') > len(sentiments) * 0.7:
                    events.append({
                        'type': 'company',
                        'entity': company,
                        'event': 'positive_trend',
                        'confidence': 0.8,
                        'news': news_list
                    })
                elif sentiments.count('negative') > len(sentiments) * 0.7:
                    events.append({
                        'type': 'company',
                        'entity': company,
                        'event': 'negative_trend',
                        'confidence': 0.8,
                        'news': news_list
                    })
        
        # Identify sector-specific events
        for sector, news_list in sector_news.items():
            # Check if there are multiple high-relevance news items
            if len([n for n in news_list if n['relevance'] > 0.7]) >= 3:
                # Check sentiment consistency
                sentiments = [n['sentiment']['category'] for n in news_list]
                if sentiments.count('positive') > len(sentiments) * 0.6:
                    events.append({
                        'type': 'sector',
                        'entity': sector,
                        'event': 'positive_trend',
                        'confidence': 0.7,
                        'news': news_list
                    })
                elif sentiments.count('negative') > len(sentiments) * 0.6:
                    events.append({
                        'type': 'sector',
                        'entity': sector,
                        'event': 'negative_trend',
                        'confidence': 0.7,
                        'news': news_list
                    })
        
        return events