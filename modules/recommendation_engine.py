# modules/recommendation_engine.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras

from components.asset_tracker import load_tracked_assets

# Fix in recommendation_engine.py
class RecommendationEngine:
    """
    Generates investment recommendations based on market conditions, user profile, and AI analysis.
    """
    
    def __init__(self):
        # Define risk profiles and asset allocations
        self.risk_profiles = {
            'conservative': {
                'stocks': 0.15,
                'etfs': 0.20,
                'bonds': 0.55,
                'cash': 0.10,
                'crypto': 0.0
            },
            'moderate': {
                'stocks': 0.30,
                'etfs': 0.30,
                'bonds': 0.30,
                'cash': 0.08,
                'crypto': 0.02
            },
            'aggressive': {
                'stocks': 0.40,
                'etfs': 0.30,
                'bonds': 0.15,
                'cash': 0.05,
                'crypto': 0.10
            },
            'very_aggressive': {
                'stocks': 0.45,
                'etfs': 0.25,
                'bonds': 0.05,
                'cash': 0.05,
                'crypto': 0.20
            }
        }
        
        # Investment horizon adjustments
        self.horizon_adjustments = {
            'short': {
                'stocks': -0.1,
                'etfs': -0.05,
                'bonds': 0.1,
                'cash': 0.15,
                'crypto': -0.1
            },
            'medium': {
                'stocks': 0,
                'etfs': 0,
                'bonds': 0,
                'cash': 0,
                'crypto': 0
            },
            'long': {
                'stocks': 0.1,
                'etfs': 0.05,
                'bonds': -0.1,
                'cash': -0.1,
                'crypto': 0.05
            }
        }
        
        # Try to load pre-trained model if it exists
        try:
            self.model = keras.models.load_model('models/recommendation_model.h5')
        except:
            # We'll create a simple model in train_model method
            self.model = None
    
    def get_available_assets(self):
        """
        Get available assets from tracked assets and add other recommendations
        """
        try:
            from components.asset_tracker import load_tracked_assets
            tracked_assets = load_tracked_assets()
        except Exception as e:
            print(f"Error loading tracked assets: {e}")
            tracked_assets = {}
        
        # Organize by asset type
        assets = {
            'stocks': [],
            'etfs': [],
            'crypto': [],
            'bonds': [],
            'cash': []
        }
        
        # Add tracked assets to their respective categories
        for symbol, details in tracked_assets.items():
            asset_type = details.get('type', '')
            # Map 'etf' to 'etfs' for consistency with our categories
            if asset_type == 'etf':
                asset_type = 'etfs'
            
            if asset_type in assets:
                assets[asset_type].append({
                    'symbol': symbol,
                    'name': details.get('name', ''),
                    'type': details.get('type', '')
                })
        
        # Add some default recommendations if categories are empty
        default_assets = {
            'stocks': [
                {'symbol': 'RY.TO', 'name': 'Royal Bank of Canada', 'type': 'Financial Stock'},
                {'symbol': 'ENB.TO', 'name': 'Enbridge Inc.', 'type': 'Energy Stock'},
                {'symbol': 'SHOP.TO', 'name': 'Shopify Inc.', 'type': 'Tech Stock'}
            ],
            'etfs': [
                {'symbol': 'XIU.TO', 'name': 'iShares S&P/TSX 60 Index ETF', 'type': 'Index ETF'},
                {'symbol': 'ZCN.TO', 'name': 'BMO S&P/TSX Capped Composite Index ETF', 'type': 'Index ETF'},
                {'symbol': 'VFV.TO', 'name': 'Vanguard S&P 500 Index ETF', 'type': 'US Index ETF'}
            ],
            'crypto': [
                {'symbol': 'BTC-CAD', 'name': 'Bitcoin (CAD)', 'type': 'Cryptocurrency'},
                {'symbol': 'ETH-CAD', 'name': 'Ethereum (CAD)', 'type': 'Cryptocurrency'},
                {'symbol': 'BTCC.B.TO', 'name': 'Purpose Bitcoin ETF', 'type': 'Crypto ETF'}
            ],
            'bonds': [
                {'symbol': 'XBB.TO', 'name': 'iShares Core Canadian Bond Index ETF', 'type': 'Bond ETF'},
                {'symbol': 'ZAG.TO', 'name': 'BMO Aggregate Bond Index ETF', 'type': 'Bond ETF'},
                {'symbol': 'XSB.TO', 'name': 'iShares Core Canadian Short Term Bond Index ETF', 'type': 'Short-Term Bond ETF'}
            ],
            'cash': [
                {'symbol': 'CSAV.TO', 'name': 'CI High Interest Savings ETF', 'type': 'Cash Equivalent ETF'},
                {'symbol': 'PSA.TO', 'name': 'Purpose High Interest Savings ETF', 'type': 'Cash Equivalent ETF'}
            ]
        }
        
        # Add default assets for any empty categories
        for asset_type, default_list in default_assets.items():
            if not assets[asset_type]:
                assets[asset_type] = default_list
        
        return assets
    
    def select_assets(self, asset_class, market_data=None, news_data=None, count=3):
        """
        Select specific assets within an asset class based on market data and news.
        Now uses dynamically tracked assets.
        
        Args:
            asset_class (str): Asset class (stocks, bonds, etc.)
            market_data (dict): Market data for assets
            news_data (list): News analysis data
            count (int): Number of assets to select
            
        Returns:
            list: Selected assets
        """
        # Get available assets
        available_assets = self.get_available_assets()
        
        # If no market data or news data, select random assets
        if market_data is None or news_data is None:
            import random
            assets = available_assets.get(asset_class, [])
            if len(assets) <= count:
                return assets
            return random.sample(assets, count)
        
        # TODO: Use market data and news data to make informed selections
        # For now, we'll just return a sample of assets
        import random
        assets = available_assets.get(asset_class, [])
        if len(assets) <= count:
            return assets
        return random.sample(assets, count)
    
    def determine_risk_profile(self, risk_level):
        """
        Determine risk profile based on user's risk level (1-10).
        
        Args:
            risk_level (int): User's risk tolerance level (1-10)
            
        Returns:
            str: Risk profile
        """
        if risk_level <= 3:
            return 'conservative'
        elif risk_level <= 6:
            return 'moderate'
        elif risk_level <= 8:
            return 'aggressive'
        else:
            return 'very_aggressive'
    
    def create_allocation(self, risk_profile, investment_horizon):
        """
        Create asset allocation based on risk profile and investment horizon.
        
        Args:
            risk_profile (str): Risk profile
            investment_horizon (str): Investment horizon
            
        Returns:
            dict: Asset allocation
        """
        # Get base allocation for risk profile
        allocation = self.risk_profiles[risk_profile].copy()
        
        # Apply adjustments based on investment horizon
        for asset_class, adjustment in self.horizon_adjustments[investment_horizon].items():
            allocation[asset_class] += adjustment
            
        # Ensure no negative allocations and normalize
        for asset_class in allocation:
            allocation[asset_class] = max(0, allocation[asset_class])
            
        # Normalize to ensure sum is 1
        total = sum(allocation.values())
        for asset_class in allocation:
            allocation[asset_class] /= total
            
        return allocation
    
    def generate_recommendations(self, risk_level, investment_horizon, investment_amount, 
                                market_data=None, news_data=None, economic_data=None):
        """
        Generate investment recommendations based on user profile and market conditions.
        
        Args:
            risk_level (int): User's risk tolerance (1-10)
            investment_horizon (str): Investment horizon (short, medium, long)
            investment_amount (float): Amount to invest
            market_data (dict): Market data
            news_data (list): News analysis data
            economic_data (dict): Economic indicators
            
        Returns:
            dict: Investment recommendations
        """
        # Determine risk profile
        risk_profile = self.determine_risk_profile(risk_level)
        
        # Create base asset allocation
        allocation = self.create_allocation(risk_profile, investment_horizon)
        
        # Adjust allocation based on market conditions if data available
        if market_data and news_data and economic_data:
            # TODO: Implement sophisticated adjustment logic
            pass
        
        # Select specific assets for each asset class
        recommendations = {
            'risk_profile': risk_profile,
            'investment_horizon': investment_horizon,
            'total_amount': investment_amount,
            'allocation': allocation,
            'specific_investments': {},
            'reasoning': {}
        }
        
        for asset_class, alloc_percent in allocation.items():
            # Skip asset classes with zero allocation
            if alloc_percent <= 0:
                continue
                
            # Calculate amount for this asset class
            amount = investment_amount * alloc_percent
            
            # Select assets
            selected_assets = self.select_assets(asset_class, market_data, news_data)
            
            # Add to recommendations
            recommendations['specific_investments'][asset_class] = {
                'amount': amount,
                'assets': selected_assets
            }
            
            # Add reasoning
            if asset_class == 'stocks' and alloc_percent > 0.3:
                recommendations['reasoning'][asset_class] = "Significant allocation to stocks based on your higher risk tolerance and potential for long-term growth in the Canadian market."
            elif asset_class == 'etfs' and alloc_percent > 0.2:
                recommendations['reasoning'][asset_class] = "ETFs provide diversified exposure to market segments with lower fees than mutual funds."
            elif asset_class == 'bonds' and alloc_percent > 0.3:
                recommendations['reasoning'][asset_class] = "Bonds provide stability and regular income in your conservative portfolio."
            elif asset_class == 'cash' and alloc_percent > 0.1:
                recommendations['reasoning'][asset_class] = "Cash and cash equivalents provide liquidity and protection against market volatility."
            elif asset_class == 'crypto' and alloc_percent > 0.05:
                recommendations['reasoning'][asset_class] = "Cryptocurrency allocation provides exposure to digital assets with potential for high growth, but with higher volatility."
        
        return recommendations