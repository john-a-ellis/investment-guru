# modules/recommendation_engine.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras

class RecommendationEngine:
    """
    Generates investment recommendations based on market conditions, user profile, and AI analysis.
    """
    
    def __init__(self):
        # Define risk profiles and asset allocations
        self.risk_profiles = {
            'conservative': {
                'stocks': 0.2,
                'bonds': 0.6,
                'cash': 0.15,
                'alternatives': 0.05
            },
            'moderate': {
                'stocks': 0.5,
                'bonds': 0.3,
                'cash': 0.1,
                'alternatives': 0.1
            },
            'aggressive': {
                'stocks': 0.7,
                'bonds': 0.15,
                'cash': 0.05,
                'alternatives': 0.1
            },
            'very_aggressive': {
                'stocks': 0.85,
                'bonds': 0.05,
                'cash': 0.02,
                'alternatives': 0.08
            }
        }
        
        # Investment horizon adjustments
        self.horizon_adjustments = {
            'short': {
                'stocks': -0.2,
                'bonds': 0.1,
                'cash': 0.15,
                'alternatives': -0.05
            },
            'medium': {
                'stocks': 0,
                'bonds': 0,
                'cash': 0,
                'alternatives': 0
            },
            'long': {
                'stocks': 0.15,
                'bonds': -0.1,
                'cash': -0.1,
                'alternatives': 0.05
            }
        }
        
        # Sample assets for each class
        self.assets = {
            'stocks': [
                {'symbol': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'type': 'ETF'},
                {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'Tech Stock'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'type': 'Tech Stock'},
                {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.', 'type': 'Tech Stock'},
                {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'type': 'Healthcare Stock'},
                {'symbol': 'PG', 'name': 'Procter & Gamble Co.', 'type': 'Consumer Stock'}
            ],
            'bonds': [
                {'symbol': 'BND', 'name': 'Vanguard Total Bond Market ETF', 'type': 'ETF'},
                {'symbol': 'AGG', 'name': 'iShares Core U.S. Aggregate Bond ETF', 'type': 'ETF'},
                {'symbol': 'GOVT', 'name': 'iShares U.S. Treasury Bond ETF', 'type': 'ETF'},
                {'symbol': 'LQD', 'name': 'iShares iBoxx $ Investment Grade Corporate Bond ETF', 'type': 'ETF'}
            ],
            'cash': [
                {'symbol': 'SHV', 'name': 'iShares Short Treasury Bond ETF', 'type': 'ETF'},
                {'symbol': 'VMFXX', 'name': 'Vanguard Federal Money Market Fund', 'type': 'Money Market'}
            ],
            'alternatives': [
                {'symbol': 'GLD', 'name': 'SPDR Gold Shares', 'type': 'Commodity ETF'},
                {'symbol': 'VNQ', 'name': 'Vanguard Real Estate ETF', 'type': 'REIT ETF'},
                {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'type': 'Cryptocurrency'}
            ]
        }
        
        # Try to load pre-trained model if it exists
        try:
            self.model = keras.models.load_model('models/recommendation_model.h5')
        except:
            # We'll create a simple model in train_model method
            self.model = None
    
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
    
    def select_assets(self, asset_class, market_data=None, news_data=None, count=3):
        """
        Select specific assets within an asset class based on market data and news.
        
        Args:
            asset_class (str): Asset class (stocks, bonds, etc.)
            market_data (dict): Market data for assets
            news_data (list): News analysis data
            count (int): Number of assets to select
            
        Returns:
            list: Selected assets
        """
        # If no market data or news data, select random assets
        if market_data is None or news_data is None:
            import random
            assets = self.assets.get(asset_class, [])
            if len(assets) <= count:
                return assets
            return random.sample(assets, count)
        
        # TODO: Use market data and news data to make informed selections
        # For now, we'll just return a sample of assets
        import random
        assets = self.assets.get(asset_class, [])
        if len(assets) <= count:
            return assets
        return random.sample(assets, count)
    
    def train_model(self, training_data=None):
        """
        Train a neural network to improve recommendations.
        
        Args:
            training_data (DataFrame): Historical data for training
            
        Returns:
            bool: True if training successful, False otherwise
        """
        # If no training data, create synthetic data for demonstration
        if training_data is None:
            # Create synthetic data
            np.random.seed(42)
            n_samples = 1000
            
            # Features: risk_level, investment_horizon_months, market_volatility, inflation_rate, interest_rate
            X = np.random.rand(n_samples, 5)
            X[:, 0] = X[:, 0] * 10  # risk_level: 0-10
            X[:, 1] = X[:, 1] * 120  # investment_horizon: 0-120 months
            X[:, 2] = X[:, 2] * 0.5  # market_volatility: 0-0.5
            X[:, 3] = X[:, 3] * 0.1  # inflation_rate: 0-0.1
            X[:, 4] = X[:, 4] * 0.1  # interest_rate: 0-0.1
            
            # Target: asset allocations (stocks, bonds, cash, alternatives)
            y = np.zeros((n_samples, 4))
            
            # Generate target allocations based on features
            for i in range(n_samples):
                risk_level = X[i, 0]
                horizon = X[i, 1]
                
                # Determine risk profile
                if risk_level <= 3:
                    profile = 'conservative'
                elif risk_level <= 6:
                    profile = 'moderate'
                elif risk_level <= 8:
                    profile = 'aggressive'
                else:
                    profile = 'very_aggressive'
                
                # Determine horizon category
                if horizon <= 12:
                    horizon_cat = 'short'
                elif horizon <= 60:
                    horizon_cat = 'medium'
                else:
                    horizon_cat = 'long'
                
                # Get allocation
                allocation = self.create_allocation(profile, horizon_cat)
                y[i, 0] = allocation['stocks']
                y[i, 1] = allocation['bonds']
                y[i, 2] = allocation['cash']
                y[i, 3] = allocation['alternatives']
                
                # Add some noise
                y[i] += np.random.normal(0, 0.05, 4)
                
                # Normalize to ensure sum is 1
                y[i] = np.maximum(y[i], 0)
                y[i] /= np.sum(y[i])
            
            # Create training and validation split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create a simple neural network model
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', input_shape=(5,)),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(4, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            
            # Train the model
            model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=0)
            
            # Save the model
            try:
                model.save('models/recommendation_model.h5')
            except:
                pass
            
            self.model = model
            return True
        
        return False
    
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
            if asset_class == 'stocks' and alloc_percent > 0.5:
                recommendations['reasoning'][asset_class] = "High allocation to stocks based on your aggressive risk profile and favorable market conditions."
            elif asset_class == 'bonds' and alloc_percent > 0.4:
                recommendations['reasoning'][asset_class] = "Significant bond allocation provides stability and income in your conservative portfolio."
            elif asset_class == 'cash' and alloc_percent > 0.1:
                recommendations['reasoning'][asset_class] = "Cash position provides liquidity and protection against market volatility."
            elif asset_class == 'alternatives' and alloc_percent > 0.05:
                recommendations['reasoning'][asset_class] = "Alternative investments provide diversification and inflation protection."
        
        return recommendations
