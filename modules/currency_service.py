# modules/currency_service.py
from datetime import datetime, timedelta
import pandas as pd
from modules.data_provider import data_provider

class CurrencyService:
    """Centralized service for currency conversion operations"""
    
    def __init__(self):
        self._current_rates = {}
        self._last_updated = datetime.min
        self._update_interval = timedelta(hours=1)
    
    def get_exchange_rate(self, from_currency, to_currency):
        """Get current exchange rate with caching"""
        if (datetime.now() - self._last_updated) > self._update_interval:
            self._update_rates()
        
        rate_key = f"{from_currency}_{to_currency}"
        if rate_key in self._current_rates:
            return self._current_rates[rate_key]
        
        # Get rate from data provider
        rate = data_provider.get_exchange_rate(from_currency, to_currency)
        self._current_rates[rate_key] = rate
        return rate
    
    def convert_value(self, amount, from_currency, to_currency):
        """Convert amount between currencies"""
        if from_currency == to_currency:
            return amount
            
        rate = self.get_exchange_rate(from_currency, to_currency)
        return amount * rate
    
    def _update_rates(self):
        """Update all cached exchange rates"""
        # Update common rates
        self._current_rates['USD_CAD'] = data_provider.get_exchange_rate('USD', 'CAD')
        self._current_rates['CAD_USD'] = 1.0 / self._current_rates['USD_CAD']
        self._last_updated = datetime.now()

# Create singleton instance
currency_service = CurrencyService()