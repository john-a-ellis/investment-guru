# modules/portfolio_data_updater.py - Modified update_portfolio_data function
import os
import json
from datetime import datetime
import yfinance as yf

def update_portfolio_data():
    """
    Updates portfolio data with current market prices and performance metrics,
    including mutual fund support
    """
    # Load portfolio
    
    portfolio = load_portfolio()
    
    # Import mutual fund provider
    logger.info("update_portfolio_data in portfolio_data_updater.py is being called")
    from modules.mutual_fund_provider import MutualFundProvider
    mutual_fund_provider = MutualFundProvider()
    
    # First, get current prices for all unique symbols
    symbol_prices = {}
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        asset_type = details.get("asset_type", "stock")
        
        if symbol not in symbol_prices:
            try:
                # Handle mutual funds differently
                if asset_type == "mutual_fund":
                    # Get the most recent price from our mutual fund provider
                    current_price = mutual_fund_provider.get_current_price(symbol)
                    
                    if current_price:
                        # Assume mutual funds are in CAD
                        symbol_prices[symbol] = {
                            "price": current_price,
                            "currency": "CAD"
                        }
                        print(f"Got mutual fund price for {symbol}: {current_price} CAD")
                    else:
                        print(f"No price data available for mutual fund {symbol}")
                        # Use the purchase price as a fallback
                        purchase_price = details.get("purchase_price", 0)
                        symbol_prices[symbol] = {
                            "price": purchase_price,
                            "currency": "CAD"
                        }
                else:
                    # For stocks, ETFs, etc., use yfinance
                    ticker = yf.Ticker(symbol)
                    price_data = ticker.history(period="1d")
                    
                    if not price_data.empty:
                        # Get the price directly from yfinance
                        current_price = price_data['Close'].iloc[-1]
                        
                        # Determine currency based on symbol
                        is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
                        currency = "CAD" if is_canadian else "USD"
                        
                        # Store the price and currency for this symbol
                        symbol_prices[symbol] = {
                            "price": current_price,
                            "currency": currency
                        }
                        
                        print(f"Got price for {symbol}: {current_price} {currency}")
                    else:
                        print(f"No price data available for {symbol}")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
    
    # Now update each investment with the consistent price
    for investment_id, details in portfolio.items():
        symbol = details.get("symbol", "")
        shares = details.get("shares", 0)
        purchase_price = details.get("purchase_price", 0)
        asset_type = details.get("asset_type", "stock")
        
        # Get the price data for this symbol
        symbol_data = symbol_prices.get(symbol)
        
        if symbol_data:
            current_price = symbol_data["price"]
            currency = symbol_data["currency"]
            
            # Store the currency for display purposes
            details["currency"] = currency
            details["asset_type"] = asset_type  # Ensure asset_type is stored
                
            # Calculate current value and gain/loss
            current_value = shares * current_price
            gain_loss = current_value - (shares * purchase_price)
            gain_loss_percent = (current_price / purchase_price - 1) * 100 if purchase_price > 0 else 0
            
            # Update investment details
            portfolio[investment_id].update({
                "current_price": current_price,
                "current_value": current_value,
                "gain_loss": gain_loss,
                "gain_loss_percent": gain_loss_percent,
                "currency": currency,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save updated portfolio
    save_portfolio(portfolio)
    
    return portfolio

# In components/portfolio_management.py - Modified create_portfolio_management_component

def create_portfolio_management_component():
    """
    Creates a component for tracking actual investments with performance data
    """
    return dbc.Card([
        dbc.CardHeader("Portfolio Management"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Add New Investment"),
                    dbc.InputGroup([
                        dbc.Input(id="investment-symbol-input", placeholder="Symbol (e.g., MFC.TO, MAW104)"),
                        dbc.Input(id="investment-shares-input", type="number", placeholder="Number of Shares/Units"),
                        dbc.Input(id="investment-price-input", type="number", placeholder="Purchase Price"),
                        dbc.Input(id="investment-date-input", type="date", 
                                 value=datetime.now().strftime("%Y-%m-%d")),
                        dbc.Select(
                            id="investment-type-select",
                            options=[
                                {"label": "Stock", "value": "stock"},
                                {"label": "ETF", "value": "etf"},
                                {"label": "Mutual Fund", "value": "mutual_fund"},
                                {"label": "Cryptocurrency", "value": "crypto"},
                                {"label": "Bond", "value": "bond"},
                                {"label": "Cash", "value": "cash"}
                            ],
                            value="stock",
                            placeholder="Asset Type"
                        ),
                        dbc.Button("Add Investment", id="add-investment-button", color="success")
                    ]),
                    html.Div(id="add-investment-feedback", className="mt-2")
                ], width=12)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H5("Current Portfolio"),
                    html.Div(id="portfolio-table")
                ], width=12)
            ])
        ])
    ])

# Modify add_investment in modules/portfolio_data_updater.py

def add_investment(symbol, shares, purchase_price, purchase_date, asset_type="stock"):
    """
    Add a new investment to the portfolio
    
    Args:
        symbol (str): Investment symbol
        shares (float): Number of shares
        purchase_price (float): Purchase price per share
        purchase_date (str): Purchase date in YYYY-MM-DD format
        asset_type (str): Type of asset (stock, etf, mutual_fund, etc.)
        
    Returns:
        bool: Success status
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Generate unique ID for this investment
    import uuid
    investment_id = str(uuid.uuid4())
    
    # Calculate initial values
    initial_value = shares * purchase_price
    
    # Default current price to purchase price
    current_price = purchase_price
    
    # For mutual funds, try to get price from our provider
    if asset_type == "mutual_fund":
        from modules.mutual_fund_provider import MutualFundProvider
        mutual_fund_provider = MutualFundProvider()
        
        # Get the most recent price
        fund_price = mutual_fund_provider.get_current_price(symbol)
        if fund_price:
            current_price = fund_price
            
        # Always treat mutual funds as CAD
        currency = "CAD"
    else:
        # Determine currency based on symbol for non-mutual fund investments
        is_canadian = symbol.endswith(".TO") or symbol.endswith(".V") or "-CAD" in symbol
        currency = "CAD" if is_canadian else "USD"
        
        # Find if we already have this symbol in our portfolio for the current price
        for existing_inv in portfolio.values():
            if existing_inv.get("symbol") == symbol:
                current_price = existing_inv.get("current_price", purchase_price)
                break
        
        # If we don't have an existing price, try to get it from the API
        if current_price == purchase_price and asset_type != "mutual_fund":
            try:
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period="1d")
                
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
            except Exception as e:
                print(f"Error getting initial price for {symbol}: {e}")
    
    # Create investment entry
    investment = {
        "symbol": symbol,
        "shares": float(shares),
        "purchase_price": float(purchase_price),
        "purchase_date": purchase_date,
        "asset_type": asset_type,
        "current_price": float(current_price),
        "current_value": float(shares) * float(current_price),
        "gain_loss": float(shares) * (float(current_price) - float(purchase_price)),
        "gain_loss_percent": ((float(current_price) / float(purchase_price)) - 1) * 100 if float(purchase_price) > 0 else 0,
        "currency": currency,
        "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to portfolio
    portfolio[investment_id] = investment
    
    # Save portfolio
    return save_portfolio(portfolio)

def remove_investment(investment_id):
    """
    Remove an investment from the portfolio
    
    Args:
        investment_id (str): ID of investment to remove
        
    Returns:
        bool: Success status
    """
    # Load current portfolio
    portfolio = load_portfolio()
    
    # Remove investment
    if investment_id in portfolio:
        del portfolio[investment_id]
        
        # Save updated portfolio
        return save_portfolio(portfolio)
    
    return False
def load_portfolio():
    """
    Load portfolio data from storage file
    """
    try:
        if os.path.exists('data/portfolio.json'):
            with open('data/portfolio.json', 'r') as f:
                return json.load(f)
        else:
            # Default empty portfolio if no file exists
            return {}
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return {}

def save_portfolio(portfolio):
    """
    Save portfolio data to storage file
    """
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/portfolio.json', 'w') as f:
            json.dump(portfolio, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        return False