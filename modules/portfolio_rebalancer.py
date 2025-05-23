# modules/portfolio_rebalancer.py
"""
Portfolio rebalancing module for the Investment Recommendation System.
Analyzes current portfolio allocation and provides recommendations for rebalancing.
Relies on portfolio_utils for data loading and analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import functions moved to portfolio_utils
from modules.portfolio_utils import (
    analyze_current_vs_target, # Use the analysis function from utils
    load_portfolio # To get portfolio data if needed directly (though analyze_current_vs_target handles it)
)
# Removed import of execute_query, get_combined_value_cad, convert_usd_to_cad
# Removed load_target_allocation, save_target_allocation, get_current_allocation, analyze_current_vs_target (they are now in portfolio_utils)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Functions that REMAIN in portfolio_rebalancer.py (Logic/Calculation based) ---

def suggest_specific_buys(analysis, risk_level=5):
    """
    Suggest specific assets to buy based on rebalancing needs and risk level.
    Uses the analysis dictionary generated by portfolio_utils.analyze_current_vs_target.

    Args:
        analysis (dict): The analysis result from portfolio_utils.analyze_current_vs_target.
        risk_level (int): Risk tolerance level (1-10).

    Returns:
        list: Specific buy recommendations [{'symbol': str, 'name': str, 'asset_type': str, 'action': 'Buy', 'amount': float, 'description': str}].
    """
    specific_buys = []
    buy_types = {k: v for k, v in analysis.get("actionable_items", {}).items() if v.get("action") == "buy"}

    for asset_type, data in buy_types.items():
        amount_to_buy = data.get("rebalance_amount", 0)
        if amount_to_buy <= 0: continue

        suggested_assets = get_suggested_assets(asset_type, risk_level) # This helper remains here
        total_weight = sum(a.get("allocation_weight", 1) for a in suggested_assets)

        for asset in suggested_assets:
            allocation_portion = asset.get("allocation_weight", 1) / total_weight if total_weight > 0 else 0
            asset_amount = amount_to_buy * allocation_portion

            if asset_amount > 0.01: # Only suggest buys over 1 cent
                specific_buys.append({
                    "symbol": asset["symbol"],
                    "name": asset["name"],
                    "asset_type": asset_type,
                    "action": "Buy",
                    "amount": asset_amount,
                    "description": f"Buy ${asset_amount:.2f} of {asset['name']} ({asset['symbol']}) to increase {asset_type} allocation"
                })
    return specific_buys

def get_suggested_assets(asset_type, risk_level):
    """
    Get suggested assets for a given asset type and risk level.
    (Logic remains the same, kept in this module for organization).

    Args:
        asset_type (str): Asset type (stock, etf, bond, etc.).
        risk_level (int): Risk tolerance level (1-10).

    Returns:
        list: Suggested assets [{'symbol': str, 'name': str, 'allocation_weight': float}].
    """
    risk_category = "conservative" if risk_level <= 3 else "moderate" if risk_level <= 7 else "aggressive"
    suggestions = {
        "stock": {
            "conservative": [{"symbol": "XIU.TO", "name": "iShares S&P/TSX 60 Index ETF", "allocation_weight": 0.4}, {"symbol": "XDV.TO", "name": "iShares Canadian Select Dividend Index ETF", "allocation_weight": 0.3}, {"symbol": "ZLB.TO", "name": "BMO Low Volatility Canadian Equity ETF", "allocation_weight": 0.3}],
            "moderate": [{"symbol": "XIC.TO", "name": "iShares Core S&P/TSX Capped Composite Index ETF", "allocation_weight": 0.3}, {"symbol": "XUU.TO", "name": "iShares Core S&P U.S. Total Market Index ETF", "allocation_weight": 0.4}, {"symbol": "XEF.TO", "name": "iShares Core MSCI EAFE IMI Index ETF", "allocation_weight": 0.3}],
            "aggressive": [{"symbol": "XQQ.TO", "name": "Invesco NASDAQ 100 Index ETF", "allocation_weight": 0.3}, {"symbol": "XIT.TO", "name": "iShares S&P/TSX Capped Information Technology Index ETF", "allocation_weight": 0.4}, {"symbol": "XHC.TO", "name": "iShares Global Healthcare Index ETF (CAD-Hedged)", "allocation_weight": 0.3}]
        },
        "etf": {
            "conservative": [{"symbol": "XCNS.TO", "name": "iShares Conservative Strategic Fixed Income ETF", "allocation_weight": 0.4}, {"symbol": "XINC.TO", "name": "iShares Core Income Balanced ETF Portfolio", "allocation_weight": 0.3}, {"symbol": "XBAL.TO", "name": "iShares Core Balanced ETF Portfolio", "allocation_weight": 0.3}],
            "moderate": [{"symbol": "XGRO.TO", "name": "iShares Core Growth ETF Portfolio", "allocation_weight": 0.4}, {"symbol": "XEQT.TO", "name": "iShares Core Equity ETF Portfolio", "allocation_weight": 0.3}, {"symbol": "VGRO.TO", "name": "Vanguard Growth ETF Portfolio", "allocation_weight": 0.3}],
            "aggressive": [{"symbol": "XEQT.TO", "name": "iShares Core Equity ETF Portfolio", "allocation_weight": 0.5}, {"symbol": "VEQT.TO", "name": "Vanguard All-Equity ETF Portfolio", "allocation_weight": 0.5}]
        },
        "bond": {
            "conservative": [{"symbol": "XBB.TO", "name": "iShares Core Canadian Universe Bond Index ETF", "allocation_weight": 0.4}, {"symbol": "ZAG.TO", "name": "BMO Aggregate Bond Index ETF", "allocation_weight": 0.3}, {"symbol": "XSB.TO", "name": "iShares Core Canadian Short Term Bond Index ETF", "allocation_weight": 0.3}],
            "moderate": [{"symbol": "ZSP.TO", "name": "BMO S&P 500 Index ETF", "allocation_weight": 0.4}, {"symbol": "XSQ.TO", "name": "iShares Short Quality Canadian Bond Index ETF", "allocation_weight": 0.3}, {"symbol": "XHY.TO", "name": "iShares U.S. High Yield Bond Index ETF (CAD-Hedged)", "allocation_weight": 0.3}],
            "aggressive": [{"symbol": "XHY.TO", "name": "iShares U.S. High Yield Bond Index ETF (CAD-Hedged)", "allocation_weight": 0.5}, {"symbol": "CHB.TO", "name": "iShares U.S. High Yield Fixed Income Index ETF (CAD-Hedged)", "allocation_weight": 0.5}]
        },
        "cash": {
            "conservative": [{"symbol": "CSAV.TO", "name": "CI High Interest Savings ETF", "allocation_weight": 0.5}, {"symbol": "PSA.TO", "name": "Purpose High Interest Savings ETF", "allocation_weight": 0.5}],
            "moderate": [{"symbol": "CSAV.TO", "name": "CI High Interest Savings ETF", "allocation_weight": 1.0}],
            "aggressive": [{"symbol": "PSA.TO", "name": "Purpose High Interest Savings ETF", "allocation_weight": 1.0}]
        },
        "mutual_fund": {
            "conservative": [{"symbol": "MAW104", "name": "Mawer Balanced Fund", "allocation_weight": 0.5}, {"symbol": "TDB622", "name": "TD Balanced Index Fund - e", "allocation_weight": 0.5}],
            "moderate": [{"symbol": "RBF448", "name": "RBC Select Balanced Portfolio Series F", "allocation_weight": 0.5}, {"symbol": "CIB885", "name": "CI Select 70/30 Equity/Income Managed Portfolio", "allocation_weight": 0.5}],
            "aggressive": [{"symbol": "TDB661", "name": "TD Science & Technology Fund - I", "allocation_weight": 0.5}, {"symbol": "CIB895", "name": "CI Select 90/10 Equity/Income Managed Portfolio", "allocation_weight": 0.5}]
        },
        "crypto": {
            "conservative": [{"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 1.0}],
            "moderate": [{"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 0.6}, {"symbol": "ETHH.B.TO", "name": "Purpose Ether ETF", "allocation_weight": 0.4}],
            "aggressive": [{"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 0.4}, {"symbol": "ETHH.B.TO", "name": "Purpose Ether ETF", "allocation_weight": 0.4}, {"symbol": "QDINI.NE", "name": "3iQ Defi Innovations Index ETF", "allocation_weight": 0.2}]
        }
    }
    asset_suggestions = suggestions.get(asset_type, {}).get(risk_category, [])
    if not asset_suggestions and risk_category != "moderate":
        asset_suggestions = suggestions.get(asset_type, {}).get("moderate", [])
    if not asset_suggestions:
        asset_suggestions = [{"symbol": "PLACEHOLDER", "name": f"Recommended {asset_type.capitalize()}", "allocation_weight": 1.0}]
    return asset_suggestions

def create_rebalance_plan(portfolio, risk_level=5):
    """
    Create a comprehensive rebalancing plan.
    Uses analyze_current_vs_target from portfolio_utils.

    Args:
        portfolio (dict): Portfolio data from load_portfolio().
        risk_level (int): Risk tolerance level (1-10).

    Returns:
        dict: Rebalancing plan including specific buy/sell actions.
    """
    # Analyze current vs target allocation using the function now in portfolio_utils
    analysis = analyze_current_vs_target(portfolio)

    # Generate specific sell recommendations based on analysis
    specific_sells = []
    symbol_allocation = analysis.get("symbol_allocation", {})
    for asset_type, data in analysis.get("actionable_items", {}).items():
        if data.get("action") == "sell":
            assets_of_type = {k: v for k, v in symbol_allocation.items() if v.get("asset_type") == asset_type}
            sorted_assets = dict(sorted(assets_of_type.items(), key=lambda item: item[1].get("value_cad", 0), reverse=True))
            amount_to_sell = abs(data.get("rebalance_amount", 0))

            for symbol, asset_data in sorted_assets.items():
                max_sell = min(amount_to_sell, asset_data.get("value_cad", 0) * 0.9) # Sell up to 90%
                if max_sell > 0.01:
                    specific_sells.append({
                        "symbol": symbol, "name": asset_data.get("name", symbol), "asset_type": asset_type,
                        "action": "Sell", "amount": max_sell,
                        "description": f"Sell ${max_sell:.2f} of {symbol} to reduce {asset_type} allocation"
                    })
                    amount_to_sell -= max_sell
                    if amount_to_sell <= 0.01: break

    # Generate specific buy recommendations using the helper function in this module
    specific_buys = suggest_specific_buys(analysis, risk_level)

    # Combine and sort recommendations
    all_recommendations = specific_sells + specific_buys
    sorted_recommendations = sorted(all_recommendations, key=lambda x: (0 if x["action"] == "Sell" else 1, -abs(x["amount"])))

    # Calculate summary stats
    total_sells = sum(rec["amount"] for rec in all_recommendations if rec["action"] == "Sell")
    total_buys = sum(rec["amount"] for rec in all_recommendations if rec["action"] == "Buy")
    current_total_drift = sum(abs(data.get("drift_percentage", 0)) for data in analysis.get("drift_analysis", {}).values())

    # Update the analysis dict with the specific recommendations generated here
    analysis["specific_recommendations"] = sorted_recommendations
    analysis["summary"] = {
        "total_sells": total_sells,
        "total_buys": total_buys,
        "net_cash_impact": total_sells - total_buys,
        "current_drift": current_total_drift,
        "action_count": len(sorted_recommendations)
    }

    # Return the modified analysis dictionary as the plan
    return analysis



# # modules/portfolio_rebalancer.py
# """
# Portfolio rebalancing module for the Investment Recommendation System.
# Analyzes current portfolio allocation and provides recommendations for rebalancing.
# """
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import logging
# from modules.db_utils import execute_query
# from modules.portfolio_utils import get_combined_value_cad, convert_usd_to_cad

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def get_current_allocation(portfolio):
#     """
#     Calculate the current allocation of the portfolio by asset type and by symbol.
    
#     Args:
#         portfolio (dict): Portfolio data
        
#     Returns:
#         tuple: (total_value_cad, asset_type_allocation, symbol_allocation)
#     """
#     if not portfolio:
#         return 0, {}, {}
    
#     # Group investments by asset type
#     asset_type_values = {}
#     symbol_values = {}
    
#     total_value_cad = 0
    
#     for inv_id, inv in portfolio.items():
#         asset_type = inv.get("asset_type", "stock")
#         symbol = inv.get("symbol", "")
#         current_value = float(inv.get("current_value", 0))
#         currency = inv.get("currency", "USD")
        
#         # Convert to CAD if necessary
#         if currency == "USD":
#             value_cad = convert_usd_to_cad(current_value)
#         else:
#             value_cad = current_value
        
#         # Add to asset type totals
#         if asset_type not in asset_type_values:
#             asset_type_values[asset_type] = 0
#         asset_type_values[asset_type] += value_cad
        
#         # Add to symbol totals
#         if symbol not in symbol_values:
#             symbol_values[symbol] = {
#                 "value_cad": 0,
#                 "currency": currency,
#                 "asset_type": asset_type,
#                 "name": inv.get("name", symbol)
#             }
#         symbol_values[symbol]["value_cad"] += value_cad
        
#         # Add to total value
#         total_value_cad += value_cad
    
#     # Calculate percentages for asset types
#     asset_type_allocation = {}
#     for asset_type, value in asset_type_values.items():
#         percentage = (value / total_value_cad * 100) if total_value_cad > 0 else 0
#         asset_type_allocation[asset_type] = {
#             "value": value,
#             "percentage": percentage
#         }
    
#     # Calculate percentages for symbols
#     symbol_allocation = {}
#     for symbol, data in symbol_values.items():
#         percentage = (data["value_cad"] / total_value_cad * 100) if total_value_cad > 0 else 0
#         symbol_allocation[symbol] = {
#             "value_cad": data["value_cad"],
#             "percentage": percentage,
#             "currency": data["currency"],
#             "asset_type": data["asset_type"],
#             "name": data.get("name", symbol)
#         }
    
#     return total_value_cad, asset_type_allocation, symbol_allocation

# def load_target_allocation():
#     """
#     Load the target allocation from the database.
    
#     Returns:
#         dict: Target allocation by asset type
#     """
#     # First, try to load from the database
#     query = """
#     SELECT * FROM target_allocation ORDER BY last_updated DESC LIMIT 1;
#     """
#     result = execute_query(query, fetchone=True)
    
#     if result:
#         # Convert RealDictRow to regular dict
#         target_dict = dict(result)
        
#         # Check if allocation is already a dict or needs to be parsed from JSON
#         allocation = target_dict.get('allocation', {})
#         if isinstance(allocation, str):
#             # Parse the JSON stored in the allocation column
#             import json
#             allocation = json.loads(allocation)
        
#         return allocation
    
#     # If no target allocation is found, return a default allocation
#     return {
#         "stock": 40,
#         "etf": 30,
#         "bond": 20,
#         "cash": 5,
#         "mutual_fund": 5,
#         "crypto": 0
#     }


# def save_target_allocation(allocation):
#     """
#     Save the target allocation to the database.
    
#     Args:
#         allocation (dict): Target allocation by asset type
        
#     Returns:
#         bool: Success status
#     """
#     try:
#         # Convert allocation to JSON string
#         import json
#         allocation_json = json.dumps(allocation)
        
#         # Check if the table exists
#         check_query = """
#         SELECT EXISTS (
#             SELECT FROM information_schema.tables 
#             WHERE table_name = 'target_allocation'
#         );
#         """
#         table_exists = execute_query(check_query, fetchone=True)
        
#         if not table_exists or not table_exists.get('exists', False):
#             # Create the table if it doesn't exist
#             create_query = """
#             CREATE TABLE target_allocation (
#                 id SERIAL PRIMARY KEY,
#                 allocation JSONB NOT NULL,
#                 last_updated TIMESTAMP NOT NULL
#             );
#             """
#             execute_query(create_query, commit=True)
        
#         # Insert the new allocation
#         insert_query = """
#         INSERT INTO target_allocation (allocation, last_updated)
#         VALUES (%s, %s);
#         """
        
#         params = (
#             allocation_json,
#             datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         )
        
#         result = execute_query(insert_query, params, commit=True)
        
#         return result is not None
#     except Exception as e:
#         logger.error(f"Error saving target allocation: {e}")
#         return False

# def analyze_current_vs_target(portfolio):
#     """
#     Compare current allocation to target allocation and calculate drift.
    
#     Args:
#         portfolio (dict): Portfolio data
        
#     Returns:
#         dict: Analysis results including drift and action items
#     """
#     # Get current allocation
#     total_value_cad, asset_type_allocation, symbol_allocation = get_current_allocation(portfolio)
    
#     # Get target allocation
#     target_allocation = load_target_allocation()
    
#     # Normalize target allocation percentages to ensure they sum to 100
#     target_sum = sum(target_allocation.values())
#     normalized_target = {k: (v / target_sum * 100) if target_sum > 0 else 0 
#                          for k, v in target_allocation.items()}
    
#     # Calculate drift for each asset type
#     drift_analysis = {}
#     for asset_type, target_pct in normalized_target.items():
#         current_data = asset_type_allocation.get(asset_type, {"value": 0, "percentage": 0})
#         current_pct = current_data["percentage"]
#         current_value = current_data["value"]
        
#         # Calculate drift (current - target)
#         drift_pct = current_pct - target_pct
        
#         # Calculate amount needed to rebalance
#         # If drift is positive, we need to sell; if negative, we need to buy
#         target_value = (target_pct / 100) * total_value_cad
#         rebalance_amount = target_value - current_value
        
#         drift_analysis[asset_type] = {
#             "current_percentage": current_pct,
#             "target_percentage": target_pct,
#             "drift_percentage": drift_pct,
#             "current_value": current_value,
#             "target_value": target_value,
#             "rebalance_amount": rebalance_amount,
#             "action": "sell" if drift_pct > 0 else "buy" if drift_pct < 0 else "hold",
#             # Threshold for actionable items (typically 5% or more)
#             "is_actionable": abs(drift_pct) >= 5
#         }
    
#     # Get actionable items sorted by absolute drift
#     actionable_items = {k: v for k, v in drift_analysis.items() if v["is_actionable"]}
#     sorted_actionable = dict(sorted(
#         actionable_items.items(), 
#         key=lambda item: abs(item[1]["drift_percentage"]), 
#         reverse=True
#     ))
    
#     # Create rebalancing recommendations
#     recommendations = []
#     for asset_type, data in sorted_actionable.items():
#         action = data["action"]
#         amount = abs(data["rebalance_amount"])
        
#         if action == "buy":
#             recommendations.append({
#                 "asset_type": asset_type,
#                 "action": "Buy",
#                 "amount": amount,
#                 "description": f"Increase {asset_type} allocation by ${amount:.2f} to reach target of {data['target_percentage']:.1f}%"
#             })
#         elif action == "sell":
#             recommendations.append({
#                 "asset_type": asset_type,
#                 "action": "Sell",
#                 "amount": amount,
#                 "description": f"Decrease {asset_type} allocation by ${amount:.2f} to reach target of {data['target_percentage']:.1f}%"
#             })
    
#     # Determine which specific assets to buy or sell
#     # For asset types we need to sell, find the most overweight assets
#     # For asset types we need to buy, find the most underweight assets or suggest new assets
#     specific_recommendations = []
    
#     # First, handle sells (reduce overweight asset types)
#     for asset_type, data in sorted_actionable.items():
#         if data["action"] == "sell":
#             # Find assets of this type sorted by value
#             assets_of_type = {k: v for k, v in symbol_allocation.items() 
#                              if v["asset_type"] == asset_type}
#             sorted_assets = dict(sorted(
#                 assets_of_type.items(), 
#                 key=lambda item: item[1]["value_cad"], 
#                 reverse=True
#             ))
            
#             amount_to_sell = abs(data["rebalance_amount"])
#             for symbol, asset_data in sorted_assets.items():
#                 # Limit sell amount to a maximum of 90% of the asset's value to avoid zeroing out
#                 max_sell = min(amount_to_sell, asset_data["value_cad"] * 0.9)
#                 if max_sell > 0:
#                     specific_recommendations.append({
#                         "symbol": symbol,
#                         "name": asset_data.get("name", symbol),
#                         "asset_type": asset_type,
#                         "action": "Sell",
#                         "amount": max_sell,
#                         "description": f"Sell ${max_sell:.2f} of {symbol} to reduce {asset_type} allocation"
#                     })
#                     amount_to_sell -= max_sell
#                     if amount_to_sell <= 0:
#                         break
    
#     # Prepare a result with all the analysis
#     result = {
#         "total_value_cad": total_value_cad,
#         "current_allocation": asset_type_allocation,
#         "symbol_allocation": symbol_allocation,
#         "target_allocation": normalized_target,
#         "drift_analysis": drift_analysis,
#         "actionable_items": sorted_actionable,
#         "rebalancing_recommendations": recommendations,
#         "specific_recommendations": specific_recommendations
#     }
    
#     return result

# def suggest_specific_buys(analysis, risk_level=5):
#     """
#     Suggest specific assets to buy based on rebalancing needs and risk level.
    
#     Args:
#         analysis (dict): The analysis result from analyze_current_vs_target
#         risk_level (int): Risk tolerance level (1-10)
        
#     Returns:
#         list: Specific buy recommendations
#     """
#     specific_buys = []
    
#     # Get asset types that need buying
#     buy_types = {k: v for k, v in analysis["actionable_items"].items() 
#                 if v["action"] == "buy"}
    
#     # For each asset type that needs buying, suggest appropriate investments
#     for asset_type, data in buy_types.items():
#         amount_to_buy = data["rebalance_amount"]
        
#         # Get suggested assets for this type based on risk level
#         suggested_assets = get_suggested_assets(asset_type, risk_level)
        
#         for asset in suggested_assets:
#             # Allocate portion of the amount to each suggested asset
#             allocation_portion = asset.get("allocation_weight", 1) / sum(a.get("allocation_weight", 1) for a in suggested_assets)
#             asset_amount = amount_to_buy * allocation_portion
            
#             if asset_amount > 0:
#                 specific_buys.append({
#                     "symbol": asset["symbol"],
#                     "name": asset["name"],
#                     "asset_type": asset_type,
#                     "action": "Buy",
#                     "amount": asset_amount,
#                     "description": f"Buy ${asset_amount:.2f} of {asset['name']} ({asset['symbol']}) to increase {asset_type} allocation"
#                 })
    
#     return specific_buys

# def get_suggested_assets(asset_type, risk_level):
#     """
#     Get suggested assets for a given asset type and risk level.
    
#     Args:
#         asset_type (str): Asset type (stock, etf, bond, etc.)
#         risk_level (int): Risk tolerance level (1-10)
        
#     Returns:
#         list: Suggested assets
#     """
#     # This function would ideally query a database of recommended investments
#     # For this implementation, we'll use hard-coded suggestions
    
#     # Adjust suggestions based on risk level
#     risk_category = "conservative" if risk_level <= 3 else "moderate" if risk_level <= 7 else "aggressive"
    
#     # Suggestions for different asset types and risk levels
#     suggestions = {
#         "stock": {
#             "conservative": [
#                 {"symbol": "XIU.TO", "name": "iShares S&P/TSX 60 Index ETF", "allocation_weight": 0.4},
#                 {"symbol": "XDV.TO", "name": "iShares Canadian Select Dividend Index ETF", "allocation_weight": 0.3},
#                 {"symbol": "ZLB.TO", "name": "BMO Low Volatility Canadian Equity ETF", "allocation_weight": 0.3}
#             ],
#             "moderate": [
#                 {"symbol": "XIC.TO", "name": "iShares Core S&P/TSX Capped Composite Index ETF", "allocation_weight": 0.3},
#                 {"symbol": "XUU.TO", "name": "iShares Core S&P U.S. Total Market Index ETF", "allocation_weight": 0.4},
#                 {"symbol": "XEF.TO", "name": "iShares Core MSCI EAFE IMI Index ETF", "allocation_weight": 0.3}
#             ],
#             "aggressive": [
#                 {"symbol": "XQQ.TO", "name": "Invesco NASDAQ 100 Index ETF", "allocation_weight": 0.3},
#                 {"symbol": "XIT.TO", "name": "iShares S&P/TSX Capped Information Technology Index ETF", "allocation_weight": 0.4},
#                 {"symbol": "XHC.TO", "name": "iShares Global Healthcare Index ETF (CAD-Hedged)", "allocation_weight": 0.3}
#             ]
#         },
#         "etf": {
#             "conservative": [
#                 {"symbol": "XCNS.TO", "name": "iShares Conservative Strategic Fixed Income ETF", "allocation_weight": 0.4},
#                 {"symbol": "XINC.TO", "name": "iShares Core Income Balanced ETF Portfolio", "allocation_weight": 0.3},
#                 {"symbol": "XBAL.TO", "name": "iShares Core Balanced ETF Portfolio", "allocation_weight": 0.3}
#             ],
#             "moderate": [
#                 {"symbol": "XGRO.TO", "name": "iShares Core Growth ETF Portfolio", "allocation_weight": 0.4},
#                 {"symbol": "XEQT.TO", "name": "iShares Core Equity ETF Portfolio", "allocation_weight": 0.3},
#                 {"symbol": "VGRO.TO", "name": "Vanguard Growth ETF Portfolio", "allocation_weight": 0.3}
#             ],
#             "aggressive": [
#                 {"symbol": "XEQT.TO", "name": "iShares Core Equity ETF Portfolio", "allocation_weight": 0.5},
#                 {"symbol": "VEQT.TO", "name": "Vanguard All-Equity ETF Portfolio", "allocation_weight": 0.5}
#             ]
#         },
#         "bond": {
#             "conservative": [
#                 {"symbol": "XBB.TO", "name": "iShares Core Canadian Universe Bond Index ETF", "allocation_weight": 0.4},
#                 {"symbol": "ZAG.TO", "name": "BMO Aggregate Bond Index ETF", "allocation_weight": 0.3},
#                 {"symbol": "XSB.TO", "name": "iShares Core Canadian Short Term Bond Index ETF", "allocation_weight": 0.3}
#             ],
#             "moderate": [
#                 {"symbol": "ZSP.TO", "name": "BMO S&P 500 Index ETF", "allocation_weight": 0.4},
#                 {"symbol": "XSQ.TO", "name": "iShares Short Quality Canadian Bond Index ETF", "allocation_weight": 0.3},
#                 {"symbol": "XHY.TO", "name": "iShares U.S. High Yield Bond Index ETF (CAD-Hedged)", "allocation_weight": 0.3}
#             ],
#             "aggressive": [
#                 {"symbol": "XHY.TO", "name": "iShares U.S. High Yield Bond Index ETF (CAD-Hedged)", "allocation_weight": 0.5},
#                 {"symbol": "CHB.TO", "name": "iShares U.S. High Yield Fixed Income Index ETF (CAD-Hedged)", "allocation_weight": 0.5}
#             ]
#         },
#         "cash": {
#             "conservative": [
#                 {"symbol": "CSAV.TO", "name": "CI High Interest Savings ETF", "allocation_weight": 0.5},
#                 {"symbol": "PSA.TO", "name": "Purpose High Interest Savings ETF", "allocation_weight": 0.5}
#             ],
#             "moderate": [
#                 {"symbol": "CSAV.TO", "name": "CI High Interest Savings ETF", "allocation_weight": 1.0}
#             ],
#             "aggressive": [
#                 {"symbol": "PSA.TO", "name": "Purpose High Interest Savings ETF", "allocation_weight": 1.0}
#             ]
#         },
#         "mutual_fund": {
#             "conservative": [
#                 {"symbol": "MAW104", "name": "Mawer Balanced Fund", "allocation_weight": 0.5},
#                 {"symbol": "TDB622", "name": "TD Balanced Index Fund - e", "allocation_weight": 0.5}
#             ],
#             "moderate": [
#                 {"symbol": "RBF448", "name": "RBC Select Balanced Portfolio Series F", "allocation_weight": 0.5},
#                 {"symbol": "CIB885", "name": "CI Select 70/30 Equity/Income Managed Portfolio", "allocation_weight": 0.5}
#             ],
#             "aggressive": [
#                 {"symbol": "TDB661", "name": "TD Science & Technology Fund - I", "allocation_weight": 0.5},
#                 {"symbol": "CIB895", "name": "CI Select 90/10 Equity/Income Managed Portfolio", "allocation_weight": 0.5}
#             ]
#         },
#         "crypto": {
#             "conservative": [
#                 {"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 1.0}
#             ],
#             "moderate": [
#                 {"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 0.6},
#                 {"symbol": "ETHH.B.TO", "name": "Purpose Ether ETF", "allocation_weight": 0.4}
#             ],
#             "aggressive": [
#                 {"symbol": "BTCC.B.TO", "name": "Purpose Bitcoin ETF", "allocation_weight": 0.4},
#                 {"symbol": "ETHH.B.TO", "name": "Purpose Ether ETF", "allocation_weight": 0.4},
#                 {"symbol": "QDINI.NE", "name": "3iQ Defi Innovations Index ETF", "allocation_weight": 0.2}
#             ]
#         }
#     }
    
#     # Get suggestions for the specific asset type and risk category
#     asset_suggestions = suggestions.get(asset_type, {}).get(risk_category, [])
    
#     # If no suggestions available for this specific combination, use moderate risk as fallback
#     if not asset_suggestions and risk_category != "moderate":
#         asset_suggestions = suggestions.get(asset_type, {}).get("moderate", [])
    
#     # If still no suggestions, provide a generic placeholder
#     if not asset_suggestions:
#         asset_suggestions = [
#             {"symbol": "PLACEHOLDER", "name": f"Recommended {asset_type.capitalize()}", "allocation_weight": 1.0}
#         ]
    
#     return asset_suggestions

# def create_rebalance_plan(portfolio, risk_level=5):
#     """
#     Create a comprehensive rebalancing plan.
    
#     Args:
#         portfolio (dict): Portfolio data
#         risk_level (int): Risk tolerance level (1-10)
        
#     Returns:
#         dict: Rebalancing plan
#     """
#     # Analyze current vs target allocation
#     analysis = analyze_current_vs_target(portfolio)
    
#     # Get specific buy recommendations
#     specific_buys = suggest_specific_buys(analysis, risk_level)
    
#     # Add specific sells from the analysis
#     specific_sells = analysis.get("specific_recommendations", [])
    
#     # Combine all recommendations
#     all_recommendations = specific_sells + specific_buys
    
#     # Sort recommendations by action (sells first, then buys) and amount
#     sorted_recommendations = sorted(
#         all_recommendations,
#         key=lambda x: (0 if x["action"] == "Sell" else 1, -abs(x["amount"]))
#     )
    
#     # Calculate rebalancing statistics
#     total_sells = sum(rec["amount"] for rec in all_recommendations if rec["action"] == "Sell")
#     total_buys = sum(rec["amount"] for rec in all_recommendations if rec["action"] == "Buy")
    
#     # Calculate drift reduction (how much closer to target the suggested changes get us)
#     current_total_drift = sum(abs(data["drift_percentage"]) for data in analysis["drift_analysis"].values())
    
#     # Create the final plan
#     rebalance_plan = {
#         "total_value_cad": analysis["total_value_cad"],
#         "current_allocation": analysis["current_allocation"],
#         "target_allocation": analysis["target_allocation"],
#         "drift_analysis": analysis["drift_analysis"],
#         "recommendations": sorted_recommendations,
#         "summary": {
#             "total_sells": total_sells,
#             "total_buys": total_buys,
#             "net_cash_impact": total_sells - total_buys,
#             "current_drift": current_total_drift,
#             "action_count": len(sorted_recommendations)
#         }
#     }
    
#     return rebalance_plan