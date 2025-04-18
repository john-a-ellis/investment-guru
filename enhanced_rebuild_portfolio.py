
from modules.portfolio_utils import load_portfolio, get_cash_positions, load_transactions, load_cash_flows, load_currency_exchanges, load_cash_flows
from modules.dividend_utils import load_dividends
from modules.data_provider import data_provider

def enhanced_rebuild_portfolio():
    """
    Enhanced rebuilding of the entire portfolio from transaction history,
    including cash positions, dividends, and currency exchanges.
    
    This function:
    1. Gets all transactions, cash flows, dividends, and currency exchanges chronologically
    2. Rebuilds portfolio positions transaction by transaction
    3. Recalculates cash positions for all currencies
    4. Updates both portfolio and cash_positions tables in the database
    5. Returns a comprehensive report of changes made
    
    Returns:
        dict: Detailed report of the rebuild process
    """
    import logging
    from datetime import datetime, timedelta
    import uuid
    from modules.db_utils import execute_query
    
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced portfolio rebuild from transaction history")
    
    # Step 1: Load all history (chronologically sorted)
    # ------------------------------------------------
    
    # Load all transactions
    all_transactions = load_transactions()
    
    # Load all cash flows (deposits/withdrawals)
    all_cash_flows = load_cash_flows()
    
    # Load all dividends
    all_dividends = load_dividends()
    
    # Load all currency exchanges
    all_currency_exchanges = load_currency_exchanges()
    
    # Step 2: Combine and sort all financial events chronologically
    # -------------------------------------------------------------
    financial_events = []
    
    # Add transactions
    for tx_id, tx in all_transactions.items():
        try:
            tx_date = datetime.strptime(tx['date'], '%Y-%m-%d')
            financial_events.append({
                'date': tx_date,
                'type': 'transaction',
                'subtype': tx['type'].lower(),  # 'buy', 'sell', 'drip'
                'symbol': tx['symbol'].upper().strip(),
                'shares': float(tx['shares']),
                'price': float(tx['price']),
                'amount': float(tx['amount']),
                'notes': tx.get('notes', ''),
                'id': tx_id,
                'currency': tx.get('currency', 'USD')  # Default to USD if not specified
            })
        except Exception as e:
            logger.error(f"Error processing transaction {tx_id}: {e}")
    
    # Add cash flows
    for flow in all_cash_flows:
        try:
            flow_date = datetime.strptime(flow['date'], '%Y-%m-%d')
            financial_events.append({
                'date': flow_date,
                'type': 'cash_flow',
                'subtype': flow['flow_type'].lower(),  # 'deposit', 'withdrawal'
                'amount': float(flow['amount']),
                'currency': flow['currency'],
                'notes': flow.get('description', ''),
                'id': flow['id']
            })
        except Exception as e:
            logger.error(f"Error processing cash flow {flow.get('id', 'unknown')}: {e}")
    
    # Add dividends
    for div in all_dividends:
        try:
            div_date = datetime.strptime(div['dividend_date'], '%Y-%m-%d')
            financial_events.append({
                'date': div_date,
                'type': 'dividend',
                'symbol': div['symbol'].upper().strip(),
                'amount_per_share': float(div['amount_per_share']),
                'shares_held': float(div['shares_held']),
                'total_amount': float(div['total_amount']),
                'currency': div['currency'],
                'is_drip': div.get('is_drip', False),
                'drip_shares': float(div.get('drip_shares', 0)),
                'drip_price': float(div.get('drip_price', 0)),
                'notes': div.get('notes', ''),
                'id': div['id']
            })
        except Exception as e:
            logger.error(f"Error processing dividend {div.get('id', 'unknown')}: {e}")
    
    # Add currency exchanges
    for ex in all_currency_exchanges:
        try:
            ex_date = datetime.strptime(ex['date'], '%Y-%m-%d')
            financial_events.append({
                'date': ex_date,
                'type': 'currency_exchange',
                'from_currency': ex['from_currency'],
                'from_amount': float(ex['from_amount']),
                'to_currency': ex['to_currency'],
                'to_amount': float(ex['to_amount']),
                'rate': float(ex['rate']),
                'notes': ex.get('description', ''),
                'id': ex['id']
            })
        except Exception as e:
            logger.error(f"Error processing currency exchange {ex.get('id', 'unknown')}: {e}")
    
    # Sort all events chronologically
    financial_events.sort(key=lambda x: x['date'])
    
    # Step 3: Initialize portfolio and cash tracking
    # ---------------------------------------------
    rebuilt_positions = {}  # Track investment positions
    cash_positions = {      # Track cash balances by currency
        'CAD': 0.0,
        'USD': 0.0
        # Add other currencies as needed
    }
    rebuild_log = []        # Log all actions for reporting
    
    # Step 4: Process all events chronologically to rebuild portfolio
    # --------------------------------------------------------------
    for event in financial_events:
        event_date = event['date']
        event_type = event['type']
        
        # Log this event
        rebuild_log.append({
            'date': event_date.strftime('%Y-%m-%d'),
            'type': event_type,
            'details': event,
            'cash_before': cash_positions.copy()
        })
        
        # Process based on event type
        if event_type == 'transaction':
            # Handle buy/sell/drip transactions
            symbol = event['symbol']
            tx_type = event['subtype']
            shares = event['shares']
            price = event['price']
            amount = event['amount']
            currency = event.get('currency', 'USD')  # Default to USD
            
            # Initialize position if it doesn't exist
            if symbol not in rebuilt_positions:
                rebuilt_positions[symbol] = {
                    'shares': 0.0,
                    'book_value': 0.0,
                    'last_date': None,
                    'asset_type': None,
                    'currency': currency,
                    'name': None,
                    'transactions': []
                }
            
            position = rebuilt_positions[symbol]
            
            # Process based on transaction type
            if tx_type == 'buy':
                # For buy: add shares, increase book value, decrease cash
                prev_shares = position['shares']
                prev_book_value = position['book_value']
                
                position['shares'] += shares
                position['book_value'] += amount
                position['currency'] = currency  # Update currency
                
                # Update cash position (deduct amount)
                cash_positions[currency] -= amount
                
                # Update position date if earlier
                if not position['last_date'] or event_date < position['last_date']:
                    position['last_date'] = event_date
                
            elif tx_type == 'sell':
                # For sell: remove shares, reduce book value proportionally, increase cash
                if position['shares'] <= 0:
                    logger.warning(f"Cannot sell {shares} shares of {symbol} on {event_date} - position shows zero shares")
                    rebuild_log[-1]['error'] = 'Attempting to sell from zero position'
                    continue
                
                if shares > position['shares']:
                    logger.warning(f"Cannot sell {shares} shares of {symbol} on {event_date} - only {position['shares']} shares available")
                    rebuild_log[-1]['error'] = 'Selling more shares than owned'
                    continue
                
                # Calculate proportion of position being sold
                proportion_sold = shares / position['shares']
                book_value_sold = position['book_value'] * proportion_sold
                
                # Update position
                position['shares'] -= shares
                position['book_value'] -= book_value_sold
                
                # Update cash position (add sale amount)
                cash_positions[currency] += amount
                
                # If shares become very small, set exactly to zero to avoid float precision issues
                if abs(position['shares']) < 0.000001:
                    position['shares'] = 0.0
                    position['book_value'] = 0.0
                
            elif tx_type == 'drip':
                # For DRIP: add shares but don't change book value (reinvested dividends)
                # No cash impact since dividends are automatically reinvested
                position['shares'] += shares
                
            # Record transaction in position history
            position['transactions'].append({
                'date': event_date,
                'type': tx_type,
                'shares': shares,
                'price': price,
                'amount': amount,
                'running_shares': position['shares'],
                'running_book_value': position['book_value']
            })
        
        elif event_type == 'cash_flow':
            # Handle deposits and withdrawals
            flow_type = event['subtype']
            amount = event['amount']
            currency = event['currency']
            
            # Ensure currency exists in tracking
            if currency not in cash_positions:
                cash_positions[currency] = 0.0
            
            # Add or subtract from cash based on flow type
            if flow_type == 'deposit':
                cash_positions[currency] += amount
            elif flow_type == 'withdrawal':
                cash_positions[currency] -= amount
        
        elif event_type == 'dividend':
            # Handle dividend payments
            symbol = event['symbol']
            is_drip = event['is_drip']
            total_amount = event['total_amount']
            currency = event['currency']
            
            # For non-DRIP dividends, increase cash
            if not is_drip:
                # Ensure currency exists in tracking
                if currency not in cash_positions:
                    cash_positions[currency] = 0.0
                
                cash_positions[currency] += total_amount
            
            # Note: DRIP shares are handled by transaction events of type 'drip'
        
        elif event_type == 'currency_exchange':
            # Handle currency exchanges
            from_currency = event['from_currency']
            from_amount = event['from_amount']
            to_currency = event['to_currency']
            to_amount = event['to_amount']
            
            # Ensure currencies exist in tracking
            if from_currency not in cash_positions:
                cash_positions[from_currency] = 0.0
            if to_currency not in cash_positions:
                cash_positions[to_currency] = 0.0
            
            # Update cash positions (subtract from source, add to target)
            cash_positions[from_currency] -= from_amount
            cash_positions[to_currency] += to_amount
        
        # Record cash positions after this event
        rebuild_log[-1]['cash_after'] = cash_positions.copy()
        rebuild_log[-1]['positions'] = {
            symbol: {
                'shares': position['shares'],
                'book_value': position['book_value']
            } for symbol, position in rebuilt_positions.items() if position['shares'] > 0
        }
    
    # Step 5: Get current portfolio and cash positions for comparison
    # -------------------------------------------------------------
    current_portfolio = load_portfolio()
    current_cash = get_cash_positions()
    
    # Step 6: Calculate changes needed to update the portfolio and cash positions
    # --------------------------------------------------------------------------
    positions_to_update = []  # existing positions to update
    positions_to_add = []     # new positions to add
    positions_to_remove = []  # current positions to remove
    cash_to_update = {}       # cash positions to update
    
    # Remove positions with zero shares
    for symbol in list(rebuilt_positions.keys()):
        if rebuilt_positions[symbol]['shares'] <= 0:
            del rebuilt_positions[symbol]
    
    # Compare rebuilt positions with current positions
    for symbol, position in rebuilt_positions.items():
        if position['shares'] <= 0:
            continue  # Skip positions with no shares
        
        # Look for this symbol in current portfolio
        existing_id = None
        for inv_id, details in current_portfolio.items():
            if details['symbol'].upper() == symbol:
                existing_id = inv_id
                break
        
        # Get current price for the symbol (reuse existing or look up)
        current_price = 0
        if existing_id:
            current_price = float(current_portfolio[existing_id].get('current_price', 0))
        else:
            # Try to get price from data provider
            try:
                from modules.data_provider import data_provider
                quote = data_provider.get_current_quote(symbol)
                if quote and 'price' in quote:
                    current_price = float(quote['price'])
            except:
                # Fallback to last transaction price
                if position['transactions']:
                    current_price = position['transactions'][-1]['price']
        
        # Calculate current value and gain/loss
        current_value = position['shares'] * current_price
        gain_loss = current_value - position['book_value']
        gain_loss_percent = ((current_value / position['book_value']) - 1) * 100 if position['book_value'] > 0 else 0
        
        # Lookup asset information if needed
        if not position.get('asset_type'):
            if existing_id:
                position['asset_type'] = current_portfolio[existing_id].get('asset_type', 'stock')
            else:
                # Try looking up in tracked assets
                tracked_assets = load_tracked_assets()
                if symbol in tracked_assets:
                    position['asset_type'] = tracked_assets[symbol].get('type', 'stock')
                    position['name'] = tracked_assets[symbol].get('name', symbol)
                else:
                    position['asset_type'] = 'stock'
        
        if not position.get('name'):
            if existing_id:
                position['name'] = current_portfolio[existing_id].get('name', symbol)
            else:
                position['name'] = symbol
        
        # If position exists, update it
        if existing_id:
            positions_to_update.append({
                'id': existing_id,
                'shares': position['shares'],
                'purchase_price': position['book_value'] / position['shares'] if position['shares'] > 0 else 0,
                'current_price': current_price,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent,
                'purchase_date': position['last_date'].strftime('%Y-%m-%d') if position['last_date'] else None,
                'asset_type': position['asset_type'],
                'currency': position['currency'],
                'name': position['name'],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # Add new position
            positions_to_add.append({
                'symbol': symbol,
                'shares': position['shares'],
                'purchase_price': position['book_value'] / position['shares'] if position['shares'] > 0 else 0,
                'current_price': current_price,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent,
                'purchase_date': position['last_date'].strftime('%Y-%m-%d') if position['last_date'] else None,
                'asset_type': position['asset_type'],
                'currency': position['currency'],
                'name': position['name'],
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Find positions to remove (in current portfolio but not in rebuilt)
    rebuilt_symbols = set(rebuilt_positions.keys())
    for inv_id, details in current_portfolio.items():
        if details['symbol'].upper() not in rebuilt_symbols:
            positions_to_remove.append(inv_id)
    
    # Prepare cash updates
    for currency, balance in cash_positions.items():
        current_balance = float(current_cash.get(currency, {}).get('balance', 0))
        
        # Add to update list if different (within a small tolerance for floating point)
        if abs(current_balance - balance) >= 0.01:
            cash_to_update[currency] = {
                'current': current_balance,
                'calculated': balance,
                'difference': balance - current_balance
            }
    
    # Step 7: Apply updates (if not dry run)
    # --------------------------------------
    changes_made = 0
    positions_added = 0
    positions_removed = 0
    positions_updated = 0
    cash_updated = 0
    
    # Function parameter - set to True to make actual changes
    apply_changes = True
    
    if apply_changes:
        # Remove positions
        for inv_id in positions_to_remove:
            remove_query = "DELETE FROM portfolio WHERE id = %s;"
            result = execute_query(remove_query, (inv_id,), commit=True)
            
            if result is not None:
                positions_removed += 1
                logger.info(f"Removed position {current_portfolio[inv_id]['symbol']} (ID: {inv_id})")
        
        # Update existing positions
        for position in positions_to_update:
            update_query = """
            UPDATE portfolio SET
                shares = %s,
                purchase_price = %s,
                current_price = %s,
                current_value = %s,
                gain_loss = %s,
                gain_loss_percent = %s,
                purchase_date = %s,
                asset_type = %s,
                currency = %s,
                name = %s,
                last_updated = %s
            WHERE id = %s;
            """
            
            params = (
                position['shares'],
                position['purchase_price'],
                position['current_price'],
                position['current_value'],
                position['gain_loss'],
                position['gain_loss_percent'],
                position['purchase_date'],
                position['asset_type'],
                position['currency'],
                position['name'],
                position['last_updated'],
                position['id']
            )
            
            result = execute_query(update_query, params, commit=True)
            
            if result is not None:
                positions_updated += 1
                changes_made += 1
                logger.info(f"Updated position for {current_portfolio[position['id']]['symbol']} (ID: {position['id']})")
        
        # Add new positions
        for position in positions_to_add:
            # Generate new ID
            new_id = str(uuid.uuid4())
            
            insert_query = """
            INSERT INTO portfolio (
                id, symbol, shares, purchase_price, purchase_date, asset_type,
                current_price, current_value, gain_loss, gain_loss_percent, 
                currency, name, added_date, last_updated
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
            """
            
            params = (
                new_id,
                position['symbol'],
                position['shares'],
                position['purchase_price'],
                position['purchase_date'],
                position['asset_type'],
                position['current_price'],
                position['current_value'],
                position['gain_loss'],
                position['gain_loss_percent'],
                position['currency'],
                position['name'],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                position['last_updated']
            )
            
            result = execute_query(insert_query, params, commit=True)
            
            if result is not None:
                positions_added += 1
                changes_made += 1
                logger.info(f"Added new position for {position['symbol']} (ID: {new_id})")
        
        # Update cash positions
        for currency, details in cash_to_update.items():
            calculated_balance = details['calculated']
            
            # Check if currency position exists
            check_query = "SELECT * FROM cash_positions WHERE currency = %s;"
            cash_result = execute_query(check_query, (currency,), fetchone=True)
            
            if cash_result:
                # Update existing cash position
                update_query = """
                UPDATE cash_positions 
                SET balance = %s, last_updated = %s 
                WHERE currency = %s;
                """
                
                params = (
                    calculated_balance,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    currency
                )
                
                result = execute_query(update_query, params, commit=True)
                
                if result is not None:
                    cash_updated += 1
                    changes_made += 1
                    logger.info(f"Updated cash position for {currency}: {details['current']} → {calculated_balance}")
            else:
                # Create new cash position
                insert_query = """
                INSERT INTO cash_positions (currency, balance, last_updated) 
                VALUES (%s, %s, %s);
                """
                
                params = (
                    currency,
                    calculated_balance,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                result = execute_query(insert_query, params, commit=True)
                
                if result is not None:
                    cash_updated += 1
                    changes_made += 1
                    logger.info(f"Created new cash position for {currency} with balance {calculated_balance}")
    
    # Step 8: Return comprehensive report
    # ----------------------------------
    return {
        'status': 'success',
        'changes_applied': apply_changes,
        'changes_made': changes_made,
        'positions_updated': positions_updated,
        'positions_added': positions_added,
        'positions_removed': positions_removed,
        'cash_updated': cash_updated,
        'rebuilt_positions': {symbol: {k: v for k, v in pos.items() if k != 'transactions'} 
                             for symbol, pos in rebuilt_positions.items() if pos['shares'] > 0},
        'cash_positions': cash_positions,
        'cash_discrepancies': cash_to_update,
        'event_log': rebuild_log,
        'rebuild_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


