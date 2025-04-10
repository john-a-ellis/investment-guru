# modules/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go

# Import the new DataProvider instance
from modules.data_provider import data_provider
# Remove direct FMP API import
# from modules.fmp_api import fmp_api

# Import necessary components from your project
from modules.model_integration import ModelIntegration # To generate signals
from modules.portfolio_risk_metrics import calculate_risk_metrics # For performance analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, symbols, start_date, end_date, initial_capital=100000, strategy_params=None):
        """
        Initializes the Backtester.

        Args:
            symbols (list): List of asset symbols to include in the backtest universe.
            start_date (str or datetime): Backtest start date.
            end_date (str or datetime): Backtest end date.
            initial_capital (float): Starting cash for the simulation.
            strategy_params (dict, optional): Parameters for the investment strategy. Defaults to None.
        """
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.strategy_params = strategy_params if strategy_params is not None else {}

        self.cash = initial_capital
        self.holdings = {symbol: 0 for symbol in self.symbols} # Shares held for each symbol
        self.portfolio_value_history = pd.Series(dtype=float)
        self.trades = [] # List to store executed trades

        # Initialize ModelIntegration for generating signals (can be customized)
        # Note: For proper backtesting, model training should ideally happen *before* the backtest
        # or be updated incrementally within the loop using data only available up to that point.
        # This example uses a simplified approach where models might be pre-trained.
        self.model_integration = ModelIntegration() # [cite: 14]

        # --- Data Loading ---
        self.historical_data = self._load_data()
        if self.historical_data.empty:
            raise ValueError("Failed to load historical data for backtesting.")

        # --- Add benchmark data (e.g., TSX) ---
        self._add_benchmark_data('^GSPTSE') # Using TSX Composite as example

    def _load_data(self):
        """Loads historical price data for all symbols using DataProvider."""
        logger.info(f"Loading historical data for symbols: {self.symbols}")
        all_data = {}
        min_date = self.start_date - timedelta(days=365) # Load extra data for initial calculations

        for symbol in self.symbols:
            try:
                # --- Use DataProvider ---
                hist = data_provider.get_historical_price(symbol, start_date=min_date, end_date=self.end_date)
                # DataProvider already handles standardization (lowercase columns, timezone-naive index)

                if not hist.empty and 'close' in hist.columns:
                     # --- Use standardized lowercase columns ---
                    all_data[symbol] = hist[['open', 'high', 'low', 'close', 'volume']]
                else:
                    logger.warning(f"No data loaded for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine into a multi-index DataFrame
        panel_data = pd.concat(all_data, axis=1)
        panel_data.columns.names = ['Symbol', 'Field'] # Keep original field names for multi-index
        panel_data = panel_data.ffill().bfill() # Fill missing values
        panel_data = panel_data.loc[self.start_date:self.end_date] # Filter to backtest period
        logger.info(f"Data loaded from {panel_data.index.min()} to {panel_data.index.max()}")
        return panel_data

    def _add_benchmark_data(self, benchmark_symbol):
        """Adds benchmark data to the historical data panel using DataProvider."""
        logger.info(f"Loading benchmark data: {benchmark_symbol}")
        try:
            min_date = self.start_date - timedelta(days=1) # Ensure we have the day before start
            # --- Use DataProvider ---
            bench_hist = data_provider.get_historical_price(benchmark_symbol, start_date=min_date, end_date=self.end_date)
            # DataProvider handles standardization

            # --- Use standardized lowercase 'close' column ---
            if not bench_hist.empty and 'close' in bench_hist.columns:
                # Align benchmark index with main data and forward fill
                # --- Use standardized lowercase 'close' column ---
                self.historical_data[(benchmark_symbol, 'close')] = bench_hist['close'].reindex(self.historical_data.index, method='ffill')
                logger.info(f"Benchmark data '{benchmark_symbol}' added.")
            else:
                 logger.warning(f"No benchmark data loaded for {benchmark_symbol}")
        except Exception as e:
            logger.error(f"Error loading benchmark data for {benchmark_symbol}: {e}")


    def _get_available_data(self, current_date):
        """Returns historical data available up to the current_date."""
        # Important: Only use data up to the *previous* day to avoid lookahead bias
        available_date = current_date - timedelta(days=1)
        return self.historical_data.loc[self.historical_data.index <= available_date]

    def _generate_signals(self, current_date, current_data_for_day):
        """
        Generates trading signals for the current day using the adapted strategy logic.

        Args:
            current_date (datetime): The current date in the backtest loop.
            current_data_for_day (pd.Series): Data for the current day (not used for signal generation
                                              to prevent lookahead, but potentially useful context).

        Returns:
            dict: Dictionary of signals {symbol: 'BUY'/'SELL'/'HOLD'}.
        """
        signals = {} # {symbol: 'BUY'/'SELL'/'HOLD'}
        available_data_panel = self._get_available_data(current_date) # Data up to previous day

        if available_data_panel.empty or len(available_data_panel) < 60: # Need enough history
            #logger.debug(f"Not enough historical data available on {current_date} to generate signals.")
            return signals

        # --- Use ModelIntegration for signal generation ---
        try:
            # Iterate through each symbol we might trade
            for symbol in self.symbols:
                # Extract the historical data slice for the specific symbol
                if symbol in available_data_panel.columns.get_level_values('Symbol'):
                    # --- Use standardized lowercase columns ---
                    symbol_historical_slice = available_data_panel[symbol].dropna()
                    # Rename columns temporarily for the model if it expects uppercase
                    # This depends on how get_backtesting_recommendation_signal is implemented
                    # If it expects 'Close', 'Open', etc., rename here:
                    # temp_slice = symbol_historical_slice.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'})

                    if symbol_historical_slice.empty or len(symbol_historical_slice) < 50: # Ensure enough data for analysis
                        #logger.debug(f"Skipping signal generation for {symbol} on {current_date} due to insufficient slice length: {len(symbol_historical_slice)}")
                        continue

                    # --- !!! CALL ADAPTED STRATEGY LOGIC !!! ---
                    # Pass the slice with the column names expected by the function
                    signal = self.model_integration.get_backtesting_recommendation_signal(
                        symbol=symbol,
                        historical_data_slice=symbol_historical_slice # or temp_slice if renaming needed
                    )
                    # --- END ADAPTED CALL ---

                    if signal in ['BUY', 'SELL', 'HOLD']:
                        signals[symbol] = signal
                    else:
                        signals[symbol] = 'HOLD' # Default to HOLD if no clear signal
                else:
                     #logger.debug(f"No data slice available for {symbol} on {current_date}")
                     pass


        except Exception as e:
            logger.error(f"Error generating signals via ModelIntegration on {current_date}: {e}")
            import traceback
            traceback.print_exc()
            signals = {symbol: 'HOLD' for symbol in self.symbols} # Fail safe: hold all

        #logger.debug(f"Generated signals for {current_date}: {signals}")
        return signals


    def _execute_trade(self, symbol, action, date, price, quantity=None, capital_fraction=0.1):
        """Simulates executing a trade."""
        if action == 'BUY':
            investment_amount = self.cash * capital_fraction
            if price > 0:
                shares_to_buy = int(investment_amount / price) # Buy whole shares
                cost = shares_to_buy * price
                if shares_to_buy > 0 and self.cash >= cost:
                    self.holdings[symbol] += shares_to_buy
                    self.cash -= cost
                    self.trades.append({'Date': date, 'Symbol': symbol, 'Action': 'BUY', 'Shares': shares_to_buy, 'Price': price, 'Cost': cost})
                    logger.debug(f"{date}: Bought {shares_to_buy} {symbol} @ {price:.2f}")
                #else: logger.debug(f"{date}: Insufficient cash to buy {symbol}")

        elif action == 'SELL':
            shares_to_sell = self.holdings[symbol] # Sell all shares for simplicity
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                self.holdings[symbol] -= shares_to_sell
                self.cash += proceeds
                self.trades.append({'Date': date, 'Symbol': symbol, 'Action': 'SELL', 'Shares': shares_to_sell, 'Price': price, 'Proceeds': proceeds})
                logger.debug(f"{date}: Sold {shares_to_sell} {symbol} @ {price:.2f}")
        #else: HOLD - do nothing

    def _update_portfolio_value(self, date, current_prices):
        """Calculates the total portfolio value for the current day."""
        holdings_value = 0
        for symbol in self.symbols:
            # --- Use standardized lowercase 'close' column ---
            if (symbol, 'close') in current_prices.index:
                 price = current_prices[(symbol, 'close')]
                 if pd.notna(price) and self.holdings[symbol] > 0 : # Only count if price is valid
                     holdings_value += self.holdings[symbol] * price
            # else: logger.warning(f"No price for {symbol} on {date}") # Optional warning


        total_value = self.cash + holdings_value
        self.portfolio_value_history[date] = total_value
        #logger.debug(f"{date}: Cash: {self.cash:.2f}, Holdings: {holdings_value:.2f}, Total: {total_value:.2f}")


    def run(self):
        """Runs the backtest simulation."""
        logger.info("Starting backtest...")
        self.portfolio_value_history = pd.Series(index=self.historical_data.index, dtype=float)

        for date, day_data in self.historical_data.iterrows():
            # 1. Update Portfolio Value with *yesterday's* close or *today's* open
            # Using yesterday's close prices for valuation before today's trades
            if date > self.start_date:
                 yesterday = date - timedelta(days=1)
                 # Find the most recent valid prices up to *or including* yesterday
                 prices_for_valuation = self.historical_data.loc[self.historical_data.index <= yesterday].iloc[-1]
                 self._update_portfolio_value(yesterday, prices_for_valuation)
            elif date == self.start_date:
                 # Set initial value
                 self.portfolio_value_history[date] = self.initial_capital


            # 2. Generate Signals based on data *up to yesterday*
            signals = self._generate_signals(date, day_data) # Pass today's data for context if needed by strategy

            # 3. Execute Trades based on signals and *today's* prices (e.g., Open price)
            for symbol, action in signals.items():
                # --- Use standardized lowercase 'open' column ---
                if (symbol, 'open') in day_data.index:
                     trade_price = day_data[(symbol, 'open')] # Trade at Open price
                     if pd.notna(trade_price):
                         self._execute_trade(symbol, action, date, trade_price)
                #else: logger.warning(f"No Open price for {symbol} on {date} to execute trade.")


        # Final portfolio valuation at the end date using end date's close prices
        final_prices = self.historical_data.iloc[-1]
        self._update_portfolio_value(self.end_date, final_prices)

        logger.info("Backtest finished.")
        self.portfolio_value_history = self.portfolio_value_history.dropna() # Clean up any NaNs


    def get_results(self):
        """Calculates and returns performance metrics."""
        if self.portfolio_value_history.empty:
            logger.warning("No portfolio history to calculate results.")
            return {}

        # --- Performance Calculation ---
        results = {}
        portfolio_returns = self.portfolio_value_history

        # Total Return
        total_return = (portfolio_returns.iloc[-1] / portfolio_returns.iloc[0]) - 1
        results['Total Return (%)'] = total_return * 100

        # Annualized Return
        delta_years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
        if delta_years > 0:
             annualized_return = ((portfolio_returns.iloc[-1] / portfolio_returns.iloc[0]) ** (1 / delta_years)) - 1
        else:
             annualized_return = 0
        results['Annualized Return (%)'] = annualized_return * 100

        # Use portfolio_risk_metrics module [cite: 19]
        # Need to adapt calculate_risk_metrics to take a Series/DataFrame directly
        # Mocking the structure it expects for now
        mock_portfolio_for_metrics = {'Total': self.portfolio_value_history}
        # --- Use standardized lowercase 'close' column ---
        if ('^GSPTSE', 'close') in self.historical_data.columns:
             mock_portfolio_for_metrics['TSX'] = self.historical_data[('^GSPTSE', 'close')]

        # We need to create a temporary DataFrame suitable for calculate_risk_metrics
        metrics_df = pd.DataFrame(mock_portfolio_for_metrics)
        # NOTE: calculate_risk_metrics expects a portfolio dict, not the metrics_df directly.
        # It internally calls get_portfolio_historical_data. This needs refactoring in portfolio_risk_metrics.py
        # For now, we calculate manually here.
        # risk_metrics = calculate_risk_metrics(portfolio={}, period='all', risk_free_rate=0.02) # Pass empty portfolio, use period='all'

        # Manually calculate metrics:
        daily_returns = portfolio_returns.pct_change().dropna()
        if not daily_returns.empty:
             results['Annualized Volatility (%)'] = daily_returns.std() * np.sqrt(252) * 100
             # Sharpe Ratio (assuming 2% risk-free rate)
             risk_free_rate_daily = (1 + 0.02)**(1/252) - 1
             sharpe_ratio = (daily_returns.mean() - risk_free_rate_daily) / daily_returns.std() if daily_returns.std() > 0 else 0
             results['Sharpe Ratio'] = sharpe_ratio * np.sqrt(252)
        else:
             results['Annualized Volatility (%)'] = 0
             results['Sharpe Ratio'] = 0


        # Max Drawdown
        rolling_max = portfolio_returns.cummax()
        drawdown = (portfolio_returns / rolling_max) - 1
        results['Max Drawdown (%)'] = drawdown.min() * 100

        # Trades Analysis
        trades_df = pd.DataFrame(self.trades)
        results['Total Trades'] = len(trades_df)
        if not trades_df.empty:
             results['Winning Trades'] = len(trades_df[ (trades_df['Action']=='SELL') | ((trades_df['Action']=='BUY') ) ]) # Simplistic win calc
             # results['Win Rate (%)'] = (results['Winning Trades'] / results['Total Trades']) * 100 if results['Total Trades'] > 0 else 0
        else:
            results['Winning Trades'] = 0
            # results['Win Rate (%)'] = 0 # Avoid adding if no trades

        results['Final Portfolio Value'] = self.portfolio_value_history.iloc[-1]

        return results

    def plot_results(self):
        """Generates a Plotly chart of portfolio performance."""
        if self.portfolio_value_history.empty:
            print("No portfolio history to plot.")
            return None

        fig = go.Figure()

        # Portfolio Value
        fig.add_trace(go.Scatter(
            x=self.portfolio_value_history.index,
            y=self.portfolio_value_history,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))

        # Add Benchmark if available
        benchmark_symbol = '^GSPTSE' # Example
        # --- Use standardized lowercase 'close' column ---
        if (benchmark_symbol, 'close') in self.historical_data.columns:
             benchmark_data = self.historical_data[(benchmark_symbol, 'close')]
             # Normalize benchmark to start at initial capital
             normalized_benchmark = (benchmark_data / benchmark_data.iloc[0]) * self.initial_capital
             fig.add_trace(go.Scatter(
                 x=normalized_benchmark.index,
                 y=normalized_benchmark,
                 mode='lines',
                 name=f'{benchmark_symbol} (Normalized)',
                 line=dict(color='grey', width=1, dash='dash')
             ))


        # Plot Buy/Sell Markers
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
             buy_trades = trades_df[trades_df['Action'] == 'BUY']
             sell_trades = trades_df[trades_df['Action'] == 'SELL']

             # Map trade dates to portfolio value history index for plotting y-value
             buy_values = self.portfolio_value_history.reindex(buy_trades['Date'], method='ffill')
             sell_values = self.portfolio_value_history.reindex(sell_trades['Date'], method='ffill')


             fig.add_trace(go.Scatter(
                 x=buy_trades['Date'], y=buy_values, mode='markers', name='Buys',
                 marker=dict(color='green', size=8, symbol='triangle-up')
             ))
             fig.add_trace(go.Scatter(
                 x=sell_trades['Date'], y=sell_values, mode='markers', name='Sells',
                 marker=dict(color='red', size=8, symbol='triangle-down')
             ))

        fig.update_layout(
            title='Backtest Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
