# modules/portfolio_tracker.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import json

class PortfolioTracker:
    """
    Tracks and evaluates the performance of the investment portfolio.
    """
    
    def __init__(self):
        self.portfolio = []
        self.historical_value = pd.DataFrame()
        self.benchmarks = {
            'SPY': 'S&P 500',
            'AGG': 'US Bonds',
            'GLD': 'Gold'
        }
    
    def add_investment(self, symbol, name, asset_type, amount, date=None, price=None):
        """
        Add a new investment to the portfolio.
        
        Args:
            symbol (str): Ticker symbol
            name (str): Investment name
            asset_type (str): Asset type (stock, bond, etc.)
            amount (float): Investment amount
            date (datetime): Investment date
            price (float): Purchase price
            
        Returns:
            dict: Added investment details
        """
        if date is None:
            date = datetime.now()
            
        # Get current price if not provided
        if price is None:
            try:
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period="1d")
                if not price_data.empty:
                    price = price_data['Close'].iloc[-1]
                else:
                    price = 0
            except:
                price = 0
        
        # Calculate shares
        shares = amount / price if price > 0 else 0
        
        investment = {
            'symbol': symbol,
            'name': name,
            'asset_type': asset_type,
            'amount': amount,
            'date': date,
            'price': price,
            'shares': shares,
            'current_value': amount  # Initially set to purchase amount
        }
        
        self.portfolio.append(investment)
        
        # Update historical value
        self.update_historical_value()
        
        return investment
    
    def update_portfolio_value(self):
        """
        Update the current value of all investments in the portfolio.
        
        Returns:
            float: Total portfolio value
        """
        total_value = 0
        
        for investment in self.portfolio:
            try:
                ticker = yf.Ticker(investment['symbol'])
                price_data = ticker.history(period="1d")
                
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
                    current_value = investment['shares'] * current_price
                    investment['current_price'] = current_price
                    investment['current_value'] = current_value
                    investment['gain_loss'] = current_value - investment['amount']
                    investment['gain_loss_percent'] = (current_value / investment['amount'] - 1) * 100
                    
                    total_value += current_value
            except Exception as e:
                print(f"Error updating {investment['symbol']}: {e}")
        
        return total_value
    
    def update_historical_value(self, days=180):
        """
        Update the historical value of the portfolio.
        
        Args:
            days (int): Number of days to retrieve historical data for
            
        Returns:
            DataFrame: Historical portfolio value
        """
        if not self.portfolio:
            return pd.DataFrame()
            
        start_date = datetime.now() - timedelta(days=days)
        
        # Get historical prices for all symbols in portfolio
        all_data = {}
        for investment in self.portfolio:
            try:
                ticker = yf.Ticker(investment['symbol'])
                hist = ticker.history(start=start_date)
                if not hist.empty:
                    all_data[investment['symbol']] = hist['Close']
            except Exception as e:
                print(f"Error getting historical data for {investment['symbol']}: {e}")
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all price data
        price_df = pd.DataFrame(all_data)
        
        # Calculate portfolio value for each day
        portfolio_values = pd.DataFrame(index=price_df.index)
        portfolio_values['Total'] = 0
        
        for investment in self.portfolio:
            if investment['symbol'] in price_df.columns:
                # Calculate value based on shares owned
                portfolio_values[investment['symbol']] = price_df[investment['symbol']] * investment['shares']
                portfolio_values['Total'] += portfolio_values[investment['symbol']]
        
        # Get benchmark data
        for symbol in self.benchmarks:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date)
                if not hist.empty:
                    # Normalize to match starting portfolio value
                    benchmark_values = hist['Close'] / hist['Close'].iloc[0] * portfolio_values['Total'].iloc[0]
                    portfolio_values[f"Benchmark_{symbol}"] = benchmark_values
            except Exception as e:
                print(f"Error getting benchmark data for {symbol}: {e}")
        
        self.historical_value = portfolio_values
        return portfolio_values
    
    def get_performance(self, period='all'):
        """
        Get portfolio performance statistics.
        
        Args:
            period (str): Time period ('1m', '3m', '6m', '1y', 'all')
            
        Returns:
            dict: Performance statistics
        """
        # Make sure portfolio is up to date
        self.update_portfolio_value()
        
        if not self.portfolio:
            return {
                'total_value': 0,
                'total_invested': 0,
                'total_gain_loss': 0,
                'total_gain_loss_percent': 0,
                'historical_value': pd.DataFrame()
            }
        
        # Calculate total invested and current value
        total_invested = sum(inv['amount'] for inv in self.portfolio)
        total_value = sum(inv['current_value'] for inv in self.portfolio)
        
        # Calculate gain/loss
        total_gain_loss = total_value - total_invested
        total_gain_loss_percent = (total_value / total_invested - 1) * 100 if total_invested > 0 else 0
        
        # Get or update historical value
        if self.historical_value.empty:
            self.update_historical_value()
            
        # Filter based on period
        historical_df = self.historical_value.copy()
        if period == '1m':
            historical_df = historical_df.iloc[-30:]
        elif period == '3m':
            historical_df = historical_df.iloc[-90:]
        elif period == '6m':
            historical_df = historical_df.iloc[-180:]
        elif period == '1y':
            historical_df = historical_df.iloc[-365:]
        
        # Calculate performance metrics
        if not historical_df.empty and len(historical_df) > 1:
            # Calculate returns
            returns = historical_df['Total'].pct_change().dropna()
            
            # Calculate metrics
            performance = {
                'total_value': total_value,
                'total_invested': total_invested,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': total_gain_loss_percent,
                'historical_value': historical_df,
                'period_return': (historical_df['Total'].iloc[-1] / historical_df['Total'].iloc[0] - 1) * 100,
                'annualized_return': ((historical_df['Total'].iloc[-1] / historical_df['Total'].iloc[0]) ** (365 / len(historical_df)) - 1) * 100,
                'volatility': returns.std() * (252 ** 0.5) * 100,  # Annualized volatility
                'sharpe_ratio': (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0,
                'max_drawdown': ((historical_df['Total'] / historical_df['Total'].cummax()) - 1).min() * 100
            }
        else:
            performance = {
                'total_value': total_value,
                'total_invested': total_invested,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': total_gain_loss_percent,
                'historical_value': historical_df,
                'period_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        return performance
    
    def get_asset_allocation(self):
        """
        Get current asset allocation breakdown.
        
        Returns:
            dict: Asset allocation
        """
        if not self.portfolio:
            return {
                'by_asset_type': {},
                'by_symbol': {}
            }
            
        # Make sure portfolio is up to date
        self.update_portfolio_value()
        
        # Calculate total value
        total_value = sum(inv['current_value'] for inv in self.portfolio)
        
        # Calculate allocation by asset type
        asset_type_allocation = {}
        for investment in self.portfolio:
            asset_type = investment['asset_type']
            value = investment['current_value']
            
            if asset_type not in asset_type_allocation:
                asset_type_allocation[asset_type] = 0
                
            asset_type_allocation[asset_type] += value
            
        # Calculate allocation by symbol
        symbol_allocation = {}
        for investment in self.portfolio:
            symbol = investment['symbol']
            value = investment['current_value']
            
            symbol_allocation[symbol] = {
                'value': value,
                'percent': (value / total_value * 100) if total_value > 0 else 0,
                'name': investment['name'],
                'asset_type': investment['asset_type']
            }
            
        # Convert asset type values to percentages
        asset_type_percent = {}
        for asset_type, value in asset_type_allocation.items():
            asset_type_percent[asset_type] = (value / total_value * 100) if total_value > 0 else 0
            
        return {
            'by_asset_type': asset_type_percent,
            'by_symbol': symbol_allocation,
            'total_value': total_value
        }
    
    def create_performance_chart(self, period='all'):
        """
        Create a Plotly performance chart.
        
        Args:
            period (str): Time period ('1m', '3m', '6m', '1y', 'all')
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Get performance data
        performance = self.get_performance(period)
        historical_df = performance['historical_value']
        
        if historical_df.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="Portfolio Performance (No Data Available)",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                template="plotly_white"
            )
            return fig
        
        # Reset index to make date a column
        df_plot = historical_df.reset_index()
        
        # Create base figure with portfolio total
        fig = px.line(
            df_plot, 
            x='Date', 
            y='Total',
            title="Portfolio Performance",
            labels={"Total": "Value ($)", "Date": ""},
            template="plotly_white"
        )
        
        # Add benchmark lines
        for symbol in self.benchmarks:
            benchmark_col = f"Benchmark_{symbol}"
            if benchmark_col in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot['Date'],
                        y=df_plot[benchmark_col],
                        mode='lines',
                        name=f"{self.benchmarks[symbol]} (Benchmark)",
                        line=dict(dash='dash')
                    )
                )
        
        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_allocation_chart(self):
        """
        Create a Plotly pie chart showing asset allocation.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Get allocation data
        allocation = self.get_asset_allocation()
        
        if not allocation['by_asset_type']:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="Asset Allocation (No Data Available)",
                template="plotly_white"
            )
            return fig
        
        # Create dataframe for plotting
        df_allocation = pd.DataFrame({
            'Asset Type': list(allocation['by_asset_type'].keys()),
            'Percentage': list(allocation['by_asset_type'].values())
        })
        
        # Create pie chart
        fig = px.pie(
            df_allocation,
            values='Percentage',
            names='Asset Type',
            title="Asset Allocation",
            template="plotly_white",
            hole=0.4
        )
        
        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_investment_performance_chart(self):
        """
        Create a Plotly bar chart showing individual investment performance.
        
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if not self.portfolio:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="Investment Performance (No Data Available)",
                template="plotly_white"
            )
            return fig
        
        # Create dataframe for plotting
        df_investments = pd.DataFrame({
            'Symbol': [inv['symbol'] for inv in self.portfolio],
            'Name': [inv['name'] for inv in self.portfolio],
            'Gain/Loss %': [inv.get('gain_loss_percent', 0) for inv in self.portfolio],
            'Current Value': [inv.get('current_value', 0) for inv in self.portfolio]
        })
        
        # Sort by gain/loss percentage
        df_investments = df_investments.sort_values('Gain/Loss %')
        
        # Create color map based on gain/loss
        colors = ['red' if x < 0 else 'green' for x in df_investments['Gain/Loss %']]
        
        # Create bar chart
        fig = px.bar(
            df_investments,
            x='Symbol',
            y='Gain/Loss %',
            color='Gain/Loss %',
            color_continuous_scale=['red', 'lightgrey', 'green'],
            range_color=[-max(abs(df_investments['Gain/Loss %'])), max(abs(df_investments['Gain/Loss %']))],
            hover_data=['Name', 'Current Value'],
            title="Individual Investment Performance",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Gain/Loss (%)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def export_performance_report(self, format='csv'):
        """
        Export portfolio performance report.
        
        Args:
            format (str): Export format ('csv', 'json', 'plotly')
            
        Returns:
            str or dict: Path to exported file or dict with figures
        """
        # Update portfolio
        performance = self.get_performance()
        allocation = self.get_asset_allocation()
        
        # Create report dataframe
        report = pd.DataFrame()
        
        # Portfolio summary
        summary = pd.DataFrame({
            'Metric': ['Total Value', 'Total Invested', 'Total Gain/Loss', 'Total Gain/Loss %', 
                      'Period Return %', 'Annualized Return %', 'Volatility %', 'Sharpe Ratio', 'Max Drawdown %'],
            'Value': [
                f"${performance['total_value']:.2f}",
                f"${performance['total_invested']:.2f}",
                f"${performance['total_gain_loss']:.2f}",
                f"{performance['total_gain_loss_percent']:.2f}%",
                f"{performance['period_return']:.2f}%",
                f"{performance['annualized_return']:.2f}%",
                f"{performance['volatility']:.2f}%",
                f"{performance['sharpe_ratio']:.2f}",
                f"{performance['max_drawdown']:.2f}%"
            ]
        })
        
        # Investment details
        investments = pd.DataFrame()
        for inv in self.portfolio:
            investments = pd.concat([investments, pd.DataFrame({
                'Symbol': [inv['symbol']],
                'Name': [inv['name']],
                'Asset Type': [inv['asset_type']],
                'Shares': [inv['shares']],
                'Purchase Price': [f"${inv['price']:.2f}"],
                'Current Price': [f"${inv.get('current_price', 0):.2f}"],
                'Amount Invested': [f"${inv['amount']:.2f}"],
                'Current Value': [f"${inv.get('current_value', 0):.2f}"],
                'Gain/Loss': [f"${inv.get('gain_loss', 0):.2f}"],
                'Gain/Loss %': [f"{inv.get('gain_loss_percent', 0):.2f}%"]
            })], ignore_index=True)
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            # Save summary
            summary_path = f"reports/portfolio_summary_{timestamp}.csv"
            summary.to_csv(summary_path, index=False)
            
            # Save investments
            investments_path = f"reports/portfolio_investments_{timestamp}.csv"
            investments.to_csv(investments_path, index=False)
            
            # Return paths
            return {
                'summary': summary_path,
                'investments': investments_path
            }
            
        elif format == 'json':
            report_data = {
                'summary': performance,
                'allocation': allocation,
                'investments': self.portfolio
            }
            
            report_path = f"reports/portfolio_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4, default=str)
                
            return report_path
            
        elif format == 'plotly':
            # Create plotly figures for report
            figures = {
                'performance': self.create_performance_chart(),
                'allocation': self.create_allocation_chart(),
                'investments': self.create_investment_performance_chart()
            }
            
            return figures
        
        return None