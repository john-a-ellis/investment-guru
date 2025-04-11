# Introduction to the AI Investment Recommendation System (AIRS)

Welcome to the AI Investment Recommendation System (AIRS), also known as Investment Guru. This application is designed to provide users with advanced investment strategies and insights, leveraging artificial intelligence for market analysis and recommendations.

## Purpose

The core goal of AIRS is to assist users in making informed investment decisions by integrating various financial data sources, applying machine learning models, and offering personalized portfolio management tools.

## Key Features

AIRS offers a comprehensive suite of tools for investors:

* **Dashboard Overview:** Presents a high-level view of market conditions, portfolio performance, and key insights.
* **Portfolio Management:** Track your investments, including stocks, ETFs, mutual funds, and more. Add new holdings, record buy/sell transactions, and view performance metrics.
* **Asset Tracking:** Monitor specific assets you are interested in, separate from your main portfolio.
* **Market Analysis:**
    * **Technical Indicators:** Visualize market trends using indicators like SMA, RSI, MACD, and Bollinger Bands.
    * **News Analysis:** Stay updated with relevant financial news, complete with sentiment analysis to gauge market mood.
    * **Market Overview:** Track the performance of your selected assets against benchmarks like the S&P/TSX Composite.
* **AI-Powered Insights:**
    * **Investment Recommendations:** Receive suggestions for potential investments based on your user profile (risk tolerance, investment horizon) and market analysis.
    * **Price Prediction:** View AI-generated price predictions for selected assets over different time horizons.
    * **Model Training & Management:** Train and manage the machine learning models used for predictions.
* **Portfolio Analysis:**
    * **Allocation:** Visualize your portfolio's diversification across different asset types and sectors.
    * **Correlation:** Analyze how your different investments move in relation to each other.
    * **Risk Assessment:** Evaluate portfolio risk using metrics like Sharpe Ratio, Sortino Ratio, Max Drawdown, Beta, and Value at Risk (VaR).
* **Rebalancing:** Compare your current allocation to your target, identify drifts, and receive recommendations on how to rebalance your portfolio.
* **User Profile:** Customize your risk tolerance and investment horizon to tailor recommendations.
* **Mutual Fund Management:** Manually add and track price data for mutual funds not readily available through standard APIs.

## Technology

The application is built using Python, leveraging the Dash framework and Plotly for interactive visualizations. It integrates various data sources through a dedicated data provider module and employs machine learning models (potentially including Prophet, LSTM, ARIMA) for prediction and analysis. Portfolio data, user profiles, and transaction history are stored and managed in a PostgreSQL database.