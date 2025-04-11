# Help: Market Analysis Features

This section covers the various market analysis tools available within the AI Investment Recommendation System (AIRS), including the Market Overview graph, News Analysis, and detailed Portfolio Analysis breakdowns.

## Market Overview & Technical Indicators

### Market Overview Graph

* **Purpose:** This graph provides a visual comparison of the performance of assets you have added to your **Tracked Assets** list (using the Asset Tracker component). It helps you quickly see how assets of interest are performing relative to each other and to a market benchmark (typically the S&P/TSX Composite Index [cite: investment-guru/main.py]).
* **Display:** Asset performance is normalized (starting at a value of 100) over the selected time period, making it easy to compare growth trajectories. The benchmark index is usually displayed as a dashed line [cite: investment-guru/main.py].
* **Controls:** You can change the time period displayed using the radio buttons below the graph (e.g., 1 Week, 1 Month, 1 Year) [cite: investment-guru/main.py]. The graph updates periodically automatically.

### Technical Indicators

While not directly plotted on the main Market Overview graph in the current UI, the system utilizes various technical indicators for its underlying analysis and AI recommendations [cite: investment-guru/modules/market_analyzer.py, investment-guru/modules/trend_analysis.py]. Understanding these indicators can provide context for the system's insights:

* **Simple Moving Average (SMA):** The average closing price over a specific period (e.g., 20-day, 50-day, 200-day). Used to identify trends; shorter SMAs crossing above longer SMAs (like 50-day above 200-day, a "Golden Cross") can signal an uptrend, while the reverse ("Death Cross") can signal a downtrend [cite: investment-guru/modules/market_analyzer.py, investment-guru/modules/trend_analysis.py].
* **Relative Strength Index (RSI):** A momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100.
    * Typically, RSI above 70 indicates an asset may be overbought (potentially due for a pullback) [cite: investment-guru/modules/trend_analysis.py].
    * RSI below 30 suggests an asset may be oversold (potentially due for a rebound) [cite: investment-guru/modules/trend_analysis.py].
* **Moving Average Convergence Divergence (MACD):** A trend-following momentum indicator showing the relationship between two exponential moving averages (EMAs) of price.
    * `MACD Line:` The difference between two EMAs (typically 12-period and 26-period).
    * `Signal Line:` An EMA (typically 9-period) of the MACD Line.
    * `MACD Histogram:` The difference between the MACD Line and the Signal Line.
    * *Signals:* Crossovers between the MACD line and the Signal line are used as buy/sell signals. Positive histogram values suggest bullish momentum; negative values suggest bearish momentum [cite: investment-guru/modules/market_analyzer.py, investment-guru/modules/trend_analysis.py].
* **Bollinger Bands:** Bands plotted two standard deviations above and below a simple moving average (typically 20-day).
    * Used to measure volatility. Bands widen during high volatility and narrow during low volatility.
    * Prices are considered high when near the upper band and low when near the lower band. Prices moving outside the bands are significant events [cite: investment-guru/modules/market_analyzer.py, investment-guru/modules/trend_analysis.py].
* **Average True Range (ATR):** Measures market volatility by decomposing the entire range of an asset price for that period. Higher ATR indicates higher volatility [cite: investment-guru/modules/market_analyzer.py, investment-guru/modules/trend_analysis.py].

## News & Events Analysis

* **Purpose:** This section automatically fetches and displays recent financial news articles relevant to the assets in your portfolio and your tracked assets list [cite: investment-guru/main.py].
* **Content:** Each news card shows:
    * Relevant Ticker Symbols (if identified).
    * Sentiment Badge (Positive, Negative, or Neutral) based on analysis of the headline and description [cite: investment-guru/main.py, investment-guru/modules/news_analyzer.py].
    * Source and Publication Date/Time.
    * Headline and a snippet of the description.
    * A link to read the full article [cite: investment-guru/main.py].
* **Sentiment Summary:** Above the news cards, a summary bar visualizes the overall sentiment distribution (Positive, Negative, Neutral) across the displayed articles [cite: investment-guru/main.py].
* **Updates:** The news feed updates automatically at regular intervals [cite: investment-guru/main.py].

## Portfolio Analysis Component

This component provides deeper insights into the composition and characteristics of your investment portfolio. It contains three main tabs [cite: investment-guru/components/portfolio_analysis.py, investment-guru/main.py]:

### Asset Allocation Tab

* **Purpose:** Shows how your portfolio is divided among different asset classes (e.g., Stocks, ETFs, Bonds, Cash).
* **Visualization:** Displays an interactive Pie Chart illustrating the percentage allocated to each asset type [cite: investment-guru/components/portfolio_analysis.py].
* **Details:** Below the chart, a table provides the specific dollar value and percentage for each asset class in your portfolio [cite: investment-guru/components/portfolio_analysis.py].

### Sector Breakdown Tab

* **Purpose:** Analyzes the distribution of your equity holdings across different market sectors (e.g., Technology, Healthcare, Finance, Energy). This helps identify concentration risk.
* **Visualization:** Displays a Bar Chart showing the percentage allocated to each sector [cite: investment-guru/components/portfolio_analysis.py].
* **Details:** A table below the chart lists the dollar value and percentage allocation for each identified sector [cite: investment-guru/components/portfolio_analysis.py]. Sector information is typically derived from market data for each holding [cite: investment-guru/components/portfolio_analysis.py].

### Correlation Analysis Tab

* **Purpose:** Measures how closely the price movements of different assets in your portfolio are related. Low correlation is generally desirable for diversification.
* **Visualization:** Displays a Heatmap where colors indicate the correlation coefficient between pairs of assets (ranging from -1 for perfect negative correlation to +1 for perfect positive correlation) [cite: investment-guru/components/portfolio_analysis.py].
* **Analysis:** Provides a summary identifying the most and least correlated pairs in your portfolio and offers guidance on diversification based on the average correlation [cite: investment-guru/components/portfolio_analysis.py]. Historical price data is used to calculate these correlations [cite: investment-guru/components/portfolio_analysis.py].