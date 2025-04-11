# Help: Portfolio Analysis & Risk Assessment

This section describes the tools available for analyzing your portfolio's composition, diversification, and risk profile.

## Allocation Analysis

Understanding how your investments are distributed is crucial for managing risk and aligning with your goals. This component provides two perspectives on your allocation:

1.  **Asset Allocation Tab:** [cite: investment-guru/components/portfolio_analysis.py]
    * **Purpose:** Shows the breakdown of your portfolio by broad asset classes (e.g., Stock, ETF, Bond, Cash, Mutual Fund, Crypto).
    * **Visualization:** An interactive Pie Chart displays the percentage of your total portfolio value allocated to each asset class.
    * **Details:** Below the chart, a table lists the exact dollar value (in CAD) and percentage for each asset type [cite: investment-guru/components/portfolio_analysis.py, investment-guru/modules/portfolio_utils.py].

2.  **Sector Breakdown Tab:** [cite: investment-guru/components/portfolio_analysis.py]
    * **Purpose:** Analyzes the distribution of your stock and ETF holdings across different market sectors (e.g., Technology, Finance, Healthcare, Energy). This helps identify potential over-concentration in a specific area of the market.
    * **Visualization:** A Bar Chart shows the percentage of your portfolio invested in each identified sector.
    * **Details:** A table provides the corresponding dollar value (in CAD) and percentage for each sector [cite: investment-guru/components/portfolio_analysis.py]. Sector information is typically derived from market data providers [cite: investment-guru/components/portfolio_analysis.py].

## Correlation Analysis

* **Purpose:** This analysis helps you understand how diversified your portfolio truly is by measuring how closely the price movements of your different assets are related [cite: investment-guru/components/portfolio_analysis.py]. Lower correlation between assets generally leads to better diversification and potentially lower overall portfolio volatility.
* **Location:** Found in the "Correlation Analysis" tab within the "Portfolio Analysis" component [cite: investment-guru/components/portfolio_analysis.py].
* **Visualization:** A Heatmap displays the correlation coefficients between pairs of assets in your portfolio.
    * Colors range typically from Red (strong negative correlation, values near -1) through neutral colors (near 0) to Blue or Green (strong positive correlation, values near +1) [cite: investment-guru/components/portfolio_analysis.py].
    * A value of +1 means assets move perfectly together; -1 means they move perfectly opposite; 0 means no correlation.
* **Analysis:** Below the heatmap, the system provides:
    * Lists of the most positively correlated (least diversifying) and most negatively/least correlated (most diversifying) pairs in your portfolio.
    * An overall assessment of your portfolio's diversification based on the average correlation, with guidance on potential improvements [cite: investment-guru/components/portfolio_analysis.py].
* **Data:** Correlations are calculated based on historical price returns over a specific period (e.g., 1 year) [cite: investment-guru/components/portfolio_analysis.py].

## Risk Assessment

This component evaluates the risk characteristics of your overall portfolio based on its historical performance.

* **Location:** Displayed in the "Portfolio Risk Analysis" card [cite: investment-guru/components/risk_metrics_component.py, investment-guru/main.py].
* **Time Period:** You can select the time period for the analysis (e.g., 3 Months, 1 Year, All Time) using the radio buttons [cite: investment-guru/components/risk_metrics_component.py].
* **Risk Rating Gauge:** A visual gauge displays an overall risk rating for your portfolio on a scale of 1 (Very Low) to 10 (Very High). The rating is calculated based on various underlying metrics [cite: investment-guru/modules/portfolio_risk_metrics.py].
* **Risk Description:** Provides a textual explanation of the calculated risk rating and its implications [cite: investment-guru/modules/portfolio_risk_metrics.py].
* **Key Risk Metrics:** Several cards display important risk metrics [cite: investment-guru/modules/portfolio_risk_metrics.py]:
    * **Sharpe Ratio:** Measures risk-adjusted return (return per unit of total risk/volatility). Higher is generally better. Calculated using portfolio returns, standard deviation, and a risk-free rate.
    * **Sortino Ratio:** Similar to Sharpe Ratio, but only considers downside volatility (risk of losses). Higher is generally better for evaluating returns relative to bad risk.
    * **Max Drawdown (%):** The largest peak-to-trough decline (percentage loss) experienced by the portfolio during the selected period. A smaller negative number (closer to zero) is better.
    * **Volatility (Ann.) (%):** The annualized standard deviation of the portfolio's daily returns. Represents the degree of price fluctuation; lower volatility generally indicates lower risk.
    * **Beta:** Measures the portfolio's volatility relative to the overall market (usually represented by a benchmark index like the S&P/TSX Composite [cite: investment-guru/modules/portfolio_risk_metrics.py]).
        * Beta = 1: Moves with the market.
        * Beta > 1: More volatile than the market.
        * Beta < 1: Less volatile than the market.
    * **Value at Risk (VaR 95%) (%):** Estimates the maximum potential loss the portfolio could experience on a single day, with 95% confidence, based on historical data. A smaller negative number is better.
* **Drawdown Chart:** A line chart visualizing the portfolio's drawdowns (percentage declines from previous peaks) over the selected time period [cite: investment-guru/modules/portfolio_risk_metrics.py].

**Note:** All risk metrics are calculated based on historical data and are not guaranteed predictors of future performance or risk.