# Help: Portfolio Rebalancing

## Overview

The Portfolio Rebalancing component helps you maintain your desired investment strategy over time. It analyzes your current asset allocation, compares it to your defined target allocation, and provides specific recommendations on adjustments needed to bring your portfolio back into balance.

This component is divided into three main tabs: Allocation Analysis, Rebalancing Plan, and Target Settings [cite: investment-guru/components/rebalancing_component.py].

## Allocation Analysis Tab

This tab provides insights into how your current portfolio allocation compares to your target.

* **Current vs Target Allocation Chart:** [cite: investment-guru/components/rebalancing_component.py]
    * A bar chart visually compares the percentage currently allocated to each asset class (e.g., Stock, ETF, Bond) against your target percentage for that class. This helps you quickly see which areas are overweight or underweight.
* **Allocation Drift Table:** [cite: investment-guru/components/rebalancing_component.py]
    * This table provides detailed numerical data for each asset class:
        * `Asset Type`: The category of investment (Stock, Bond, etc.).
        * `Current`: The current percentage allocation in your portfolio.
        * `Target`: Your desired target percentage allocation.
        * `Drift`: The difference between Current and Target percentages (Current % - Target %). Positive drift means overweight, negative means underweight.
        * `Current Value`: The current market value (in CAD) held in this asset class.
        * `Target Value`: The ideal market value (in CAD) based on your target percentage and total portfolio value.
        * `Action`: Indicates whether the system suggests Buying, Selling, or Holding within this asset class based on a drift threshold (typically 5% or more [cite: investment-guru/modules/portfolio_utils.py]). Actionable drifts are often highlighted (e.g., red for Sell, green for Buy).
* **Rebalance Summary:** [cite: investment-guru/components/rebalancing_component.py]
    * Provides an overall assessment based on the drift analysis.
    * Indicates if rebalancing is recommended and suggests an urgency level (e.g., Low, Medium, High) based on the total portfolio drift [cite: investment-guru/components/rebalancing_component.py].

## Rebalancing Plan Tab

This tab provides specific, actionable steps suggested to rebalance your portfolio.

* **Rebalancing Recommendations:** [cite: investment-guru/components/rebalancing_component.py]
    * **Summary:** Shows the total dollar amount the plan suggests selling, the total amount to buy, and the net cash impact (Total Sells - Total Buys).
    * **Action Cards:** Lists individual buy or sell recommendations for specific assets (symbols). Each card typically includes:
        * Action (Buy or Sell)
        * Symbol and Name
        * Asset Type
        * Dollar Amount to transact
        * A brief description of the reason (e.g., "to reduce Stock allocation").
    * Recommendations aim to bring your portfolio closer to your target allocation by selling assets in overweight categories and suggesting buys in underweight categories [cite: investment-guru/modules/portfolio_rebalancer.py]. Buy suggestions may be based on your risk profile [cite: investment-guru/modules/portfolio_rebalancer.py].
* **Note:** A disclaimer reminds users to consider transaction costs and tax implications before executing the recommended trades [cite: investment-guru/components/rebalancing_component.py].

## Target Settings Tab

This tab allows you to define and save your desired long-term asset allocation strategy.

* **Target Allocation Sliders:** [cite: investment-guru/components/rebalancing_component.py]
    * Use the sliders to set your target percentage for each asset class (Stock, ETF, Bond, Cash, Mutual Fund, Crypto).
    * The current percentage value for each slider is displayed next to it.
* **Total Allocation Warning:** A message will appear if the sum of your target percentages does not equal 100% [cite: investment-guru/components/rebalancing_component.py, investment-guru/main.py]. Ensure your targets sum to 100 before saving.
* **Target Allocation Chart:** [cite: investment-guru/components/rebalancing_component.py]
    * A Pie Chart dynamically updates as you adjust the sliders, visualizing the target allocation you are setting.
* **Save Target Allocation Button:** [cite: investment-guru/components/rebalancing_component.py, investment-guru/main.py]
    * Click this button to save your adjusted target percentages. This saved target will be used for future rebalancing analysis [cite: investment-guru/modules/portfolio_utils.py].
    * Feedback will indicate whether the save was successful or if there was an error (e.g., percentages not summing to 100) [cite: investment-guru/main.py].

**How Targets are Used:** The target allocation you save here is compared against your current portfolio holdings (calculated from the Portfolio Management component) in the "Allocation Analysis" tab to determine drift and generate the recommendations shown in the "Rebalancing Plan" tab [cite: investment-guru/modules/portfolio_rebalancer.py, investment-guru/modules/portfolio_utils.py].