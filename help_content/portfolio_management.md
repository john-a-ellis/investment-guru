# Help: Portfolio Management

## Overview

The Portfolio Management component is your central hub for tracking the specific investments you own. It allows you to add new investment lots, view detailed performance data for each holding, and record buy/sell transactions directly related to your portfolio.

## Adding a New Investment

You can add a new investment purchase (also known as a 'lot') to your portfolio using the form at the top of the component:

1.  **Symbol:** Enter the ticker symbol for the asset (e.g., `MFC.TO`, `AAPL`, `MAW104`). Use the standard symbol recognized by market data providers.
2.  **Number of Shares/Units:** Enter the quantity you purchased.
3.  **Purchase Price:** Enter the price per share/unit at which you bought the asset.
4.  **Purchase Date:** Select the date of the purchase using the calendar picker. It defaults to the current date.
5.  **Asset Type:** Select the type of asset from the dropdown (e.g., Stock, ETF, Mutual Fund, Crypto).
6.  **Click "Add Investment":** This will add the specific purchase lot to your portfolio and automatically record a corresponding "buy" transaction.

A feedback message will appear indicating whether the investment was added successfully.

## Quick Transaction Recording

For quickly recording buy or sell transactions without adding a new *initial* investment lot, use the "Quick Transaction Recording" section:

1.  **Type:** Select "Buy" or "Sell".
2.  **Symbol:** Enter the ticker symbol of the asset involved in the transaction.
3.  **Shares:** Enter the number of shares/units bought or sold.
4.  **Price:** Enter the price per share/unit for the transaction.
5.  **Date:** Select the transaction date. It defaults to the current date.
6.  **Click "Record Transaction":** This records the transaction and updates your portfolio holdings accordingly (e.g., increases shares on a buy, decreases on a sell).

Feedback on the transaction status will be displayed.

## Current Portfolio View

This section displays your current holdings, grouped by asset symbol in an accordion format.

### Portfolio Summary

A summary card is displayed above the accordion, showing the overall performance of your portfolio:
* **Book Value:** The total cost basis of your investments.
* **Current Value:** The current market value of all your holdings.
* **Gain/Loss ($):** The total profit or loss in currency value.
* **Gain/Loss (%):** The total profit or loss as a percentage.

### Investment Accordion

Each item in the accordion represents a unique asset symbol you hold.

* **Header:** The collapsed view (header) for each symbol provides a quick summary:
    * Symbol
    * Asset Type
    * Total Shares Held
    * Current Market Price
    * Total Book Value (for that symbol)
    * Total Current Value (for that symbol)
    * Total Gain/Loss ($) (for that symbol)
    * Total Gain/Loss (%) (for that symbol)
    * Currency

* **Expanding an Item:** Clicking on an accordion header expands it to reveal three tabs with more details for that specific symbol:

    1.  **Positions Tab:**
        * Lists every individual purchase lot for that symbol.
        * Shows Purchase Date, Shares, Purchase Price, Book Value (for that lot), Current Value (for that lot), Gain/Loss ($), and Gain/Loss (%).
        * Includes a "Remove" button for each lot. **Important:** Clicking "Remove" deletes that specific purchase record from your portfolio entirely. Use the "Record Transaction" tab or the "Quick Transaction Recording" section to record *sales*.

    2.  **Transaction History Tab:**
        * Displays a table of all recorded buy and sell transactions associated with that specific symbol, ordered by date (most recent first).
        * Includes Date, Type (Buy/Sell), Shares, Price, and Total Amount for each transaction.

    3.  **Record Transaction Tab:**
        * Provides dedicated forms to quickly record a **Buy** or **Sell** transaction specifically for the symbol associated with that accordion item.
        * Enter Shares, Price, and Date, then click the "Buy" or "Sell" button.
        * Feedback will appear below the respective form. Recording a transaction here updates your holdings and adds an entry to the Transaction History tab.

### Data Updates

The portfolio data (current prices, values, gain/loss) is automatically updated periodically. Adding investments or recording transactions will also trigger an update of the portfolio table.