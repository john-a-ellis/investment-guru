# Help: Asset Tracker

## Overview

The Asset Tracker component allows you to create and manage a personalized watchlist of financial assets (stocks, ETFs, mutual funds, etc.) that you want to monitor, independent of your actual portfolio holdings. This is useful for keeping an eye on potential investments or simply tracking assets of interest.

## Purpose

* **Create a Watchlist:** Add assets you want to follow without adding them to your investment portfolio.
* **Monitor Performance:** Tracked assets are used in other parts of the application, such as the "Market Overview" graph, allowing you to visualize their performance relative to each other and market benchmarks.
* **Information Source:** The list of tracked assets can also be used to tailor news feeds or analysis sections to your interests.

## Adding an Asset to Track

To add an asset to your tracking list:

1.  **Symbol:** Enter the ticker symbol or fund code (e.g., `MFC.TO`, `AAPL`, `MAW104`).
2.  **Name:** Provide a descriptive name for the asset (e.g., `Manulife Financial`, `Apple Inc.`). Both Symbol and Name are required.
3.  **Asset Type:** Select the appropriate type (Stock, ETF, Mutual Fund, etc.) from the dropdown menu.
4.  **Click "Add":** The asset will be added to the "Currently Tracked Assets" table below.

A feedback message will confirm if the asset was added successfully or if there was an issue (e.g., the symbol is already being tracked).

## Currently Tracked Assets Table

This table displays all the assets currently on your tracking list.

* **Columns:**
    * `Symbol`: The ticker symbol or fund code.
    * `Name`: The descriptive name you provided.
    * `Type`: The asset type (Stock, ETF, etc.).
    * `Added Date`: The date you added the asset to the list.
    * `Actions`: Contains a "Remove" button for each asset.
* **Updating:** The table updates automatically when you add or remove an asset.

## Removing a Tracked Asset

To remove an asset from your tracking list:

1.  Locate the asset you wish to remove in the "Currently Tracked Assets" table.
2.  Click the **"Remove"** button in the "Actions" column for that specific asset.
3.  The asset will be removed from the list, and the table will update.

**Note:** Removing an asset here only removes it from your tracking list. It does *not* affect any holdings listed in the main Portfolio Management component.