# Help: Mutual Fund Management

## Overview

The Mutual Fund Management component provides a way to manually add and view historical price data (Net Asset Value or NAV) for mutual funds. This is particularly useful for funds whose prices might not be automatically retrieved by the system's primary data providers.

## Adding a Price Point

If you need to add a specific NAV price for a mutual fund on a particular date:

1.  **Fund Code:** Enter the unique code or symbol for the mutual fund (e.g., `MAW104`) [cite: investment-guru/components/mutual_fund_manager.py].
2.  **Date:** Select the date for which the price is valid using the calendar picker [cite: investment-guru/components/mutual_fund_manager.py].
3.  **NAV Price:** Enter the Net Asset Value per unit/share for that date [cite: investment-guru/components/mutual_fund_manager.py].
4.  **Click "Add Price":** This saves the price point to the system's database for that specific fund and date [cite: investment-guru/main.py, investment-guru/modules/mutual_fund_provider.py, investment-guru/modules/mutual_fund_db.py].

Feedback will indicate if the price was added successfully [cite: investment-guru/main.py]. If a price for that fund on that date already exists, it will be updated [cite: investment-guru/modules/mutual_fund_db.py].

## Viewing Price History

To view the price history you have manually entered for a specific fund:

1.  **Enter Fund Code:** Type the fund code into the "Enter Fund Code to View" input field [cite: investment-guru/components/mutual_fund_manager.py].
2.  **Click "View":** A table will appear displaying the dates and corresponding prices stored in the database for that fund, ordered by date [cite: investment-guru/main.py, investment-guru/modules/mutual_fund_provider.py].

If no data has been entered for that fund code, a message indicating this will be shown [cite: investment-guru/main.py].

## How This Data is Used

The price data entered here is used by the system, primarily via the central `DataProvider` module, to:
* Calculate the current value of your mutual fund holdings in the Portfolio Management component [cite: investment-guru/modules/data_provider.py, investment-guru/modules/portfolio_utils.py].
* Provide historical data points for charts and analysis involving these specific mutual funds [cite: investment-guru/modules/data_provider.py].
The system may prioritize this manually entered data over other sources for these specific funds [cite: investment-guru/modules/data_provider.py, investment-guru/modules/mutual_fund_provider.py].