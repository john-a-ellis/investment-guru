# Help: User Profile

## Overview

The User Profile component allows you to set your personal investment preferences, which help tailor the system's recommendations and analysis to your specific situation.

## Settings

* **Risk Tolerance:** [cite: investment-guru/components/user_profile.py]
    * **Control:** A slider ranging from 1 (very risk-averse) to 10 (very risk-tolerant).
    * **Purpose:** Indicates your comfort level with investment risk. A lower score suggests a preference for lower-risk, potentially lower-return investments (like bonds and cash), while a higher score suggests openness to higher-risk, potentially higher-return investments (like stocks and crypto). This setting directly influences the asset allocation suggested by the recommendation engine [cite: investment-guru/modules/recommendation_engine.py].
* **Investment Horizon:** [cite: investment-guru/components/user_profile.py]
    * **Control:** A dropdown menu with options:
        * Short Term (< 1 year)
        * Medium Term (1-5 years)
        * Long Term (> 5 years)
    * **Purpose:** Defines the length of time you plan to keep your investments before needing the money. Longer horizons generally allow for taking on more risk, as there's more time to recover from potential downturns. This adjusts the base asset allocation [cite: investment-guru/modules/recommendation_engine.py].
* **Initial Investment:** [cite: investment-guru/components/user_profile.py]
    * **Control:** A numerical input field.
    * **Purpose:** Enter the starting capital amount for your portfolio. This value is used as a reference point, potentially for calculating overall returns or scaling recommendations.

## Saving Your Profile

* **Persistence:** The values you set for Risk Tolerance, Investment Horizon, and Initial Investment are automatically remembered locally in your browser session [cite: investment-guru/components/user_profile.py].
* **Update Profile Button:** Clicking this button explicitly saves your current settings to the system's database [cite: investment-guru/main.py, investment-guru/modules/portfolio_utils.py]. This ensures your preferences are used consistently, especially for generating recommendations. Feedback will be displayed indicating if the update was successful [cite: investment-guru/main.py].