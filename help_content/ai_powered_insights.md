# Help: AI-Powered Insights

This section details the components of the AI Investment Recommendation System (AIRS) that leverage artificial intelligence and machine learning models to provide insights, predictions, and recommendations.

## Investment Recommendations

AIRS offers recommendations in two main ways:

1.  **General Recommendations Card:**
    * **Purpose:** Provides broad investment ideas based on your user profile (risk level, investment horizon) and potentially overall market conditions [cite: investment-guru/main.py, investment-guru/modules/recommendation_engine.py].
    * **How it Works:** Click the "Generate Recommendations" button. The system (currently using sample data [cite: investment-guru/main.py]) presents suggested asset classes, allocation percentages, example specific assets, and reasoning based on your profile [cite: investment-guru/main.py, investment-guru/modules/recommendation_engine.py].
    * **Note:** The underlying `RecommendationEngine` is designed to use market data, news, and economic indicators for more sophisticated recommendations, though the current UI button triggers sample data [cite: investment-guru/modules/recommendation_engine.py].

2.  **Portfolio Insights Tab (within AI Investment Analysis):**
    * **Purpose:** Offers specific Buy/Sell recommendations for assets based on ML analysis of your current portfolio holdings and tracked assets [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/model_integration.py].
    * **How it Works:** Click the "Generate Portfolio Insights" button within the "Portfolio Insights" tab of the "AI Investment Analysis" component [cite: investment-guru/components/ml_prediction_component.py].
    * **Output:**
        * **Portfolio Health Score:** A score (0-100) assessing the overall health of your portfolio based on ML analysis [cite: investment-guru/components/ml_prediction_component.py].
        * **Buy/Sell Recommendations:** Tables listing specific assets recommended for buying or selling, along with confidence scores and expected returns. Buttons may be available to directly initiate (or record) these transactions [cite: investment-guru/components/ml_prediction_component.py].

## Price Prediction

* **Purpose:** Utilizes machine learning models (like Prophet, and potentially LSTM or ARIMA) to forecast the future price movements of selected assets [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/price_prediction.py].
* **Location:** Found in the "Price Prediction" tab within the "AI Investment Analysis" component [cite: investment-guru/components/ml_prediction_component.py].
* **How to Use:**
    1.  Select an asset from the dropdown list (includes portfolio and tracked assets) [cite: investment-guru/components/ml_prediction_component.py].
    2.  Choose a "Prediction Horizon" (e.g., 30, 60, or 90 days) [cite: investment-guru/components/ml_prediction_component.py].
    3.  Click the "Analyze" button [cite: investment-guru/components/ml_prediction_component.py].
* **Output:**
    * **Prediction Chart:** Displays the historical price alongside the model's predicted future price path. It may also include confidence interval bands (upper/lower bounds) [cite: investment-guru/components/ml_prediction_component.py].
    * **Prediction Details:** A summary card showing the asset, model used (e.g., Prophet, Fallback), prediction horizon, current price, predicted future price, and the expected percentage return. It also provides a simplified "Investment Recommendation" (e.g., Consider Buying, Consider Selling, Hold/Neutral) based on the prediction [cite: investment-guru/components/ml_prediction_component.py].
* **Disclaimer:** Remember that these are model-based predictions based on historical data and assumptions. They are not guarantees of future performance. Always conduct your own research [cite: investment-guru/components/ml_prediction_component.py].

## Model Training & Management

* **Purpose:** Allows users to manage the machine learning models used for price predictions, including initiating training for specific assets and viewing model status and performance [cite: investment-guru/components/ml_prediction_component.py].
* **Location:** Found in the "Model Training" tab within the "AI Investment Analysis" component [cite: investment-guru/components/ml_prediction_component.py].
* **Sections:**
    1.  **Trained Model Overview:**
        * Displays a table listing models that have been trained and their metadata saved in the database [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/db_utils.py].
        * Information includes the model filename, associated symbol, model type (e.g., Prophet), training date, key performance metrics (like MAE, MSE, RMSE - loaded from the database `metrics` field [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/db_utils.py]), and any notes.
        * Use the "Refresh Model List" button to fetch the latest data from the database [cite: investment-guru/components/ml_prediction_component.py].
    2.  **Train New Model / View Status:**
        * **Select Asset:** Choose an asset from the dropdown list for which you want to train (or retrain) a prediction model [cite: investment-guru/components/ml_prediction_component.py].
        * **Train Model Button:** Clicking this initiates the training process for the selected asset. Training runs in the background (asynchronously) [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/model_integration.py].
        * **Immediate Feedback:** A message appears indicating that training has started [cite: investment-guru/components/ml_prediction_component.py].
        * **Training Status Table:** This table shows the live status (Pending, In Progress, Completed, Failed) for models currently being trained or previously trained in the current session. It may also show errors if training failed [cite: investment-guru/components/ml_prediction_component.py, investment-guru/modules/model_integration.py]. Upon successful completion and metadata saving, the model will appear in the "Trained Model Overview" table after refreshing.