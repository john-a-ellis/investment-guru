# run_backtest.py
from modules.backtester import Backtester
from datetime import datetime

if __name__ == "__main__":
    # --- Configuration ---
    SYMBOLS_TO_TEST = ['MFC.TO', 'TRI.TO', 'CGL-C.TO', 'CWW.TO', 'XTR.TO'] # Example symbols [cite: 27]
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"
    INITIAL_CAPITAL = 100000.0
    STRATEGY_PARAMS = {} # Add any params your strategy needs

    print(f"Starting backtest for {SYMBOLS_TO_TEST} from {START_DATE} to {END_DATE}")

    try:
        # --- Initialize ---
        bt = Backtester(
            symbols=SYMBOLS_TO_TEST,
            start_date=START_DATE,
            end_date=END_date,
            initial_capital=INITIAL_CAPITAL,
            strategy_params=STRATEGY_PARAMS
        )

        # --- Run ---
        bt.run()

        # --- Results ---
        results = bt.get_results()
        print("\n--- Backtest Results ---")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}" if isinstance(value, (int, float)) else f"{metric}: {value}")

        # --- Plot ---
        fig = bt.plot_results()
        if fig:
            fig.show() # Display the plot
            # Optionally save the plot
            # fig.write_html("backtest_results.html")

    except Exception as e:
        print(f"\n--- Backtest Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()