"""
Display ALL Backtest Results in Terminal with Proper Column Formatting
=======================================================================

Shows every row of backtest data in properly aligned columns.
"""

import pandas as pd
from pathlib import Path

def display_results(ticker):
    """Display ALL backtest results for a given ticker in a formatted way."""
    
    results_path = Path(f"MA_strategy/backtest_results/{ticker}_2015-01-01_2025-09-30_backtest_results.csv")
    
    print(f"\n{'='*120}")
    print(f"BACKTEST RESULTS FOR {ticker}")
    print(f"{'='*120}\n")
    
    # Load data
    df = pd.read_csv(results_path, parse_dates=["Date"])
    
    # Display basic info
    print(f"Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(df)}")
    print(f"Days in position: {df['Position'].sum():.0f} ({df['Position'].sum()/len(df)*100:.1f}%)")
    print(f"Number of trades: {df['Trade'].sum():.0f}")
    
    # Display ALL rows with key columns
    print(f"\n{'-'*120}")
    print("ALL DATA:")
    print(f"{'-'*120}")
    
    display_cols = ["Date", "Close", "Signal_combined", "Position", "Trade", "Return", 
                    "StratRetNet", "Equity", "BH_Equity"]
    
    # Format the dataframe for display
    df_display = df[display_cols].copy()
    df_display["Date"] = df_display["Date"].dt.strftime('%Y-%m-%d')
    
    # Create formatted string with proper alignment
    print(f"{'Date':<12} {'Close':>10} {'Signal':>8} {'Pos':>5} {'Trade':>6} {'Return':>10} {'StratRet':>10} {'Equity':>10} {'BH_Eq':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*8} {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for _, row in df_display.iterrows():
        print(f"{row['Date']:<12} {row['Close']:>10.2f} {row['Signal_combined']:>8.0f} "
              f"{row['Position']:>5.0f} {row['Trade']:>6.0f} {row['Return']:>10.6f} {row['StratRetNet']:>10.6f} "
              f"{row['Equity']:>10.4f} {row['BH_Equity']:>10.4f}")
    
    # Calculate and display performance metrics
    print(f"\n{'='*120}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*120}")
    
    total_return_strat = (df['Equity'].iloc[-1] - 1) * 100
    total_return_bh = (df['BH_Equity'].iloc[-1] - 1) * 100
    
    n_years = len(df) / 252
    cagr_strat = ((df['Equity'].iloc[-1]) ** (1/n_years) - 1) * 100
    cagr_bh = ((df['BH_Equity'].iloc[-1]) ** (1/n_years) - 1) * 100
    
    print(f"\n{'Metric':<30} {'Strategy':>20} {'Buy & Hold':>20}")
    print(f"{'-'*30} {'-'*20} {'-'*20}")
    print(f"{'Initial Capital':<30} {'$1.00':>20} {'$1.00':>20}")
    print(f"{'Final Value':<30} ${df['Equity'].iloc[-1]:>19.2f} ${df['BH_Equity'].iloc[-1]:>19.2f}")
    print(f"{'Total Return':<30} {total_return_strat:>19.2f}% {total_return_bh:>19.2f}%")
    print(f"{'CAGR (Annualized)':<30} {cagr_strat:>19.2f}% {cagr_bh:>19.2f}%")
    
    print(f"\n{'='*120}\n")


def main():
    """Display results for both AAPL and SPY."""
    for ticker in ["AAPL", "SPY"]:
        try:
            display_results(ticker)
        except FileNotFoundError:
            print(f"\n⚠️  Results file not found for {ticker}. Run backtest_signal_strategy.py first.\n")


if __name__ == "__main__":
    main()
