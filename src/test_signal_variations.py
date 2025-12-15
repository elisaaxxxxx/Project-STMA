#!/usr/bin/env python3
"""
Test different signal combination strategies to find optimal performance.

Ce script utilise la configuration centralisée pour :
- Tester différentes combinaisons de signaux MA
- Effectuer une analyse walk-forward pour éviter le look-ahead bias
- Comparer les performances avec différentes stratégies

Les paramètres sont définis dans project_config.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# Importer la configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import (TICKERS, START_DATE, END_DATE, TRANSACTION_COST, 
                           TRADING_DAYS_PER_YEAR, RESULTS_VARIATIONS_DIR, 
                           get_signals_file_path, print_config, validate_config)

def compute_metrics(returns_series):
    """Calculate CAGR, Volatility, Sharpe, MaxDD from a returns series."""
    n_days = len(returns_series)
    years = n_days / TRADING_DAYS_PER_YEAR
    
    # CAGR
    cum_ret = (1 + returns_series).prod()
    cagr = (cum_ret ** (1/years) - 1) if years > 0 else 0
    
    # Volatility (annualized)
    vol = returns_series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Sharpe (assuming 0 risk-free rate)
    sharpe = (cagr / vol) if vol > 0 else 0
    
    # Max Drawdown
    equity = (1 + returns_series).cumprod()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe': sharpe,
        'MaxDD': max_dd
    }

def backtest_strategy(df, signal_col, cost=None):
    """
    Backtest a strategy using a given signal column.
    
    Args:
        df: DataFrame with Close prices and signal column
        signal_col: Name of the signal column to use
        cost: Transaction cost per trade (uses config if None)
    
    Returns:
        dict with metrics and equity curve
    """
    if cost is None:
        cost = TRANSACTION_COST
    
    df = df.copy()
    
    # Calculate returns
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    
    # Position = yesterday's signal (avoid look-ahead)
    df["Position"] = df[signal_col].shift(1).fillna(0.0)
    
    # Detect trades
    df["Trade"] = (df["Position"] != df["Position"].shift(1)).astype(int)
    
    # Calculate strategy returns
    df["StratRetGross"] = df["Position"] * df["Return"]
    df["StratRetNet"] = df["StratRetGross"] - df["Trade"] * cost
    
    # Equity curves
    df["Equity"] = (1 + df["StratRetNet"]).cumprod()
    
    # Compute metrics
    metrics = compute_metrics(df["StratRetNet"])
    
    # Add additional info
    metrics['Total_Trades'] = df["Trade"].sum()
    metrics['Days_in_Position'] = df["Position"].sum()
    metrics['Final_Equity'] = df["Equity"].iloc[-1]
    
    return metrics, df

def create_signal_variations(df):
    """
    Create different signal combination strategies.
    
    Expects df to have: Signal_5_20_short, Signal_10_50_medium, 
                        Signal_20_100_long, Signal_50_200_vlong
    """
    df = df.copy()
    
    # Get the 4 individual signals
    sig_short = df['Signal_5_20_short'].fillna(0)
    sig_medium = df['Signal_10_50_medium'].fillna(0)
    sig_long = df['Signal_20_100_long'].fillna(0)
    sig_very_long = df['Signal_50_200_vlong'].fillna(0)
    
    # Original strategy: >= 2 signals bullish
    df['Buy_Original'] = df['Buy'].fillna(0)
    
    # Strategy 1: Short-term only (5,20)
    df['Buy_ShortTerm'] = sig_short.astype(int)
    
    # Strategy 2: Long-term only (50,200)
    df['Buy_LongTerm'] = sig_very_long.astype(int)
    
    # Strategy 3: Short OR Long (either 5,20 or 50,200)
    df['Buy_ShortOrLong'] = ((sig_short == 1) | (sig_very_long == 1)).astype(int)
    
    # Strategy 4: All 4 signals must be bullish
    df['Buy_All4'] = ((sig_short == 1) & (sig_medium == 1) & 
                      (sig_long == 1) & (sig_very_long == 1)).astype(int)
    
    # Strategy 5: At least 3 out of 4 signals bullish
    ones_count = sig_short + sig_medium + sig_long + sig_very_long
    df['Buy_3of4'] = (ones_count >= 3).astype(int)
    
    # Strategy 6: Medium-term only (10,50)
    df['Buy_MediumTerm'] = sig_medium.astype(int)
    
    # Strategy 7: Short AND Medium (both 5,20 and 10,50)
    df['Buy_ShortAndMedium'] = ((sig_short == 1) & (sig_medium == 1)).astype(int)
    
    # Strategy 8: Long AND Very Long (both 20,100 and 50,200)
    df['Buy_LongAndVeryLong'] = ((sig_long == 1) & (sig_very_long == 1)).astype(int)
    
    return df

def walk_forward_analysis(df, strategies, train_years=2, test_months=6):
    """
    Perform walk-forward analysis to avoid look-ahead bias.
    
    Args:
        df: DataFrame with all strategy signals
        strategies: List of (name, signal_col) tuples
        train_years: Years of training data to use for strategy selection
        test_months: Months of test data for out-of-sample evaluation
    
    Returns:
        Combined results from all walk-forward periods
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD ANALYSIS")
    print(f"Training window: {train_years} years")
    print(f"Test window: {test_months} months") 
    print(f"{'='*80}\n")
    
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Convert to datetime if needed
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate window sizes in days
    train_days = train_years * 252  # Approximate trading days per year
    test_days = test_months * 21   # Approximate trading days per month
    
    walk_forward_results = []
    all_test_returns = []
    strategy_selections = []
    
    # Start after we have enough training data
    start_idx = train_days
    
    period = 1
    while start_idx + test_days < len(df):
        print(f"Period {period}:")
        
        # Define training and test periods
        train_start = start_idx - train_days
        train_end = start_idx
        test_start = start_idx
        test_end = min(start_idx + test_days, len(df))
        
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        print(f"  Training: {train_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['Date'].iloc[-1].strftime('%Y-%m-%d')} ({len(train_df)} days)")
        print(f"  Testing:  {test_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Date'].iloc[-1].strftime('%Y-%m-%d')} ({len(test_df)} days)")
        
        # Test all strategies on training data to select the best one
        train_results = []
        for name, signal_col in strategies:
            if signal_col in train_df.columns:
                metrics, _ = backtest_strategy(train_df, signal_col, cost=0.001)
                metrics['Strategy'] = name
                metrics['Signal_Col'] = signal_col
                train_results.append(metrics)
        
        if not train_results:
            print("  No valid strategies found, skipping period")
            start_idx += test_days
            period += 1
            continue
        
        # Select best strategy based on training Sharpe ratio
        train_results_df = pd.DataFrame(train_results)
        best_idx = train_results_df['Sharpe'].idxmax()
        best_strategy = train_results_df.iloc[best_idx]
        
        print(f"  Best strategy (training): {best_strategy['Strategy']} (Sharpe: {best_strategy['Sharpe']:.2f})")
        
        # Test the selected strategy on out-of-sample test data
        test_metrics, test_result_df = backtest_strategy(test_df, best_strategy['Signal_Col'], cost=0.001)
        test_metrics['Strategy'] = best_strategy['Strategy']
        test_metrics['Period'] = period
        test_metrics['Train_Start'] = train_df['Date'].iloc[0]
        test_metrics['Train_End'] = train_df['Date'].iloc[-1]
        test_metrics['Test_Start'] = test_df['Date'].iloc[0]
        test_metrics['Test_End'] = test_df['Date'].iloc[-1]
        test_metrics['Train_Sharpe'] = best_strategy['Sharpe']
        
        print(f"  Test performance: CAGR: {test_metrics['CAGR']:.2%}, Sharpe: {test_metrics['Sharpe']:.2f}, MaxDD: {test_metrics['MaxDD']:.2%}")
        
        walk_forward_results.append(test_metrics)
        
        # Store test period returns for overall performance calculation
        test_returns = test_result_df['StratRetNet'].values
        all_test_returns.extend(test_returns)
        
        # Track strategy selections
        strategy_selections.append({
            'Period': period,
            'Strategy': best_strategy['Strategy'],
            'Train_Sharpe': best_strategy['Sharpe'],
            'Test_Sharpe': test_metrics['Sharpe'],
            'Test_Start': test_df['Date'].iloc[0],
            'Test_End': test_df['Date'].iloc[-1]
        })
        
        # Move to next period
        start_idx += test_days
        period += 1
        print()
    
    # Calculate overall performance across all test periods
    all_test_returns = pd.Series(all_test_returns)
    overall_metrics = compute_metrics(all_test_returns)
    
    print(f"{'='*80}")
    print(f"WALK-FORWARD OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Total periods: {len(walk_forward_results)}")
    print(f"Overall CAGR: {overall_metrics['CAGR']:.2%}")
    print(f"Overall Sharpe: {overall_metrics['Sharpe']:.2f}")
    print(f"Overall Max DD: {overall_metrics['MaxDD']:.2%}")
    print(f"Overall Volatility: {overall_metrics['Volatility']:.2%}")
    
    return walk_forward_results, strategy_selections, overall_metrics

def run_comparison(ticker, start_date, end_date):
    """Run all strategy variations for a given ticker using walk-forward analysis."""
    
    # Load data with signals using config path
    input_file = get_signals_file_path(ticker, start_date, end_date)
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found. Run generate_signals.py first.")
        return None
    
    print(f"\nLoading data from: {input_file} (read-only)")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create signal variations
    df = create_signal_variations(df)
    
    # List of strategies to test
    strategies = [
        ('Original (>=2 signals)', 'Buy_Original'),
        ('Short-term only (5,20)', 'Buy_ShortTerm'),
        ('Medium-term only (10,50)', 'Buy_MediumTerm'),
        ('Long-term only (50,200)', 'Buy_LongTerm'),
        ('Short OR Long', 'Buy_ShortOrLong'),
        ('Short AND Medium', 'Buy_ShortAndMedium'),
        ('Long AND VeryLong', 'Buy_LongAndVeryLong'),
        ('>=3 of 4 signals', 'Buy_3of4'),
        ('All 4 signals', 'Buy_All4'),
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing Signal Variations for {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    # Run walk-forward analysis (no look-ahead bias)
    wf_results, strategy_selections, overall_metrics = walk_forward_analysis(df, strategies)
    
    print(f"\n{'='*60}")
    print(f"STRATEGY SELECTION SUMMARY")
    print(f"{'='*60}")
    strategy_counts = {}
    for selection in strategy_selections:
        strategy = selection['Strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(strategy_selections) * 100
        print(f"{strategy:30s}: {count:2d} periods ({pct:4.1f}%)")
    
    # Also run traditional analysis for comparison (WITH look-ahead bias)
    print(f"\n{'='*80}")
    print(f"TRADITIONAL ANALYSIS (WITH LOOK-AHEAD BIAS - FOR COMPARISON)")
    print(f"{'='*80}\n")
    
    traditional_results = []
    equity_curves = {}
    
    # Backtest each strategy on full dataset
    for name, signal_col in strategies:
        metrics, result_df = backtest_strategy(df, signal_col, cost=0.001)
        
        metrics['Strategy'] = name
        traditional_results.append(metrics)
        
        # Set Date as index for proper plotting
        if 'Date' in result_df.columns:
            result_df = result_df.set_index('Date')
        equity_curves[name] = result_df['Equity']
        
        print(f"{name:30s} | CAGR: {metrics['CAGR']:7.2%} | Sharpe: {metrics['Sharpe']:6.2f} | "
              f"MaxDD: {metrics['MaxDD']:7.2%} | Trades: {int(metrics['Total_Trades']):4d}")
    
    # Add Buy & Hold for comparison
    bh_returns = df["Close"].pct_change().fillna(0.0)
    bh_metrics = compute_metrics(bh_returns)
    bh_metrics['Strategy'] = 'Buy & Hold'
    bh_metrics['Total_Trades'] = 0
    bh_metrics['Days_in_Position'] = len(df)
    bh_metrics['Final_Equity'] = (1 + bh_returns).cumprod().iloc[-1]
    traditional_results.append(bh_metrics)
    
    # Set Date as index for Buy & Hold equity curve
    bh_equity = (1 + bh_returns).cumprod()
    if 'Date' in df.columns:
        bh_equity.index = df['Date'].values
    equity_curves['Buy & Hold'] = bh_equity
    
    print(f"\n{'Buy & Hold':30s} | CAGR: {bh_metrics['CAGR']:7.2%} | Sharpe: {bh_metrics['Sharpe']:6.2f} | "
          f"MaxDD: {bh_metrics['MaxDD']:7.2%} | Trades: {0:4d}")
    
    # Create results DataFrames
    traditional_results_df = pd.DataFrame(traditional_results)
    traditional_results_df = traditional_results_df[['Strategy', 'CAGR', 'Volatility', 'Sharpe', 'MaxDD', 
                             'Total_Trades', 'Days_in_Position', 'Final_Equity']]
    
    # Create walk-forward results summary
    wf_results_df = pd.DataFrame(wf_results)
    
    # Combine overall walk-forward performance with strategy info
    results_df = pd.DataFrame([{
        'Strategy': 'Walk-Forward (No Look-Ahead)',
        'CAGR': overall_metrics['CAGR'],
        'Volatility': overall_metrics['Volatility'], 
        'Sharpe': overall_metrics['Sharpe'],
        'MaxDD': overall_metrics['MaxDD'],
        'Total_Trades': 'Variable',
        'Days_in_Position': 'Variable',
        'Final_Equity': (1 + overall_metrics['CAGR']) ** (len(df)/252)
    }])
    
    # Save results to configured testing folder
    output_dir = Path(RESULTS_VARIATIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save walk-forward detailed results
    wf_detailed_file = output_dir / f"{ticker}_walk_forward_detailed.csv"
    wf_results_df.to_csv(wf_detailed_file, index=False)
    
    # Save strategy selections
    selections_df = pd.DataFrame(strategy_selections)
    selections_file = output_dir / f"{ticker}_strategy_selections.csv"
    selections_df.to_csv(selections_file, index=False)
    
    # Save traditional comparison (with look-ahead)
    traditional_file = output_dir / f"{ticker}_traditional_comparison.csv"
    traditional_results_df.to_csv(traditional_file, index=False)
    
    # Save summary comparison
    summary_file = output_dir / f"{ticker}_signal_variations_comparison.csv"
    
    # Create comprehensive comparison
    comparison_df = pd.concat([
        results_df,  # Walk-forward results
        traditional_results_df  # Traditional results
    ], ignore_index=True)
    
    comparison_df.to_csv(summary_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Walk-forward detailed: {wf_detailed_file}")
    print(f"  Strategy selections: {selections_file}")
    print(f"  Traditional comparison: {traditional_file}")
    print(f"  Summary comparison: {summary_file}")
    
    # Plot equity curves
    plot_equity_curves(equity_curves, ticker, start_date, end_date, output_dir)
    
    # Comparison summary
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"Walk-Forward (No Look-Ahead): CAGR: {overall_metrics['CAGR']:7.2%} | Sharpe: {overall_metrics['Sharpe']:6.2f} | MaxDD: {overall_metrics['MaxDD']:7.2%}")
    
    # Find best traditional strategy
    best_traditional = traditional_results_df.loc[traditional_results_df['Sharpe'].idxmax()]
    print(f"Best Traditional (Look-Ahead):  CAGR: {best_traditional['CAGR']:7.2%} | Sharpe: {best_traditional['Sharpe']:6.2f} | MaxDD: {best_traditional['MaxDD']:7.2%}")
    print(f"  Strategy: {best_traditional['Strategy']}")
    
    print(f"Buy & Hold:                     CAGR: {bh_metrics['CAGR']:7.2%} | Sharpe: {bh_metrics['Sharpe']:6.2f} | MaxDD: {bh_metrics['MaxDD']:7.2%}")
    
    return {
        'walk_forward': overall_metrics,
        'traditional_results': traditional_results_df,
        'strategy_selections': strategy_selections,
        'detailed_wf_results': wf_results_df
    }

def plot_equity_curves(equity_curves, ticker, start_date, end_date, output_dir):
    """Plot equity curves for all strategies."""
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: All strategies
    plt.subplot(2, 1, 1)
    for name, equity in equity_curves.items():
        if name != 'Buy & Hold':
            plt.plot(equity, label=name, alpha=0.7, linewidth=1.5)
    plt.plot(equity_curves['Buy & Hold'], label='Buy & Hold', 
             color='black', linewidth=2, linestyle='--')
    plt.title(f'{ticker}: All Strategy Variations - TRADITIONAL ANALYSIS (WITH LOOK-AHEAD BIAS)\n({start_date} to {end_date})', 
              fontsize=13, fontweight='bold')
    plt.ylabel('Equity', fontsize=11)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Top performers vs Buy & Hold
    plt.subplot(2, 1, 2)
    
    # Calculate Sharpe for each strategy
    sharpes = {}
    for name, equity in equity_curves.items():
        if name != 'Buy & Hold':
            returns = equity.pct_change().fillna(0)
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            sharpes[name] = sharpe
    
    # Get top 3 strategies
    top_3 = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for name, _ in top_3:
        plt.plot(equity_curves[name], label=name, linewidth=2, alpha=0.8)
    plt.plot(equity_curves['Buy & Hold'], label='Buy & Hold', 
             color='black', linewidth=2, linestyle='--')
    
    plt.title(f'{ticker}: Top 3 Strategies (by Sharpe Ratio) - BIASED (Full Dataset)', 
              fontsize=13, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Equity', fontsize=11)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / f"{ticker}_signal_variations_equity_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Equity curves plot saved to: {plot_file}")
    plt.close()

def main():
    """Main function to test all signal variations using walk-forward analysis."""
    
    # Validation de la configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Erreurs de configuration :")
        for error in errors:
            print(f"  - {error}")
        return
    
    print_config()
    
    # Utilise les paramètres de la configuration
    tickers = TICKERS
    start_date = START_DATE
    end_date = END_DATE
    
    all_results = {}
    
    for ticker in tickers:
        results = run_comparison(ticker, start_date, end_date)
        if results is not None:
            all_results[ticker] = results
    
    # Print final summary
    print(f"\n{'='*120}")
    print("FINAL SUMMARY: Walk-Forward vs Traditional Analysis")
    print(f"{'='*120}\n")
    
    print(f"{'Ticker':<6} | {'Method':<25} | {'CAGR':<8} | {'Sharpe':<7} | {'MaxDD':<8} | {'Notes'}")
    print(f"{'-'*120}")
    
    for ticker, results in all_results.items():
        # Walk-forward results
        wf_metrics = results['walk_forward']
        print(f"{ticker:<6} | {'Walk-Forward (Clean)':<25} | {wf_metrics['CAGR']:>7.2%} | {wf_metrics['Sharpe']:>6.2f} | {wf_metrics['MaxDD']:>7.2%} | No look-ahead bias")
        
        # Best traditional strategy
        traditional_df = results['traditional_results']
        best_traditional = traditional_df.loc[traditional_df['Sharpe'].idxmax()]
        print(f"{ticker:<6} | {'Best Traditional':<25} | {best_traditional['CAGR']:>7.2%} | {best_traditional['Sharpe']:>6.2f} | {best_traditional['MaxDD']:>7.2%} | {best_traditional['Strategy']}")
        
        # Buy & Hold
        bh_row = traditional_df[traditional_df['Strategy'] == 'Buy & Hold'].iloc[0]
        print(f"{ticker:<6} | {'Buy & Hold':<25} | {bh_row['CAGR']:>7.2%} | {bh_row['Sharpe']:>6.2f} | {bh_row['MaxDD']:>7.2%} | Benchmark")
        print()
    
    print(f"\n{'='*120}")
    print("KEY INSIGHTS:")
    print("- Walk-Forward results show realistic performance without look-ahead bias")
    print("- Traditional analysis results are artificially inflated due to hindsight")
    print("- The difference demonstrates the importance of proper backtesting methodology")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
