#!/usr/bin/env python3
"""
MAIN PIPELINE - ML-Enhanced Moving Average Trading Strategy
===========================================================

WHAT THIS CODE DOES:
-------------------
This is the master orchestration script that coordinates the entire research project.
It executes both traditional and ML-enhanced trading strategies, compares their performance,
and generates academic tables for the final report.

RESEARCH QUESTION:
-----------------
Can machine learning improve upon traditional walk-forward moving average strategies, 
and what is the economic significance of the improvement after accounting for transaction costs?

HOW THIS PIPELINE WORKS:
-----------------------
The pipeline is divided into two main branches that can run independently or together:

1. TRADITIONAL PIPELINE (--traditional or --all):
   Step 1: Download stock data (src/data_loader.py)
           - Downloads historical price data for all tickers from Yahoo Finance
           - Saves to: data/SRC/raw/
   
   Step 2: Calculate moving averages (src/calculate_moving_averages.py)
           - Computes 4 MA pairs: (5,20), (10,50), (20,100), (50,200)
           - Saves to: data/SRC/processed/*_with_MAs.csv
   
   Step 3: Generate trading signals (src/generate_signals.py)
           - Creates binary signals (1=bullish, 0=bearish) for each MA pair
           - Combines into 4-digit signal (e.g., "1111" = all bullish)
           - Buy signal = at least 2 out of 4 MAs bullish
           - Saves to: data/SRC/processed/*_with_signals.csv
   
   Step 4: Test strategy variations (src/test_signal_variations.py)
           - Tests 10 different rule combinations (e.g., "at least 2/4", "at least 3/4")
           - Compares Walk-Forward (no bias) vs Best Traditional (biased)
           - Walk-Forward: trains on 36 months → tests on 6 months → rolls forward
           - Saves to: data/SRC/results/variations/
   
   Step 5: Backtest best strategy (src/backtest_signal_strategy.py)
           - Applies best walk-forward strategy to entire period
           - Accounts for 0.1% transaction costs per trade
           - Saves to: data/SRC/results/backtest/

2. ML PIPELINE (--ml or --all):
   Step 1: Create ML training data (ML/create_ml_data.py)
           - Engineers 21 features in 3 categories:
             * 14 global features (market conditions, volatility, momentum)
             * 5 MA-specific features (distance to MA, crossover signals)
             * 2 MA pair parameters (which MA pair: 5-20, 10-50, etc.)
           - Creates one row per (date, MA_pair) combination
           - Expands dataset from ~6,500 rows to ~78,000 rows (12 MA pairs × dates)
           - Saves to: data/ML/*_ml_data.csv
   
   Step 2: Analyze regularization (ML/analyze_lasso_regularization.py)
           - Tests different values of alpha (regularization strength)
           - Finds optimal alpha that balances bias vs variance
           - Creates 4-panel diagnostic plots
           - Saves analysis to: data/ML/regularization_analysis/
   
   Step 3: Train regression model (ML/train_regression_model.py)
           - Trains 5 models: Linear, Ridge, Lasso, ElasticNet, SGD
           - Uses walk-forward validation (36m train → 6m test)
           - Lasso wins due to automatic feature selection (2-3 features from 21)
           - Low R² (~1%) is NORMAL in finance - we care about economic value, not fit
           - Saves models to: ML/models/
   
   Step 4: Show optimal features (ML/show_optimal_features.py)
           - Displays which features Lasso selected (non-zero coefficients)
           - Example for AAPL: signal_t (+0.002721), spy_ret_20d (+0.000487)
           - Only 2 features needed out of 21 available
   
   Step 5: Backtest ML strategy (ML/backtest_ml_strategy.py)
           - For each day: predicts next-day returns for all 12 MA pairs
           - Selects the MA pair with highest predicted return
           - Generates buy/sell signal using that pair's crossover
           - Dynamically switches between MA pairs as market conditions change
           - Accounts for 0.1% transaction costs
           - Saves to: data/ML/backtest_results/

3. RESULTS GENERATION:
   After both pipelines complete, this script automatically runs show_results.py
   which generates 8 academic tables:
   
   - Table 1: Overall Performance (AVERAGES across 7 tickers)
   - Table 2: CAGR by Ticker (INDIVIDUAL results per ticker)
   - Table 3: Sharpe by Ticker (INDIVIDUAL results per ticker)
   - Table 4: ML Metrics (AVERAGES: R², RMSE, MAE, feature counts)
   - Table 5: Economic Significance (AAPL EXAMPLE: terminal wealth comparison)
   - Table 6: Feature Importance (AAPL EXAMPLE: which features selected)
   - Table 7: Model Comparison (AAPL EXAMPLE: Lasso vs other models)
   - Table 8: Transaction Cost Impact (AVERAGES: with vs without costs)


FOLDER STRUCTURE:
----------------
data/
  ├── SRC/
  │   ├── raw/                    # Downloaded price data
  │   ├── processed/              # Data with MAs and signals
  │   └── results/                # Traditional backtest results
  ├── ML/                         # ML training data and results
  │   ├── backtest_results/
  │   ├── regularization_analysis/
  │   └── models/
  └── tables_for_report/          # Academic tables (CSV format)

USAGE:
-----
    python main.py --all          # Run complete pipeline (traditional + ML)
    python main.py --traditional  # Run only traditional strategies
    python main.py --ml           # Run only ML strategy
    python main.py --config       # View current configuration (tickers, dates)

CONFIGURATION:
-------------
All parameters are defined in project_config.py:
- TICKERS: List of stocks to analyze (default: AAPL, NVDA, JPM, BAC, PG, KO, JNJ)
- START_DATE: Beginning of analysis period (default: 2000-01-01)
- END_DATE: End of analysis period (default: 2025-11-01)
- BENCHMARK_TICKER: Market benchmark (default: SPY - S&P 500)

REQUIREMENTS:
------------
- AAPL must be in TICKERS list (required for Tables 5, 6, 7 which use AAPL as example)
- Minimum 5+ years of data for meaningful walk-forward analysis
- Internet connection for initial data download from Yahoo Finance
"""

import sys
import argparse
import subprocess
from pathlib import Path
import os

# Import configuration
from project_config import (TICKERS, ALL_TICKERS, BENCHMARK_TICKER, 
                           START_DATE, END_DATE, print_config, validate_config)

def ensure_directories():
    """Creates all necessary directories."""
    dirs = [
        'data/SRC/raw',
        'data/SRC/processed', 
        'data/SRC/results/backtest',
        'data/SRC/results/variations',
        'data/ML',
        'data/ML/backtest_results',
        'data/ML/regularization_analysis',
        'ML/models'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ All directories created")

def check_data_files():
    """Checks if raw files exist."""
    missing = []
    for ticker in ALL_TICKERS:  # Check all tickers including benchmark
        file = Path(f"data/SRC/raw/{ticker}_{START_DATE}_{END_DATE}.csv")
        if not file.exists():
            missing.append(ticker)
    
    if missing:
        print(f"\n⚠️  Missing data: {', '.join(missing)}")
        print("📥 Automatic download...")
        
        try:
            result = subprocess.run(
                [sys.executable, "src/data_loader.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Data downloaded!")
                return []
            else:
                print(f"❌ Error: {result.stderr}")
                return missing
        except Exception as e:
            print(f"❌ Error: {e}")
            return missing
    
    return []

def run_traditional_pipeline():
    """Executes the traditional pipeline."""
    print("\n" + "="*70)
    print("🚀 TRADITIONAL PIPELINE")
    print("="*70)
    print_config()
    
    # Check data
    ensure_directories()
    missing = check_data_files()
    if missing:
        print(f"\n❌ Unable to download: {', '.join(missing)}")
        return False
    
    # Steps
    scripts = [
        ("Moving averages", "src/calculate_moving_averages.py"),
        ("Signals", "src/generate_signals.py"),
        ("Backtest", "src/backtest_signal_strategy.py"),
        ("Variations", "src/test_signal_variations.py")
    ]
    
    for name, script in scripts:
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print("="*70)
        
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False  # Display output in real time
        )
        
        if result.returncode != 0:
            print(f"\n❌ Failed: {name}")
            return False
    
    print("\n" + "="*70)
    print("✅ TRADITIONAL PIPELINE COMPLETED!")
    print("="*80 + "\n")
    print("📊 Results:")
    print("  • data/SRC/processed/ - Data with MA and signals")
    print("  • data/SRC/results/backtest/ - Backtests")
    print("  • data/SRC/results/variations/ - Walk-forward")
    print("="*70)
    
    return True

def run_ml_pipeline():
    """Executes the ML pipeline."""
    print("\n" + "="*70)
    print("🤖 MACHINE LEARNING PIPELINE")
    print("="*70)
    print_config()
    
    ensure_directories()
    
    # Check that processed data exists
    print("\n📋 Checking processed data...")
    missing = []
    for ticker in TICKERS:
        file = Path(f"data/SRC/processed/{ticker}_{START_DATE}_{END_DATE}_with_signals.csv")
        if not file.exists():
            missing.append(ticker)
    
    if missing:
        print(f"⚠️  Missing processed data: {', '.join(missing)}")
        print("📊 Running traditional pipeline first...")
        if not run_traditional_pipeline():
            return False
    
    # ML steps
    ml_steps = [
        ("Create ML datasets", "ML/create_ml_data.py", None),
        ("Train models", "ML/train_regression_model.py", None),
        ("Regularization analysis", "ML/analyze_lasso_regularization.py", ["--n-alphas", "50"]),
        ("ML Backtest", "ML/backtest_ml_strategy.py", ["--model", "lasso_regression"])
    ]
    
    for name, script, extra_args in ml_steps:
        print(f"\n{'='*70}")
        print(f"🤖 {name}")
        print("="*70)
        
        for ticker in TICKERS:
            print(f"\n📊 {ticker}...")
            
            cmd = [sys.executable, script, "--ticker", ticker]
            if extra_args:
                cmd.extend(extra_args)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Error for {ticker}:")
                print(result.stderr)
                return False
            else:
                # Display only important lines
                for line in result.stdout.split('\n'):
                    if any(x in line for x in ['✅', '✓', 'CAGR', 'Sharpe', 'Test R²', 'Best']):
                        print(line)
    
    print("\n" + "="*70)
    print("✅ ML PIPELINE COMPLETED!")
    print("="*70)
    print("📊 ML Results:")
    print("  • data/ML/ - ML Datasets")
    print("  • ML/models/ - Trained models")
    print("  • data/ML/regularization_analysis/ - Analyses")
    print("  • data/ML/backtest_results/ - ML Backtests")
    print("="*70)
    
    return True

def run_full_pipeline():
    """Complete pipeline."""
    print("\n" + "="*80)
    print("🚀 COMPLETE PIPELINE (TRADITIONAL + ML)")
    print("="*80)
    
    # Phase 1: Traditional
    print("\n" + "="*80)
    print("PHASE 1: TRADITIONAL PIPELINE")
    print("="*80)
    
    if not run_traditional_pipeline():
        print("\n❌ Phase 1 failed")
        return False
    
    # Phase 2: ML
    print("\n" + "="*80)
    print("PHASE 2: ML PIPELINE")
    print("="*80)
    
    if not run_ml_pipeline():
        print("\n❌ Phase 2 failed")
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("✅✅✅ COMPLETE PIPELINE FINISHED! ✅✅✅")
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"\n  TRADITIONAL PIPELINE:")
    print(f"    • Data: data/SRC/processed/")
    print(f"    • Backtests: data/SRC/results/backtest/")
    print(f"    • Walk-forward: data/SRC/results/variations/")
    print(f"\n  ML PIPELINE:")
    print(f"    • Datasets: data/ML/")
    print(f"    • Models: ML/models/")
    print(f"    • Analyses: data/ML/regularization_analysis/")
    print(f"    • ML Backtests: data/ML/backtest_results/")
    print(f"  TICKERS: {', '.join(TICKERS)}")
    print(f"  BENCHMARK: {BENCHMARK_TICKER}")
    print(f"  PERIOD: {START_DATE} → {END_DATE}")
    print("="*80 + "\n")
    
    # Show comprehensive results
    show_results()
    
    return True

def show_results():
    """Display comprehensive results using show_results.py."""
    
    print("\n" + "="*80)
    print("📊 DISPLAYING COMPLETE RESULTS")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            ['python3', 'show_results.py'],
            check=True,
            capture_output=False  # Show output directly
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error displaying results: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    
    # Config validation
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    parser = argparse.ArgumentParser(
        description="Pipeline V2 - Fixed structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Complete pipeline')
    parser.add_argument('--traditional', action='store_true',
                       help='Traditional pipeline')
    parser.add_argument('--ml', action='store_true',
                       help='ML pipeline')
    parser.add_argument('--config', action='store_true',
                       help='Display config')
    
    args = parser.parse_args()
    
    # If no arguments
    if not any(vars(args).values()):
        print("\n" + "="*70)
        print("🎯 PIPELINE V2 - FIXED STRUCTURE")
        print("="*70)
        print_config()
        print("\n📋 OPTIONS:")
        print("  1. Complete pipeline (Traditional + ML)")
        print("  2. Traditional pipeline only")
        print("  3. ML pipeline only")
        print("  4. Display configuration")
        print("  5. Quit")
        
        try:
            choice = input("\n👉 Choose (1-5): ").strip()
            
            if choice == '1':
                run_full_pipeline()
            elif choice == '2':
                run_traditional_pipeline()
            elif choice == '3':
                run_ml_pipeline()
            elif choice == '4':
                print_config()
            elif choice == '5':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid option")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
        return 0
    
    # Execute according to arguments
    if args.config:
        print_config()
    
    success = True
    if args.all:
        success = run_full_pipeline()
    elif args.traditional:
        success = run_traditional_pipeline()
    elif args.ml:
        success = run_ml_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
