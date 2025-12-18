#!/usr/bin/env python3
"""
Pipeline V2 - Fixed for new folder structure
============================================

Folder structure:
data/
  ├── raw/          # Downloaded data
  ├── processed/    # Data with MA and signals
  ├── results/      # Traditional backtest results
  └── ML/           # ML data and results

Usage:
    python main.py --all          # Complete pipeline
    python main.py --traditional  # Traditional pipeline
    python main.py --ml           # ML pipeline
    python main.py --config       # View config
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
