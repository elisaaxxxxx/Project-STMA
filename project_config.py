"""
Centralized configuration for the trading strategy project
===========================================================

Modify the values below to change tickers, dates, 
and other project parameters.

All scripts will automatically use these parameters.
"""

# ===== TICKERS TO ANALYZE =====
# Tickers to trade (change this list according to your needs)
TICKERS = [
    # Tech (best ML performers)
    'AAPL',   # 📱 Apple - ML +2.19% vs B&H
    'NVDA',   # 🎮 Nvidia - ML +21.49% vs B&H
    
    # Finance (stable, predictable)
    'JPM',    # 🏦 JP Morgan - Leading bank
    'BAC',    # 🏦 Bank of America
    
    # Consumer Staples (defensive, stable)
    'PG',     # 🧼 Procter & Gamble - Consumer goods
    'KO',     # 🥤 Coca-Cola - Beverages
    
    # Healthcare (stable growth)
    'JNJ',    # 💊 Johnson & Johnson - Pharma
]

# SPY as benchmark only (for ML features)
BENCHMARK_TICKER = 'SPY'  # 📊 S&P 500 ETF - Benchmark only

# Complete list (tickers + benchmark) for data download
ALL_TICKERS = TICKERS + [BENCHMARK_TICKER]

# Characteristics of each ticker:
# - AAPL: Tech leader, strong growth, high volatility
# - NVDA: Semiconductor, very strong growth, very volatile (AI boom)
# - JPM: Bank, cyclical, correlated with interest rates
# - JNJ: Pharma/Healthcare, defensive, low volatility
# - XOM: Energy, cyclical, correlated with oil

# Note: SPY is used only as BENCHMARK (in ML features)
# but is NOT traded directly

# Examples of other interesting tickers to test:
# Tech stocks (FAANG+):
# TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', ]

# Diversified ETFs:
# TICKERS = ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA', 'EFA', 'VWO']

# Specific sectors:
# TICKERS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI']  # Energy, Finance, Tech, Health, Industry

# Commodities:
# TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']  # Gold, Silver, Oil, Gas, Agriculture

# Defensive stocks:
# TICKERS = ['JNJ', 'PG', 'KO', 'WMT', 'PFE']  # Consumer staples & healthcare

# Crypto (if supported by yfinance):
# TICKERS = ['BTC-USD', 'ETH-USD']

# ===== ANALYSIS PERIOD =====
# Format: 'YYYY-MM-DD'
START_DATE = '2000-01-01'
END_DATE = '2025-11-01'  # Until November 1st, 2025

# Examples of other periods:
# START_DATE = '2020-01-01'  # Last 5 years
# START_DATE = '2010-01-01'  # Last 15 years
# END_DATE = '2024-12-31'    # Until end of 2024

# ===== MOVING AVERAGES PARAMETERS =====
# Moving average periods to calculate
MA_PERIODS = [5, 10, 20, 50, 100, 200]

# Comparisons to generate signals (short term vs long term)
MA_COMPARISONS = [
    {'short': 5, 'long': 20, 'name': 'Signal_5_20_short'},      # Short term
    {'short': 10, 'long': 50, 'name': 'Signal_10_50_medium'},   # Medium term  
    {'short': 20, 'long': 100, 'name': 'Signal_20_100_long'},   # Long term
    {'short': 50, 'long': 200, 'name': 'Signal_50_200_vlong'}   # Very long term
]

# ===== BACKTEST PARAMETERS =====
# Transaction cost per trade (in percentage)
TRANSACTION_COST = 0.001  # 0.1% per transaction

# Number of trading days per year (for annualization)
TRADING_DAYS_PER_YEAR = 252

# ===== WALK-FORWARD PARAMETERS =====
# Training period in months
TRAINING_MONTHS = 36  # 3 years

# Test period in months
TEST_MONTHS = 6  # 6 months

# ===== DIRECTORIES =====
# Use ABSOLUTE paths based on this file's location
# This ensures data is always created in the right place,
# regardless of where the script is launched from
import os
from pathlib import Path

# Find the project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Organized structure: SRC for traditional pipeline, ML for machine learning
DATA_RAW_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'raw')                    # Raw data (downloaded CSVs)
DATA_PROCESSED_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'processed')        # Data with MA and signals  
RESULTS_BACKTEST_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'results' / 'backtest')     # Backtest results
RESULTS_VARIATIONS_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'results' / 'variations')  # Variation tests

# Old names for compatibility (DEPRECATED)
DATA_DIR = DATA_RAW_DIR
STRATEGY_DIR = DATA_PROCESSED_DIR
RESULTS_DIR = RESULTS_BACKTEST_DIR
VARIATIONS_DIR = RESULTS_VARIATIONS_DIR

# ===== UTILITY FUNCTIONS =====

def get_data_file_path(ticker, start_date=None, end_date=None):
    """Generates path to raw data file."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_RAW_DIR}/{ticker}_{start_date}_{end_date}.csv"

def get_ma_file_path(ticker, start_date=None, end_date=None):
    """Generates path to file with moving averages."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_PROCESSED_DIR}/{ticker}_{start_date}_{end_date}_with_MAs.csv"

def get_signals_file_path(ticker, start_date=None, end_date=None):
    """Generates path to file with signals."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_PROCESSED_DIR}/{ticker}_{start_date}_{end_date}_with_signals.csv"

def get_backtest_file_path(ticker, start_date=None, end_date=None):
    """Generates path to backtest results file."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{RESULTS_BACKTEST_DIR}/{ticker}_{start_date}_{end_date}_backtest_results.csv"

def print_config():
    """Displays current configuration."""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"Tickers: {TICKERS}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Moving averages: {MA_PERIODS}")
    print(f"Transaction cost: {TRANSACTION_COST:.4f}")
    print(f"Walk-Forward: {TRAINING_MONTHS} months training, {TEST_MONTHS} months test")
    print("=" * 60)

def validate_config():
    """Validates configuration."""
    errors = []
    
    if not TICKERS:
        errors.append("TICKERS cannot be empty")
    
    try:
        from datetime import datetime
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
        if start_dt >= end_dt:
            errors.append("START_DATE must be before END_DATE")
    except ValueError:
        errors.append("Invalid date format (use YYYY-MM-DD)")
    
    if not MA_PERIODS or not all(isinstance(p, int) and p > 0 for p in MA_PERIODS):
        errors.append("MA_PERIODS must contain positive integers")
    
    if not (0 <= TRANSACTION_COST <= 1):
        errors.append("TRANSACTION_COST must be between 0 and 1")
    
    return len(errors) == 0, errors

# ===== CONFIGURATION MANAGEMENT FUNCTIONS =====

def update_tickers(new_tickers):
    """Updates the list of tickers in the configuration file."""
    import os
    import shutil
    
    print(f"🔄 Updating tickers: {new_tickers}")
    
    # Read the file
    with open('project_config.py', 'r') as f:
        content = f.read()
    
    # Build the new TICKERS line
    tickers_list = [f"'{ticker.strip()}'" for ticker in new_tickers]
    new_tickers_line = f"TICKERS = [{', '.join(tickers_list)}]"
    
    # Replace the TICKERS line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('TICKERS = [') and not line.strip().startswith('#'):
            lines[i] = new_tickers_line
            break
    
    # Save
    with open('project_config.py', 'w') as f:
        f.write('\n'.join(lines))
    
    # Clear cache
    clear_cache()
    
    # Check which data is missing and propose download
    check_and_download_missing_data(new_tickers)
    
    print("✅ Tickers updated!")

def check_and_download_missing_data(tickers):
    """Checks and automatically downloads missing data."""
    import os
    import sys
    from pathlib import Path
    
    missing_tickers = []
    
    # Check which files are missing
    for ticker in tickers:
        data_file = f"data/raw/{ticker}_{START_DATE}_{END_DATE}.csv"
        if not os.path.exists(data_file):
            missing_tickers.append(ticker)
    
    if missing_tickers:
        print(f"\n📥 Missing data for: {', '.join(missing_tickers)}")
        print("🔄 Automatic download in progress...")
        
        try:
            # Launch data_loader via subprocess to avoid import issues
            import subprocess
            result = subprocess.run([
                sys.executable, 'src/data_loader.py'
            ], cwd='.', capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Data downloaded successfully!")
                
                # Complete automatic processing for new data
                print("🔄 Automatic processing of new data...")
                
                # Calculate moving averages
                ma_result = subprocess.run([
                    sys.executable, 'run_pipeline.py', '--ma'
                ], cwd='.', capture_output=True, text=True)
                
                if ma_result.returncode == 0:
                    print("✅ Moving averages calculated!")
                    
                    # Generate signals
                    signals_result = subprocess.run([
                        sys.executable, 'run_pipeline.py', '--signals'
                    ], cwd='.', capture_output=True, text=True)
                    
                    if signals_result.returncode == 0:
                        print("✅ Signals generated!")
                        print("🎉 New data completely processed!")
                    else:
                        print("⚠️  Error generating signals, but data downloaded")
                else:
                    print("⚠️  Error calculating MA, but data downloaded")
                    
            else:
                print(f"❌ Error during download: {result.stderr}")
                print("💡 You can download manually with: python src/data_loader.py")
            
        except Exception as e:
            print(f"❌ Error during download: {e}")
            print("💡 You can download manually with: python src/data_loader.py")
    else:
        print("✅ All data is available!")

def update_dates(start_date, end_date):
    """Updates dates in the configuration file."""
    print(f"🔄 Updating dates: {start_date} → {end_date}")
    
    # Read the file
    with open('project_config.py', 'r') as f:
        content = f.read()
    
    # Replace dates
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('START_DATE = ') and not line.strip().startswith('#'):
            lines[i] = f"START_DATE = '{start_date}'"
        elif line.strip().startswith('END_DATE = ') and not line.strip().startswith('#'):
            lines[i] = f"END_DATE = '{end_date}'"
    
    # Save
    with open('project_config.py', 'w') as f:
        f.write('\n'.join(lines))
    
    # Clear cache
    clear_cache()
    print("✅ Dates updated!")

def clear_cache():
    """Clears Python caches to force reload."""
    import os
    import shutil
    import sys
    
    print("🧹 Cleaning Python caches...")
    
    # Delete __pycache__
    cache_dirs = ['__pycache__', 'src/__pycache__']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"   Deleted: {cache_dir}")
    
    # Remove module from loaded modules
    if 'project_config' in sys.modules:
        del sys.modules['project_config']
        print("   Module project_config reloaded")

def manage_config():
    """Interactive interface to manage configuration."""
    import sys
    
    if len(sys.argv) == 1:
        # Interactive mode
        print("\n🎛️  CONFIGURATION MANAGER")
        print("="*50)
        print("1. Display current configuration")
        print("2. Modify tickers")
        print("3. Modify dates")
        print("4. Clear caches")
        print("5. Quit")
        
        while True:
            try:
                choice = input("\nChoose an option (1-5): ").strip()
                
                if choice == '1':
                    print_config()
                elif choice == '2':
                    current_tickers = ', '.join(TICKERS)
                    print(f"Current tickers: {current_tickers}")
                    new_tickers = input("New tickers (separated by commas): ").strip()
                    if new_tickers:
                        tickers = [t.strip().upper() for t in new_tickers.split(',')]
                        update_tickers(tickers)
                elif choice == '3':
                    print(f"Current dates: {START_DATE} → {END_DATE}")
                    start = input("New start date (YYYY-MM-DD): ").strip()
                    end = input("New end date (YYYY-MM-DD): ").strip()
                    if start and end:
                        update_dates(start, end)
                elif choice == '4':
                    clear_cache()
                elif choice == '5':
                    print("Goodbye!")
                    break
                else:
                    print("❌ Invalid option, choose 1-5")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    else:
        # Command line mode
        import argparse
        
        parser = argparse.ArgumentParser(description="Configuration management")
        parser.add_argument('--show', action='store_true', help='Display configuration')
        parser.add_argument('--tickers', type=str, help='New tickers (ex: AAPL,MSFT,SPY)')
        parser.add_argument('--dates', nargs=2, help='New dates (START END)')
        parser.add_argument('--clear', action='store_true', help='Clear caches')
        
        args = parser.parse_args()
        
        if args.show:
            print_config()
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(',')]
            update_tickers(tickers)
        if args.dates:
            update_dates(args.dates[0], args.dates[1])
        if args.clear:
            clear_cache()

# Automatic validation on import
if __name__ == "__main__":
    manage_config()