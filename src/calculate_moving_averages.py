"""
Calculate Moving Averages for Technical Analysis

PURPOSE:
This script computes Simple Moving Averages (SMA) for stock price data.
Moving averages smooth out price data to identify trends and are commonly used
in technical trading strategies (e.g., crossover strategies, support/resistance).

KEY CONCEPTS:
- Simple Moving Average (SMA): Average closing price over a specified window
  Example: 50-day MA = average of last 50 closing prices
- Multiple Windows: Script calculates several MAs simultaneously (e.g., 20, 50, 200 days)
- Rolling Calculation: MA is recalculated daily as new data arrives

TECHNICAL DETAILS:
The formula for SMA is:
    SMA(n) = (Price₁ + Price₂ + ... + Priceₙ) / n
Where n is the window period (e.g., 20 days, 50 days, 200 days)

WORKFLOW:
1. Load raw price data from CSV (Date, Open, High, Low, Close, Volume)
2. For each configured period (e.g., [20, 50, 200]):
   - Calculate rolling mean of closing prices
   - Store as new column (e.g., 'MA_20', 'MA_50', 'MA_200')
3. Save enhanced DataFrame to processed data directory

COMMON MOVING AVERAGE PERIODS:
- Short-term: 10, 20 days (reacts quickly to price changes)
- Medium-term: 50 days (balance between responsiveness and stability)
- Long-term: 100, 200 days (identifies major trends)

INPUT:
Raw CSV files from data/raw/ with columns: Date, Open, High, Low, Close, Volume

OUTPUT:
Enhanced CSV files in data/processed/ with additional MA columns:
Date, Open, High, Low, Close, Volume, MA_20, MA_50, MA_200 (example)

CONFIGURATION:
- MA_PERIODS: List of periods defined in project_config.py
- DATA_PROCESSED_DIR: Output directory path from configuration

USAGE:
Can be run standalone or imported as a module:
    python calculate_moving_averages.py
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import MA_PERIODS, DATA_PROCESSED_DIR, get_ma_file_path


def calculate_moving_averages(input_file, output_folder=None):
    """
    Calculates moving averages according to configuration.
    
    Args:
        input_file: Path to CSV file with data
        output_folder: Output directory (uses STRATEGY_DIR if None)
    """
    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR
    
    # Read the CSV file
    print(f"Reading {input_file.name}...")
    df = pd.read_csv(input_file)
    
    # Use configured MA periods
    windows = MA_PERIODS
    
    # Calculate moving average for each window
    for window in windows:
        column_name = f"MA_{window}"
        # Calculate: for each day, take that day's Close + previous (window-1) days
        # min_periods=window means we only show the average when we have enough data
        df[column_name] = df['Close'].rolling(window=window, min_periods=window).mean()
        print(f"  ✓ Calculated {column_name}")
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the result
    output_file = output_path / f"{input_file.stem}_with_MAs.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}\n")
    
    return output_file


def main():
    """Processes all CSV files according to configuration."""
    from project_config import DATA_RAW_DIR, print_config, validate_config
    
    # Configuration validation
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    print_config()
    
    # Input folder containing the stock data
    input_folder = Path(DATA_RAW_DIR)
    
    # Find all CSV files
    csv_files = list(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    print("=" * 60)
    
    # Process each file
    for csv_file in csv_files:
        calculate_moving_averages(csv_file)
    
    print("=" * 60)
    print("✅ All done! Check the MA_strategy/ folder for results.")


if __name__ == "__main__":
    main()
