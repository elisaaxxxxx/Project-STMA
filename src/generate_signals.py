"""
Generate Trading Signals from Moving Averages
==============================================

This script uses centralized configuration to:
1. Read files with moving averages
2. Generate signals according to configured comparisons
3. Create a combined signal and a buy signal
4. Save to configured directory

Parameters are defined in project_config.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import MA_COMPARISONS, DATA_PROCESSED_DIR


def generate_signals(input_file, output_folder=None):
    """
    Generates trading signals according to configuration.
    
    Args:
        input_file: Path to CSV file with moving averages
        output_folder: Output directory (uses STRATEGY_DIR if None)
    """
    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR
    
    # Read the CSV file with moving averages
    print(f"Reading {input_file.name}...")
    df = pd.read_csv(input_file)
    
    # Use configured comparisons
    # Convert dict format to tuple for compatibility
    comparisons = [(comp['short'], comp['long'], comp['name']) for comp in MA_COMPARISONS]
    
    # Generate signals for each comparison
    signal_columns = []
    for short, long, signal_name in comparisons:
        short_col = f"MA_{short}"
        long_col = f"MA_{long}"
        
        # Check if columns exist
        if short_col not in df.columns or long_col not in df.columns:
            print(f"  ⚠️  Skipping {signal_name}: Missing {short_col} or {long_col}")
            continue
        
        # Create signal: 1 if short MA > long MA (bullish), 0 if short MA <= long MA (bearish)
        # Keep NA when either MA value is missing
        condition = (df[short_col] > df[long_col]).astype(int)
        # Set to NA where either value is NaN, using Int64 to support NA values with integers
        df[signal_name] = condition.where(df[short_col].notna() & df[long_col].notna(), other=pd.NA).astype('Int64')
        signal_columns.append(signal_name)
        print(f"  ✓ Generated {signal_name}")
    
    # Create combined signal string (e.g., "1010" for signals 1,0,1,0)
    if signal_columns:
        # Represent each signal as '1', '0' or 'NA' in the combined string
        def _sig_repr(v):
            if pd.isna(v):
                return 'NA'
            return str(int(v))

        df['Combined_Signal'] = df[signal_columns].apply(
            lambda row: ''.join([_sig_repr(v) for v in row]),
            axis=1
        )
        print(f"  ✓ Generated Combined_Signal")

        # Compute Buy signal:
        # - Buy = 1 if count of available signals equal to 1 is >= 2
        # - Buy = 0 if all signals are present and count of ones < 2
        # - Buy = NA otherwise (not enough data)
        total_signals = len(signal_columns)
        ones_count = df[signal_columns].fillna(0).sum(axis=1).astype(int)
        available_count = df[signal_columns].notna().sum(axis=1).astype(int)

        df['Buy'] = np.where(
            ones_count >= 2,
            1,
            np.where(available_count == total_signals, 0, pd.NA)
        )
        # Make Buy an integer nullable column
        df['Buy'] = df['Buy'].astype('Int64')
        print(f"  ✓ Generated Buy signal (1=buy,0=no,NA=unknown)")
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the result
    # Remove '_with_MAs' if it exists and add '_signals'
    base_name = input_file.stem.replace('_with_MAs', '')
    output_file = output_path / f"{base_name}_with_signals.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}\n")
    
    # Show signal summary
    print("  Signal Summary:")
    for short, long, signal_name in comparisons:
        if signal_name in df.columns:
            bullish = df[signal_name].sum()
            bearish = (df[signal_name] == 0).sum()
            total = len(df[signal_name].dropna())
            print(f"    {signal_name}: {bullish} bullish / {bearish} bearish (of {total} total)")
    
    # Show combined signal distribution
    if 'Combined_Signal' in df.columns:
        print(f"\n  Combined Signal Distribution:")
        signal_counts = df['Combined_Signal'].value_counts().sort_index()
        for signal, count in signal_counts.items():
            print(f"    {signal}: {count} occurrences")
    print()
    
    return output_file


def main():
    """Processes all files with moving averages according to configuration."""
    from project_config import print_config, validate_config
    
    # Configuration validation
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    print_config()
    
    # Input folder containing the CSV files with moving averages
    input_folder = Path(DATA_PROCESSED_DIR)
    
    # Find all CSV files with moving averages
    csv_files = list(input_folder.glob("*_with_MAs.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files with moving averages found in {input_folder}")
        print("   Run calculate_moving_averages.py first!")
        return
    
    print(f"Found {len(csv_files)} file(s) to process\n")
    print("=" * 60)
    
    # Process each file
    for csv_file in csv_files:
        generate_signals(csv_file)
    
    print("=" * 60)
    print("✅ All done! Check the MA_strategy/ folder for files with _signals.csv")


if __name__ == "__main__":
    main()
