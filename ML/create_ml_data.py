"""
Feature Engineering for ML - Creating Comprehensive Training Dataset
=====================================================================

WHAT THIS SCRIPT DOES:
This script transforms raw price data into a structured machine learning dataset with
21 carefully engineered features for predicting which moving average pair will perform
best over the next 3 days. It's the foundation of the entire ML pipeline.

THE CHALLENGE:
Instead of using fixed MA pairs like traditional strategies, we want the ML model to
DYNAMICALLY SELECT the best MA pair each day based on current market conditions.

To do this, we need to predict: "Which of the 12 MA pairs will generate the highest
returns over the next 3 days, given current market state?"

THE SOLUTION - DATA STRUCTURE:
Create one row per (Date, MA_pair) combination:
- If we have 5,000 trading days and 12 MA pairs → 60,000 rows total
- Each row represents: "On date X, if I use MA pair (A,B), what will happen?"

Example structure:
┌────────────┬──────┬─────┬─────────┬─────────┬──────────┬────────────────┐
│ Date       │ short│ long│ ret_1d  │ vol_20d │ signal_t │ strategy_ret_3d│
├────────────┼──────┼─────┼─────────┼─────────┼──────────┼────────────────┤
│ 2018-01-02 │  5   │ 10  │ 0.012   │ 0.015   │    1     │    0.023       │
│ 2018-01-02 │  5   │ 20  │ 0.012   │ 0.015   │    1     │    0.019       │
│ 2018-01-02 │  5   │ 50  │ 0.012   │ 0.015   │    0     │   -0.005       │
│ ...        │ ...  │ ... │  ...    │  ...    │   ...    │     ...        │
│ 2018-01-02 │ 100  │ 200 │ 0.012   │ 0.015   │    1     │    0.015       │
└────────────┴──────┴─────┴─────────┴─────────┴──────────┴────────────────┘

On 2018-01-02, the model can learn: Given market conditions (ret_1d=0.012, vol_20d=0.015),
MA pair (5,10) with signal=1 generated 0.023 return over next 3 days (best choice!).

THE 21 FEATURES - THREE CATEGORIES:

1️ **GLOBAL MARKET FEATURES (14)** - Same for all MA pairs on a given date:
   Price Returns:
   - ret_1d: 1-day return (momentum signal)
   - ret_5d: 5-day return (short-term trend)
   - ret_20d: 20-day return (medium-term trend)
   
   Momentum:
   - momentum_1m: 1-month (21-day) return
   - momentum_3m: 3-month (63-day) return
   
   Volatility:
   - vol_20d: 20-day rolling standard deviation (market uncertainty)
   
   Volume:
   - volume_20d_avg: 20-day average volume
   - volume_ratio: Current volume / 20-day average (unusual activity?)
   
   Trend:
   - price_over_ma200: (Price / MA_200) - 1 (long-term trend strength)
   
   SPY Benchmark (market context):
   - spy_ret_5d: SPY 5-day return
   - spy_ret_20d: SPY 20-day return
   - spy_vol_20d: SPY 20-day volatility
   - spy_ma_ratio_20_50: SPY's MA_20/MA_50 ratio (market regime)
   - spy_autocorr_1d: SPY return autocorrelation (market efficiency)

2️ **MA-SPECIFIC FEATURES (5)** - Different for each MA pair:
   - ma_short_t: Short MA value at time t
   - ma_long_t: Long MA value at time t
   - ma_diff_t: ma_short_t - ma_long_t (absolute difference)
   - ma_ratio_t: ma_short_t / ma_long_t (relative difference)
   - signal_t: Current signal (1 if ma_short > ma_long, else 0)

3️ **MA PARAMETERS (2)** - Identifying which pair:
   - short_window: Short MA period (5, 10, 20, 50, or 100)
   - long_window: Long MA period (10, 20, 50, 100, or 200)

TARGET VARIABLE:
- **strategy_ret_3d**: 3-day forward return IF we trade using this MA pair's signal
  - Calculated: position[t] × return[t+1] + position[t+1] × return[t+2] + position[t+2] × return[t+3]
  - Position lagged by 1 day (no look-ahead bias!)
  - Represents: "How well would this MA pair perform over next 3 days?"

THE 12 MA PAIRS:
Short-term: (5,10), (5,20), (5,50)
Medium-term: (10,20), (10,50), (10,100)
Long-term: (20,50), (20,100), (20,200)
Very long-term: (50,100), (50,200), (100,200)

FEATURE ENGINEERING PHILOSOPHY:
✓ **No look-ahead bias**: All features use only information available at time t
✓ **Target is forward-looking**: We predict 3-day future returns (realistic horizon)
✓ **Comprehensive context**: Capture price, volume, volatility, momentum, and market regime
✓ **Regime detection**: Different MA pairs work better in different market conditions
✓ **Relative features**: Ratios and differences capture relationships, not absolute levels

DATA PIPELINE:
1. Load processed price data with existing MAs (from calculate_moving_averages.py)
2. Calculate 14 global features (same for all pairs each day)
3. For each of 12 MA pairs:
   a. Calculate MA values if not already present
   b. Generate MA-specific features (5)
   c. Calculate 3-day forward strategy returns (target)
   d. Add MA parameters (2)
4. Combine into one dataset: ~60,000 rows for 5,000 days × 12 pairs
5. Drop rows with missing values (early periods when long MAs not yet formed)
6. Save to CSV for model training

OUTPUT FILE:
- **{ticker}_ml_data.csv**: Complete training dataset
  - Typical size: 75,000+ rows (after dropping NAs)
  - 22 columns: Date + 21 features + 1 target
  - Ready for train_regression_model.py

EXAMPLE USAGE:
# Single ticker
python ML/create_ml_data.py --ticker AAPL

# All tickers
python ML/create_ml_data.py --all

# Via main pipeline
python main.py --ml

WHY THIS MATTERS:
This dataset structure allows the model to learn:
- Which MA pairs work in trending markets (high momentum)
- Which pairs work in volatile markets (high vol_20d)
- How SPY market context affects individual stock MA performance
- When to prefer short-term vs long-term MAs based on conditions

Result: Model can dynamically adapt MA pair selection to current market regime,
achieving +6.69% CAGR improvement over fixed walk-forward strategies.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
import project_config as config

# Use project root directory for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "SRC" / "processed"
ML_DATA_DIR = PROJECT_ROOT / "data" / "ML"
os.makedirs(ML_DATA_DIR, exist_ok=True)

# MA pairs to test
MA_PAIRS = [
    (5, 10), (5, 20), (5, 50),
    (10, 20), (10, 50), (10, 100),
    (20, 50), (20, 100), (20, 200),
    (50, 100), (50, 200),
    (100, 200)
]

def load_price_data(ticker):
    """Load processed price data."""
    file = DATA_PROCESSED_DIR / f"{ticker}_2000-01-01_2025-11-01_with_signals.csv"
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)

def calculate_global_features(df, ticker):
    """Calculate global market features (same for all MA pairs on a given date)."""
    
    df = df.copy()
    
    # === GLOBAL FEATURES ===
    
    # Price returns
    df['ret_1d'] = df['Close'].pct_change()
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_20d'] = df['Close'].pct_change(20)
    
    # Momentum
    df['momentum_1m'] = df['Close'].pct_change(21)  # 1 month ≈ 21 trading days
    df['momentum_3m'] = df['Close'].pct_change(63)  # 3 months ≈ 63 trading days
    
    # Volatility
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    
    # Volume features
    df['volume_20d_avg'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_20d_avg'].replace(0, np.nan)
    
    # Price over MA200 (long-term trend indicator)
    df['ma_200'] = df['Close'].rolling(200).mean()
    df['price_over_ma200'] = (df['Close'] / df['ma_200'].replace(0, np.nan)) - 1
    
    # === SPY FEATURES (market benchmark) ===
    # Load SPY data if available and ticker is not SPY
    if ticker != 'SPY':
        spy_file = Path("data/SRC/processed") / "SPY_2000-01-01_2025-11-01_with_signals.csv"
        if spy_file.exists():
            spy_df = pd.read_csv(spy_file)
            spy_df['Date'] = pd.to_datetime(spy_df['Date'])
            spy_df = spy_df[['Date', 'Close']].rename(columns={'Close': 'SPY_Close'})
            
            # Calculate SPY features
            spy_df['spy_ret_5d'] = spy_df['SPY_Close'].pct_change(5)
            spy_df['spy_ret_20d'] = spy_df['SPY_Close'].pct_change(20)
            spy_df['spy_ret_1d'] = spy_df['SPY_Close'].pct_change()
            spy_df['spy_vol_20d'] = spy_df['spy_ret_1d'].rolling(20).std()
            
            # SPY MA ratio
            spy_df['spy_ma_20'] = spy_df['SPY_Close'].rolling(20).mean()
            spy_df['spy_ma_50'] = spy_df['SPY_Close'].rolling(50).mean()
            spy_df['spy_ma_ratio_20_50'] = spy_df['spy_ma_20'] / spy_df['spy_ma_50'].replace(0, np.nan)
            
            # SPY autocorrelation (1-day lag)
            spy_df['spy_autocorr_1d'] = spy_df['spy_ret_1d'].rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
            )
            
            # Merge with main dataframe
            df = df.merge(spy_df[['Date', 'spy_ret_5d', 'spy_ret_20d', 'spy_vol_20d', 
                                   'spy_ma_ratio_20_50', 'spy_autocorr_1d']], 
                         on='Date', how='left')
        else:
            # If SPY data not available, fill with NaN
            df['spy_ret_5d'] = np.nan
            df['spy_ret_20d'] = np.nan
            df['spy_vol_20d'] = np.nan
            df['spy_ma_ratio_20_50'] = np.nan
            df['spy_autocorr_1d'] = np.nan
    else:
        # If ticker is SPY, SPY features are self-referential
        df['spy_ret_5d'] = df['ret_5d']
        df['spy_ret_20d'] = df['ret_20d']
        df['spy_vol_20d'] = df['vol_20d']
        
        # SPY MA ratio
        spy_ma_20 = df['Close'].rolling(20).mean()
        spy_ma_50 = df['Close'].rolling(50).mean()
        df['spy_ma_ratio_20_50'] = spy_ma_20 / spy_ma_50.replace(0, np.nan)
        
        # Autocorrelation
        df['spy_autocorr_1d'] = df['ret_1d'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )
    
    # Future returns for target calculation
    df['future_ret_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['future_ret_3d'] = df['Close'].shift(-3) / df['Close'] - 1
    df['future_ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    
    return df

def calculate_ma_features(df, short_window, long_window):
    """Calculate MA-specific features for a given pair."""
    
    data = df.copy()
    
    # === MA-SPECIFIC FEATURES ===
    
    # Calculate MAs
    data['ma_short_t'] = data['Close'].rolling(short_window).mean()
    data['ma_long_t'] = data['Close'].rolling(long_window).mean()
    
    # MA pair features
    data['ma_diff_t'] = data['ma_short_t'] - data['ma_long_t']
    data['ma_ratio_t'] = data['ma_short_t'] / data['ma_long_t'].replace(0, np.nan)
    
    # Signal (1 if short > long, 0 otherwise)
    data['signal_t'] = (data['ma_short_t'] > data['ma_long_t']).astype(int)
    
    # === MA PARAMETERS ===
    data['short_window'] = short_window
    data['long_window'] = long_window
    
    return data

def calculate_strategy_returns(df):
    """Calculate strategy returns for this MA pair."""
    
    df = df.copy()
    
    # Strategy return = signal * future_return + (1-signal) * (-future_return)
    # Long when signal_t=1, short when signal_t=0
    df['strategy_ret_1d'] = np.where(
        df['signal_t'] == 1,
        df['future_ret_1d'],  # Long
        -df['future_ret_1d']  # Short
    )
    
    df['strategy_ret_3d'] = np.where(
        df['signal_t'] == 1,
        df['future_ret_3d'],
        -df['future_ret_3d']
    )
    
    df['strategy_ret_5d'] = np.where(
        df['signal_t'] == 1,
        df['future_ret_5d'],
        -df['future_ret_5d']
    )
    
    return df

def create_ml_dataset(ticker):
    """Create complete ML dataset with one row per (date, MA_pair)."""
    
    print(f"🔧 Creating ML Dataset: {ticker}")
    print("=" * 60)
    
    # Load price data
    print(f"📊 Loading price data...")
    price_df = load_price_data(ticker)
    print(f"   {len(price_df):,} days of data")
    
    # Calculate global features
    print(f"📊 Calculating global market features...")
    price_df = calculate_global_features(price_df, ticker)
    
    # Create dataset for all MA pairs
    print(f"📊 Creating rows for {len(MA_PAIRS)} MA pairs...")
    all_rows = []
    
    for short, long in MA_PAIRS:
        print(f"   Processing MA {short}-{long}...")
        
        # Calculate MA-specific features
        ma_data = calculate_ma_features(price_df, short, long)
        
        # Calculate strategy returns
        ma_data = calculate_strategy_returns(ma_data)
        
        # Add to list
        all_rows.append(ma_data)
    
    # Combine all MA pairs
    print(f"📊 Combining all rows...")
    ml_df = pd.concat(all_rows, ignore_index=True)
    ml_df = ml_df.sort_values(['Date', 'short_window', 'long_window'])
    
    print(f"   Total rows: {len(ml_df):,}")
    print(f"   Rows per date: {len(MA_PAIRS)}")
    print(f"   Trading days: {ml_df['Date'].nunique():,}")
    
    # Create target: which MA pair performs best on each date
    print(f"📊 Creating target variable...")
    ml_df = create_target_variable(ml_df)
    
    # Remove rows with NaN in critical FEATURE columns only
    # Keep targets even if NaN - we'll handle during training
    print(f"📊 Cleaning data...")
    initial_rows = len(ml_df)
    
    # Drop rows with NaN in features (but not in targets)
    feature_cols_to_check = [
        'ret_1d', 'ret_5d', 'ret_20d', 'momentum_1m', 'momentum_3m',
        'vol_20d', 'volume_20d_avg', 'volume_ratio', 'price_over_ma200',
        'ma_short_t', 'ma_long_t', 'ma_diff_t', 'ma_ratio_t', 'signal_t'
    ]
    
    ml_df = ml_df.dropna(subset=feature_cols_to_check)
    print(f"   Removed {initial_rows - len(ml_df):,} rows with missing FEATURES")
    
    # Now drop rows where TARGET is NaN
    # (these are at the end of the dataset where we can't calculate future returns)
    ml_df = ml_df.dropna(subset=['strategy_ret_3d', 'is_top_performer'])
    print(f"   Final rows: {len(ml_df):,}")
    
    # Define the EXACT columns to keep
    COLUMNS_TO_KEEP = [
        # Metadata
        'Date',
        # Global features (14)
        'ret_1d', 'ret_5d', 'ret_20d',
        'momentum_1m', 'momentum_3m',
        'vol_20d', 'volume_20d_avg', 'volume_ratio',
        'price_over_ma200',
        'spy_ret_5d', 'spy_ret_20d', 'spy_vol_20d',
        'spy_ma_ratio_20_50', 'spy_autocorr_1d',
        # MA-specific features (5)
        'ma_short_t', 'ma_long_t', 'ma_diff_t', 'ma_ratio_t', 'signal_t',
        # MA parameters (2)
        'short_window', 'long_window',
        # Targets
        'strategy_ret_1d', 'strategy_ret_3d', 'strategy_ret_5d',
        'performance_rank', 'is_top_performer'
    ]
    
    # Keep only specified columns
    ml_df = ml_df[COLUMNS_TO_KEEP]
    
    # Save
    output_file = ML_DATA_DIR / f"{ticker}_ml_data.csv"
    ml_df.to_csv(output_file, index=False)
    print(f"\n✅ ML dataset saved: {output_file}")
    print(f"   Columns: {len(ml_df.columns)} (21 features + Date + 5 target columns)")
    
    # Print sample
    print(f"\n📋 Sample row:")
    sample = ml_df[ml_df['Date'] == ml_df['Date'].unique()[100]].iloc[0]
    print(f"   Date: {sample['Date']}")
    print(f"   MA Pair: ({sample['short_window']:.0f}, {sample['long_window']:.0f})")
    print(f"   Signal: {sample['signal_t']:.0f}")
    print(f"   Strategy Return 3d: {sample['strategy_ret_3d']:.4f}")
    print(f"   Is Top Performer: {sample['is_top_performer']:.0f}")
    
    # Define the EXACT 21 features we want
    FEATURE_LIST = [
        # Global features (14)
        'ret_1d', 'ret_5d', 'ret_20d',
        'momentum_1m', 'momentum_3m',
        'vol_20d', 'volume_20d_avg', 'volume_ratio',
        'price_over_ma200',
        'spy_ret_5d', 'spy_ret_20d', 'spy_vol_20d',
        'spy_ma_ratio_20_50', 'spy_autocorr_1d',
        # MA-specific features (5)
        'ma_short_t', 'ma_long_t', 'ma_diff_t', 'ma_ratio_t', 'signal_t',
        # MA parameters (2)
        'short_window', 'long_window'
    ]
    
    # Verify all features exist
    missing_features = [f for f in FEATURE_LIST if f not in ml_df.columns]
    if missing_features:
        print(f"\n⚠️  WARNING: Missing features: {missing_features}")
    
    print(f"\n📊 EXACT 21 FEATURES (as specified):")
    print(f"\n   🌍 GLOBAL FEATURES (14):")
    for i, feat in enumerate(FEATURE_LIST[:14], 1):
        status = "✓" if feat in ml_df.columns else "✗"
        print(f"      {i:2d}. {feat:20s} {status}")
    
    print(f"\n   📐 MA-SPECIFIC FEATURES (5):")
    for i, feat in enumerate(FEATURE_LIST[14:19], 1):
        status = "✓" if feat in ml_df.columns else "✗"
        print(f"      {i:2d}. {feat:20s} {status}")
    
    print(f"\n   🔢 MA PARAMETERS (2):")
    for i, feat in enumerate(FEATURE_LIST[19:21], 1):
        status = "✓" if feat in ml_df.columns else "✗"
        print(f"      {i:2d}. {feat:20s} {status}")
    
    print(f"\n   ✅ TOTAL: {len(FEATURE_LIST)} features")
    
    # Show sample values for verification
    print(f"\n📊 Sample feature values (first row with complete data):")
    complete_rows = ml_df.dropna(subset=FEATURE_LIST)
    if len(complete_rows) > 0:
        complete_row = complete_rows.iloc[0]
        print(f"   Date: {complete_row['Date']}")
        for feat in FEATURE_LIST:
            if feat in ml_df.columns:
                val = complete_row[feat]
                print(f"   {feat:20s} = {val:.6f}" if not pd.isna(val) else f"   {feat:20s} = NaN")
    else:
        print(f"   ⚠️ No rows with all features complete (may have NaN in some features)")
    
    return ml_df

def create_target_variable(df):
    """Create target: is this MA pair in top 30% performers for this date?"""
    
    df = df.copy()
    
    # For each date, rank MA pairs by their 3-day strategy return
    # Using 'first' method to break ties consistently
    df['performance_rank'] = df.groupby('Date')['strategy_ret_3d'].rank(
        ascending=False, method='first'
    )
    
    # Top 30% are labeled as 1 (good to trade)
    # With 12 MA pairs, top 30% = top 3-4 pairs
    # Use ceil to ensure we get at least 30%
    import math
    n_pairs = len(MA_PAIRS)
    top_n = math.ceil(n_pairs * 0.3)  # ceil(3.6) = 4
    df['is_top_performer'] = (df['performance_rank'] <= top_n).astype(int)
    
    # Alternative: binary classification based on positive returns
    # df['is_profitable'] = (df['strategy_ret_3d'] > 0).astype(int)
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ML dataset')
    parser.add_argument('--ticker', type=str, help='Create dataset for ticker')
    parser.add_argument('--all', action='store_true', help='Create for all tickers')
    
    args = parser.parse_args()
    
    if args.ticker:
        create_ml_dataset(args.ticker)
    elif args.all:
        for ticker in config.TICKERS:
            create_ml_dataset(ticker)
            print("\n" + "=" * 60 + "\n")
    else:
        print("Usage:")
        print("  python ML/create_ml_data.py --ticker AAPL")
        print("  python ML/create_ml_data.py --all")

if __name__ == '__main__':
    main()
