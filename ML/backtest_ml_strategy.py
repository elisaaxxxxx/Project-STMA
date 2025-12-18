"""
ML Strategy Backtesting
=======================

Backtest the ML-guided trading strategy:
1. Each day, predict returns for all 12 MA pairs
2. Select MA pair with highest predicted return
3. Trade using that pair's signal
4. Measure actual performance

Usage:
    python ML/backtest_ml_strategy.py --ticker AAPL
    python ML/backtest_ml_strategy.py --ticker AAPL --model lasso
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(str(Path(__file__).parent.parent))
import project_config as config

# Use project root directory for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ML_DATA_DIR = PROJECT_ROOT / "data" / "ML"
ML_MODELS_DIR = PROJECT_ROOT / "ML" / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "ML" / "backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Features
GLOBAL_FEATURES = [
    'ret_1d', 'ret_5d', 'ret_20d', 'momentum_1m', 'momentum_3m',
    'vol_20d', 'volume_20d_avg', 'volume_ratio', 'price_over_ma200',
    'spy_ret_5d', 'spy_ret_20d', 'spy_vol_20d', 'spy_ma_ratio_20_50', 'spy_autocorr_1d'
]

MA_SPECIFIC_FEATURES = [
    'ma_short_t', 'ma_long_t', 'ma_diff_t', 'ma_ratio_t', 'signal_t'
]

MA_PARAMETERS = [
    'short_window', 'long_window'
]

ALL_FEATURES = GLOBAL_FEATURES + MA_SPECIFIC_FEATURES + MA_PARAMETERS


def load_ml_model(ticker, model_name='lasso_regression'):
    """Load trained model and scaler."""
    
    scaler_file = ML_MODELS_DIR / f"{ticker}_regression_scaler.pkl"
    model_file = ML_MODELS_DIR / f"{ticker}_regression_{model_name}.pkl"
    
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)
    
    return model, scaler


def load_ml_data(ticker):
    """Load ML dataset and merge with price data for actual returns."""
    # Load ML features
    ml_file = ML_DATA_DIR / f"{ticker}_ml_data.csv"
    df = pd.read_csv(ml_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load price data to calculate actual future returns
    price_file = Path(f"data/SRC/raw/{ticker}_2000-01-01_2025-11-01.csv")
    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Calculate future returns
    price_df['future_ret_1d'] = price_df['Close'].shift(-1) / price_df['Close'] - 1
    
    # Merge
    df = df.merge(price_df[['Date', 'future_ret_1d', 'Close']], on='Date', how='left')
    
    return df.sort_values('Date').reset_index(drop=True)


def backtest_ml_strategy(df, model, scaler, train_end_date, transaction_cost=0.001):
    """
    Backtest ML-guided strategy on test period only.
    
    Strategy:
    1. For each date in test period, predict returns for all 12 MA pairs
    2. Select MA pair with highest predicted return
    3. Trade using that pair's signal (long if signal=1, cash if signal=0)
    4. Track performance
    """
    
    # Only use test period (after training)
    test_df = df[df['Date'] >= train_end_date].copy()
    
    if len(test_df) == 0:
        raise ValueError("No test data available after training period")
    
    print(f"   Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"   Test rows: {len(test_df):,}")
    
    # Get unique dates
    unique_dates = sorted(test_df['Date'].unique())
    
    # Initialize tracking
    results = []
    equity = 1.0  # Start with $1
    position = 0  # 0 = cash, 1 = long
    
    for date in unique_dates:
        # Get all MA pairs for this date
        date_data = test_df[test_df['Date'] == date].copy()
        
        if len(date_data) == 0:
            continue
        
        # Prepare features
        X = date_data[ALL_FEATURES].values
        X_scaled = scaler.transform(X)
        
        # Predict returns for all MA pairs
        predictions = model.predict(X_scaled)
        date_data['predicted_return'] = predictions
        
        # Select MA pair with highest predicted return
        best_idx = predictions.argmax()
        best_row = date_data.iloc[best_idx]
        
        # Get signal and actual return
        signal = int(best_row['signal_t'])
        actual_return = best_row['future_ret_1d']  # Use 1-day return for daily trading
        
        # Previous position
        prev_position = position
        
        # New position based on signal
        position = signal  # 1 = long, 0 = cash
        
        # Calculate strategy return
        if position == 1:  # Long position
            # Apply transaction cost if we just entered
            if prev_position == 0:
                equity *= (1 - transaction_cost)
            # Apply market return
            if not pd.isna(actual_return):
                equity *= (1 + actual_return)
            strategy_return = actual_return if not pd.isna(actual_return) else 0
        else:  # Cash position
            # Apply transaction cost if we just exited
            if prev_position == 1:
                equity *= (1 - transaction_cost)
            strategy_return = 0
        
        # Track results
        results.append({
            'Date': date,
            'selected_ma_short': int(best_row['short_window']),
            'selected_ma_long': int(best_row['long_window']),
            'predicted_return': best_row['predicted_return'],
            'actual_return': actual_return,
            'signal': signal,
            'position': position,
            'equity': equity,
            'strategy_return': strategy_return
        })
    
    return pd.DataFrame(results)


def calculate_metrics(results_df, buy_hold_data):
    """Calculate performance metrics."""
    
    # Strategy metrics
    total_return = (results_df['equity'].iloc[-1] - 1) * 100
    
    # Daily returns
    daily_returns = results_df['strategy_return'].fillna(0)
    
    # Annualized metrics
    n_days = len(results_df)
    years = n_days / 252
    cagr = ((results_df['equity'].iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Volatility
    annual_vol = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # Maximum drawdown
    equity_curve = results_df['equity']
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Win rate
    wins = (daily_returns > 0).sum()
    total_trades = (daily_returns != 0).sum()
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Buy & Hold metrics
    bh_return = (buy_hold_data['Close'].iloc[-1] / buy_hold_data['Close'].iloc[0] - 1) * 100
    bh_cagr = ((buy_hold_data['Close'].iloc[-1] / buy_hold_data['Close'].iloc[0]) ** (1/years) - 1) * 100
    bh_returns = buy_hold_data['Close'].pct_change().fillna(0)
    bh_vol = bh_returns.std() * np.sqrt(252) * 100
    bh_sharpe = (bh_returns.mean() / bh_returns.std() * np.sqrt(252)) if bh_returns.std() > 0 else 0
    
    bh_cummax = buy_hold_data['Close'].cummax()
    bh_drawdown = (buy_hold_data['Close'] - bh_cummax) / bh_cummax * 100
    bh_max_dd = bh_drawdown.min()
    
    metrics = {
        'ML Strategy': {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Total Trades': total_trades
        },
        'Buy & Hold': {
            'Total Return': bh_return,
            'CAGR': bh_cagr,
            'Volatility': bh_vol,
            'Sharpe Ratio': bh_sharpe,
            'Max Drawdown': bh_max_dd
        }
    }
    
    return metrics


def plot_results(results_df, buy_hold_data, ticker, model_name, save_path):
    """Plot equity curve comparison."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Normalize buy & hold to start at 1.0
    bh_normalized = buy_hold_data['Close'] / buy_hold_data['Close'].iloc[0]
    
    # 1. Equity curves
    ax1 = axes[0]
    ax1.plot(results_df['Date'], results_df['equity'], label='ML Strategy', linewidth=2, color='blue')
    ax1.plot(buy_hold_data['Date'], bh_normalized, label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
    ax1.set_ylabel('Portfolio Value ($1 initial)', fontsize=12)
    ax1.set_title(f'{ticker} - ML Strategy vs Buy & Hold (Model: {model_name})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(results_df['Date'].min(), results_df['Date'].max())
    
    # 2. Selected MA pairs over time
    ax2 = axes[1]
    ma_labels = [f"({int(row['selected_ma_short'])},{int(row['selected_ma_long'])})" 
                 for _, row in results_df.iterrows()]
    ma_combinations = results_df['selected_ma_short'].astype(str) + '_' + results_df['selected_ma_long'].astype(str)
    unique_mas = ma_combinations.unique()
    
    # Color map for different MA pairs
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_mas)))
    color_map = dict(zip(unique_mas, colors))
    
    for ma_combo in unique_mas:
        mask = ma_combinations == ma_combo
        dates_subset = results_df[mask]['Date']
        equity_subset = results_df[mask]['equity']
        if len(dates_subset) > 0:
            ax2.scatter(dates_subset, equity_subset, c=[color_map[ma_combo]], 
                       label=ma_combo.replace('_', ','), alpha=0.6, s=20)
    
    ax2.set_ylabel('Equity when MA pair selected', fontsize=12)
    ax2.set_title('MA Pair Selection Over Time', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Position (in market or cash)
    ax3 = axes[2]
    ax3.fill_between(results_df['Date'], 0, results_df['position'], 
                     where=results_df['position']==1, alpha=0.3, color='green', label='Long Position')
    ax3.fill_between(results_df['Date'], 0, 1, 
                     where=results_df['position']==0, alpha=0.3, color='red', label='Cash')
    ax3.set_ylabel('Position', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Position Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved: {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest ML strategy')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker to backtest')
    parser.add_argument('--model', type=str, default='lasso_regression', 
                       help='Model to use (lasso_regression, random_forest, gradient_boosting, etc.)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training ratio (default: 0.7)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🎯 ML STRATEGY BACKTEST: {args.ticker}")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    
    # Load ML data
    print(f"\n📂 Loading ML data...")
    df = load_ml_data(args.ticker)
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Determine train/test split
    unique_dates = sorted(df['Date'].unique())
    split_idx = int(len(unique_dates) * args.train_ratio)
    train_end_date = unique_dates[split_idx]
    
    print(f"\n📅 Data Split:")
    print(f"   Training end: {train_end_date.date()}")
    print(f"   Testing starts: {unique_dates[split_idx].date()}")
    
    # Load model
    print(f"\n🤖 Loading trained model...")
    try:
        model, scaler = load_ml_model(args.ticker, args.model)
        print(f"   ✓ Model loaded: {args.model}")
        print(f"   ✓ Scaler loaded")
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        print(f"\n   Please train the model first:")
        print(f"   python ML/train_regression_model.py --ticker {args.ticker}")
        return
    
    # Run backtest
    print(f"\n🎯 Running backtest on test period...")
    results_df = backtest_ml_strategy(df, model, scaler, train_end_date)
    
    print(f"   ✓ Backtest complete: {len(results_df):,} trading days")
    
    # Load buy & hold data for comparison
    print(f"\n📊 Loading buy & hold comparison data...")
    price_file = Path(f"data/SRC/raw/{args.ticker}_2000-01-01_2025-11-01.csv")
    buy_hold_data = pd.read_csv(price_file)
    buy_hold_data['Date'] = pd.to_datetime(buy_hold_data['Date'])
    buy_hold_data = buy_hold_data[buy_hold_data['Date'] >= train_end_date].reset_index(drop=True)
    
    # Calculate metrics
    print(f"\n📈 Calculating performance metrics...")
    metrics = calculate_metrics(results_df, buy_hold_data)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST RESULTS")
    print(f"{'='*80}")
    
    print(f"\n🤖 ML STRATEGY ({args.model}):")
    print(f"   {'─'*70}")
    for metric, value in metrics['ML Strategy'].items():
        if 'Rate' in metric or 'Trades' in metric:
            print(f"   {metric:20s}: {value:8.0f}")
        else:
            print(f"   {metric:20s}: {value:8.2f}%")
    
    print(f"\n📊 BUY & HOLD:")
    print(f"   {'─'*70}")
    for metric, value in metrics['Buy & Hold'].items():
        print(f"   {metric:20s}: {value:8.2f}%")
    
    print(f"\n{'='*80}")
    print(f"💡 COMPARISON")
    print(f"{'='*80}")
    
    outperformance = metrics['ML Strategy']['CAGR'] - metrics['Buy & Hold']['CAGR']
    print(f"   CAGR Difference:    {outperformance:8.2f}% {'🟢' if outperformance > 0 else '🔴'}")
    
    sharpe_diff = metrics['ML Strategy']['Sharpe Ratio'] - metrics['Buy & Hold']['Sharpe Ratio']
    print(f"   Sharpe Difference:  {sharpe_diff:8.2f}  {'🟢' if sharpe_diff > 0 else '🔴'}")
    
    dd_diff = metrics['ML Strategy']['Max Drawdown'] - metrics['Buy & Hold']['Max Drawdown']
    print(f"   DrawDown Difference: {dd_diff:8.2f}% {'🟢' if dd_diff > 0 else '🔴'} (closer to 0 is better)")
    
    # Save results
    print(f"\n💾 Saving results...")
    results_file = RESULTS_DIR / f"{args.ticker}_{args.model}_backtest_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"   ✓ Results saved: {results_file}")
    
    # Plot
    print(f"\n📈 Creating plots...")
    plot_file = RESULTS_DIR / f"{args.ticker}_{args.model}_backtest_plot.png"
    plot_results(results_df, buy_hold_data, args.ticker, args.model, plot_file)
    
    print(f"\n{'='*80}")
    print(f"✅ BACKTEST COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  • {results_file.name}")
    print(f"  • {plot_file.name}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
