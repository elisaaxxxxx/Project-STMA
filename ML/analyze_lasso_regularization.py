"""
Lasso Regularization Analysis - Bias-Variance Tradeoff
========================================================

Visualize how Lasso's alpha (regularization strength) affects:
- Test R² score
- Number of non-zero coefficients
- Model complexity

This shows the bias-variance tradeoff directly on your data.

Usage:
    python ML/analyze_lasso_regularization.py --ticker AAPL
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
import project_config as config

# Use project root directory for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ML_DATA_DIR = PROJECT_ROOT / "data" / "ML"
RESULTS_DIR = PROJECT_ROOT / "data" / "ML" / "regularization_analysis"
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
TARGET = 'strategy_ret_3d'


def load_and_split_data(ticker, train_ratio=0.7):
    """Load ML dataset and split chronologically."""
    
    file = ML_DATA_DIR / f"{ticker}_ml_data.csv"
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Split chronologically
    unique_dates = sorted(df['Date'].unique())
    split_idx = int(len(unique_dates) * train_ratio)
    split_date = unique_dates[split_idx]
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare features and target."""
    
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df[TARGET].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def analyze_regularization(X_train, X_test, y_train, y_test, alphas):
    """Train Lasso models with different alpha values and track performance."""
    
    results = []
    
    print(f"\n   Testing {len(alphas)} alpha values...")
    print(f"   Alpha range: {alphas.min():.2e} to {alphas.max():.2e}")
    
    for i, alpha in enumerate(alphas):
        # Train model
        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Count non-zero coefficients (model complexity)
        n_nonzero = np.sum(model.coef_ != 0)
        
        results.append({
            'alpha': alpha,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'n_nonzero_coefs': n_nonzero,
            'intercept': model.intercept_
        })
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(alphas):
            print(f"   Progress: {i+1}/{len(alphas)} completed", end='\r')
    
    print()  # New line after progress
    
    return pd.DataFrame(results)


def plot_regularization_analysis(results_df, ticker, save_path):
    """Create comprehensive regularization analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. R² vs Alpha (LOG SCALE) - Main plot showing bias-variance tradeoff
    ax1 = axes[0, 0]
    ax1.semilogx(results_df['alpha'], results_df['train_r2'], 
                 label='Training R²', linewidth=2, color='blue', marker='o', markersize=4)
    ax1.semilogx(results_df['alpha'], results_df['test_r2'], 
                 label='Test R²', linewidth=2, color='red', marker='s', markersize=4)
    
    # Mark best test R²
    best_idx = results_df['test_r2'].idxmax()
    best_alpha = results_df.loc[best_idx, 'alpha']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    
    ax1.axvline(best_alpha, color='green', linestyle='--', alpha=0.5, 
                label=f'Best α={best_alpha:.2e}')
    ax1.plot(best_alpha, best_r2, 'g*', markersize=20, 
             label=f'Best Test R²={best_r2:.4f}')
    
    ax1.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Bias-Variance Tradeoff: R² vs Regularization', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Auto-adjust y-axis to remove excess white space
    y_min = min(results_df['test_r2'].min(), results_df['train_r2'].min())
    y_max = max(results_df['test_r2'].max(), results_df['train_r2'].max())
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Add annotations
    ax1.annotate('High Variance\n(Overfitting)', 
                xy=(results_df['alpha'].min(), results_df['train_r2'].iloc[0]),
                xytext=(results_df['alpha'].min()*10, results_df['train_r2'].iloc[0]+0.05),
                fontsize=9, color='blue', style='italic',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    
    ax1.annotate('High Bias\n(Underfitting)', 
                xy=(results_df['alpha'].max(), results_df['test_r2'].iloc[-1]),
                xytext=(results_df['alpha'].max()/10, results_df['test_r2'].iloc[-1]+0.05),
                fontsize=9, color='red', style='italic',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    
    # 2. Number of Features vs Alpha
    ax2 = axes[0, 1]
    ax2.semilogx(results_df['alpha'], results_df['n_nonzero_coefs'], 
                 linewidth=2, color='purple', marker='d', markersize=4)
    ax2.axvline(best_alpha, color='green', linestyle='--', alpha=0.5)
    
    # Mark best model's feature count
    best_n_features = results_df.loc[best_idx, 'n_nonzero_coefs']
    ax2.plot(best_alpha, best_n_features, 'g*', markersize=20,
             label=f'Best: {int(best_n_features)} features')
    
    ax2.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Non-Zero Coefficients', fontsize=12, fontweight='bold')
    ax2.set_title('Model Complexity vs Regularization', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # 3. RMSE vs Alpha
    ax3 = axes[1, 0]
    ax3.semilogx(results_df['alpha'], results_df['train_rmse'], 
                 label='Training RMSE', linewidth=2, color='blue', marker='o', markersize=4)
    ax3.semilogx(results_df['alpha'], results_df['test_rmse'], 
                 label='Test RMSE', linewidth=2, color='red', marker='s', markersize=4)
    ax3.axvline(best_alpha, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Error vs Regularization', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Overfitting Gap (Train R² - Test R²)
    ax4 = axes[1, 1]
    gap = results_df['train_r2'] - results_df['test_r2']
    ax4.semilogx(results_df['alpha'], gap, 
                 linewidth=2, color='orange', marker='^', markersize=4)
    ax4.axvline(best_alpha, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Mark best model's gap
    best_gap = gap.iloc[best_idx]
    ax4.plot(best_alpha, best_gap, 'g*', markersize=20,
             label=f'Best gap: {best_gap:.4f}')
    
    ax4.set_xlabel('Regularization Strength (α)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Train R² - Test R² (Overfitting Gap)', fontsize=12, fontweight='bold')
    ax4.set_title('Generalization Gap vs Regularization', 
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Fill region where gap > 0 (overfitting)
    ax4.fill_between(results_df['alpha'], 0, gap, 
                     where=(gap > 0), alpha=0.3, color='red', 
                     label='Overfitting region')
    
    plt.suptitle(f'{ticker} - Lasso Regularization Analysis\n'
                 f'Bias-Variance Tradeoff Visualization', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved: {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Lasso regularization')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker to analyze')
    parser.add_argument('--n-alphas', type=int, default=50, 
                       help='Number of alpha values to test (default: 50)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"📊 LASSO REGULARIZATION ANALYSIS: {args.ticker}")
    print(f"{'='*80}")
    print(f"\nObjective: Visualize bias-variance tradeoff")
    print(f"Method: Test {args.n_alphas} different regularization strengths (α)")
    
    # Load and split data
    print(f"\n📂 Loading data...")
    train_df, test_df = load_and_split_data(args.ticker)
    
    print(f"   Training rows: {len(train_df):,}")
    print(f"   Testing rows:  {len(test_df):,}")
    print(f"   Features: {len(ALL_FEATURES)}")
    
    # Prepare data
    print(f"\n🔧 Preparing features...")
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
    print(f"   ✓ Features scaled")
    
    # Generate alpha values (log scale from 10^-5 to 10^1 for better range)
    alphas = np.logspace(-5, 1, args.n_alphas)
    
    print(f"\n🤖 Training {args.n_alphas} Lasso models...")
    results_df = analyze_regularization(X_train, X_test, y_train, y_test, alphas)
    
    # Find best model
    best_idx = results_df['test_r2'].idxmax()
    best_alpha = results_df.loc[best_idx, 'alpha']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    best_n_features = int(results_df.loc[best_idx, 'n_nonzero_coefs'])
    
    print(f"\n{'='*80}")
    print(f"📈 RESULTS")
    print(f"{'='*80}")
    
    print(f"\n🏆 BEST MODEL:")
    print(f"   {'─'*70}")
    print(f"   Alpha (α):              {best_alpha:.2e}")
    print(f"   Test R²:                {best_r2:.6f}")
    print(f"   Train R²:               {results_df.loc[best_idx, 'train_r2']:.6f}")
    print(f"   Overfitting gap:        {results_df.loc[best_idx, 'train_r2'] - best_r2:.6f}")
    print(f"   Non-zero coefficients:  {best_n_features} / {len(ALL_FEATURES)}")
    print(f"   Test RMSE:              {results_df.loc[best_idx, 'test_rmse']:.6f}")
    print(f"   Test MAE:               {results_df.loc[best_idx, 'test_mae']:.6f}")
    
    print(f"\n📊 EXTREMES:")
    print(f"   {'─'*70}")
    print(f"   Weakest regularization (α={alphas[0]:.2e}):")
    print(f"      Test R²: {results_df['test_r2'].iloc[0]:.6f}")
    print(f"      Features: {int(results_df['n_nonzero_coefs'].iloc[0])} (overfitting)")
    
    print(f"\n   Strongest regularization (α={alphas[-1]:.2e}):")
    print(f"      Test R²: {results_df['test_r2'].iloc[-1]:.6f}")
    print(f"      Features: {int(results_df['n_nonzero_coefs'].iloc[-1])} (underfitting)")
    
    # Save results
    print(f"\n💾 Saving results...")
    results_file = RESULTS_DIR / f"{args.ticker}_lasso_regularization_analysis.csv"
    results_df.to_csv(results_file, index=False)
    print(f"   ✓ Results saved: {results_file}")
    
    # Plot
    print(f"\n📊 Creating visualization...")
    plot_file = RESULTS_DIR / f"{args.ticker}_lasso_regularization_analysis.png"
    plot_regularization_analysis(results_df, args.ticker, plot_file)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   {'─'*70}")
    print(f"   • Left side (low α): High variance, overfitting")
    print(f"   • Sweet spot (α={best_alpha:.2e}): Best generalization")
    print(f"   • Right side (high α): High bias, underfitting")
    print(f"   • Optimal model uses only {best_n_features} most important features")
    
    print(f"\n📁 Files saved:")
    print(f"   • {results_file.name}")
    print(f"   • {plot_file.name}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
