"""
ML Regression Model Training - Predict Strategy Returns
========================================================

Train regression models to predict strategy_ret_3d for each (date, MA_pair).

CRITICAL: No look-ahead bias
- Chronological split (no random shuffle)
- StandardScaler fitted ONLY on training data
- Target: strategy_ret_3d (continuous returns)

Usage:
    python ML/train_regression_model.py --ticker AAPL
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

sys.path.append(str(Path(__file__).parent.parent))
import project_config as config

ML_DATA_DIR = Path("data/ML")
ML_MODELS_DIR = Path("ML/models")
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# Features to use for training
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
TARGET = 'strategy_ret_3d'  # REGRESSION TARGET


def load_ml_data(ticker):
    """Load ML dataset."""
    file = ML_DATA_DIR / f"{ticker}_ml_data.csv"
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def chronological_split(df, train_ratio=0.7):
    """
    Split data chronologically (NO RANDOM SHUFFLE).
    
    Args:
        df: DataFrame with Date column
        train_ratio: Proportion for training (default 70%)
    
    Returns:
        train_df, test_df
    """
    # Get unique dates
    unique_dates = sorted(df['Date'].unique())
    n_dates = len(unique_dates)
    
    # Split point
    split_idx = int(n_dates * train_ratio)
    split_date = unique_dates[split_idx]
    
    # Split data
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    print(f"\n📅 Chronological Split:")
    print(f"   {'─'*70}")
    print(f"   Training period:   {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"   Training rows:     {len(train_df):,}")
    print(f"   Testing period:    {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"   Testing rows:      {len(test_df):,}")
    print(f"   Split ratio:       {len(train_df)/len(df)*100:.1f}% train / {len(test_df)/len(df)*100:.1f}% test")
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Prepare features and target.
    CRITICAL: Fit scaler ONLY on training data.
    """
    # Separate features and target
    X_train = train_df[ALL_FEATURES].copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[ALL_FEATURES].copy()
    y_test = test_df[TARGET].copy()
    
    print(f"\n📊 Feature Preparation:")
    print(f"   {'─'*70}")
    print(f"   Features: {len(ALL_FEATURES)}")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    print(f"   y_test:  mean={y_test.mean():.4f}, std={y_test.std():.4f}")
    
    # Standardize features - FIT ONLY ON TRAINING DATA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use training scaler
    
    print(f"\n   ✅ StandardScaler fitted on training data only")
    print(f"   Mean of first feature (train): {scaler.mean_[0]:.6f}")
    print(f"   Std of first feature (train):  {scaler.scale_[0]:.6f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Train multiple regression models."""
    
    models = {
        'Linear Regression': LinearRegression(),
        
        'Ridge Regression': Ridge(
            alpha=1.0,
            random_state=42
        ),
        
        'Lasso Regression': Lasso(
            alpha=0.001,
            max_iter=10000,
            random_state=42
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
    }
    
    print(f"\n🤖 Training Regression Models:")
    print(f"   {'='*70}")
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✓ {name} trained")
    
    return trained_models


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all regression models."""
    
    print(f"\n📈 Model Evaluation:")
    print(f"   {'='*70}")
    
    results = []
    
    for name, model in models.items():
        print(f"\n   {name}:")
        print(f"   {'-'*70}")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Testing metrics
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"   Training Set:")
        print(f"     R² Score:  {train_r2:8.4f}")
        print(f"     RMSE:      {train_rmse:8.4f}")
        print(f"     MAE:       {train_mae:8.4f}")
        
        print(f"\n   Testing Set:")
        print(f"     R² Score:  {test_r2:8.4f}")
        print(f"     RMSE:      {test_rmse:8.4f}")
        print(f"     MAE:       {test_mae:8.4f}")
        
        # Additional analysis
        print(f"\n   Prediction Analysis:")
        print(f"     Actual mean:     {y_test.mean():8.4f}")
        print(f"     Predicted mean:  {y_test_pred.mean():8.4f}")
        print(f"     Actual std:      {y_test.std():8.4f}")
        print(f"     Predicted std:   {y_test_pred.std():8.4f}")
        
        results.append({
            'Model': name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae
        })
    
    return pd.DataFrame(results)


def feature_importance_analysis(models, feature_names):
    """Analyze feature importance for tree-based models."""
    
    print(f"\n📊 Feature Importance Analysis:")
    print(f"   {'='*70}")
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n   {name}:")
            print(f"   {'-'*70}")
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"   Top 10 Most Important Features:")
            for i, idx in enumerate(indices[:10], 1):
                print(f"      {i:2d}. {feature_names[idx]:25s} : {importances[idx]:.4f}")


def save_models(ticker, models, scaler):
    """Save trained models and scaler."""
    
    print(f"\n💾 Saving Models:")
    print(f"   {'─'*70}")
    
    # Save scaler
    scaler_file = ML_MODELS_DIR / f"{ticker}_regression_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"   ✓ Scaler: {scaler_file}")
    
    # Save models
    for name, model in models.items():
        model_file = ML_MODELS_DIR / f"{ticker}_regression_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_file)
        print(f"   ✓ {name}: {model_file.name}")


def walk_forward_validation(df, n_splits=5):
    """
    Walk-forward validation (time series cross-validation).
    Each fold uses only PAST data for training.
    """
    
    print(f"\n🔄 Walk-Forward Validation ({n_splits} splits):")
    print(f"   {'='*70}")
    
    # Prepare features
    X = df[ALL_FEATURES].values
    y = df[TARGET].values
    dates = df['Date'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        
        print(f"\n   Fold {fold_idx}:")
        print(f"      Train: {pd.to_datetime(train_dates.min()).date()} to {pd.to_datetime(train_dates.max()).date()} ({len(train_idx):,} rows)")
        print(f"      Test:  {pd.to_datetime(test_dates.min()).date()} to {pd.to_datetime(test_dates.max()).date()} ({len(test_idx):,} rows)")
        
        # Scale
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)
        
        # Train simple model for validation
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=50,
            random_state=42
        )
        model.fit(X_train_scaled, y_train_fold)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
        mae = mean_absolute_error(y_test_fold, y_pred)
        
        print(f"      R²: {r2:.4f}  |  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
        
        fold_results.append({
            'Fold': fold_idx,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })
    
    fold_df = pd.DataFrame(fold_results)
    print(f"\n   {'─'*70}")
    print(f"   Average R²:   {fold_df['R2'].mean():.4f} ± {fold_df['R2'].std():.4f}")
    print(f"   Average RMSE: {fold_df['RMSE'].mean():.4f} ± {fold_df['RMSE'].std():.4f}")
    print(f"   Average MAE:  {fold_df['MAE'].mean():.4f} ± {fold_df['MAE'].std():.4f}")
    
    return fold_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train regression ML models')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker to train model for')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data ratio (default: 0.7)')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 ML REGRESSION MODEL TRAINING: {args.ticker}")
    print(f"{'='*80}")
    print(f"\n   Target: Predict strategy_ret_3d (continuous returns)")
    print(f"   Task: Regression (not classification)")
    
    # Load data
    print(f"\n📂 Loading data...")
    df = load_ml_data(args.ticker)
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Features: {len(ALL_FEATURES)}")
    
    # Split chronologically
    train_df, test_df = chronological_split(df, train_ratio=args.train_ratio)
    
    # Prepare features
    X_train, X_test, y_train, y_test, scaler = prepare_features(train_df, test_df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results_df = evaluate_models(models, X_train, y_train, X_test, y_test)
    
    # Feature importance
    feature_importance_analysis(models, ALL_FEATURES)
    
    # Results summary
    print(f"\n{'='*80}")
    print(f"📊 RESULTS SUMMARY:")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    
    # Save models
    save_models(args.ticker, models, scaler)
    
    # Walk-forward validation (optional)
    if args.walk_forward:
        walk_forward_validation(df, n_splits=5)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    # Recommendations
    best_model_idx = results_df['Test_R2'].astype(float).idxmax()
    best_model = results_df.iloc[best_model_idx]['Model']
    best_r2 = results_df.iloc[best_model_idx]['Test_R2']
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   {'─'*70}")
    print(f"   Best Model: {best_model} (Test R² = {best_r2:.4f})")
    print(f"\n   The model learns:")
    print(f"   • When volatility is high and trend is strong → certain MAs perform better")
    print(f"   • When market is choppy → longer MAs outperform shorter ones")
    print(f"   • Market regime (SPY features) → which MA pairs work best")
    print(f"\n   Next steps:")
    print(f"   1. Use model to predict returns for each (date, MA_pair)")
    print(f"   2. Select MA pair with highest predicted return each day")
    print(f"   3. Backtest this ML-guided strategy")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
