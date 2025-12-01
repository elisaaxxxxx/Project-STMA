"""
Inspect Saved ML Models
=======================

Shows detailed information about the trained models.

Usage:
    python ML/inspect_models.py --ticker AAPL
"""

import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

ML_MODELS_DIR = Path("ML/models")

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


def inspect_scaler(ticker):
    """Show scaler details."""
    
    scaler_file = ML_MODELS_DIR / f"{ticker}_regression_scaler.pkl"
    scaler = joblib.load(scaler_file)
    
    print(f"\n{'='*80}")
    print(f"📊 SCALER DETAILS")
    print(f"{'='*80}")
    print(f"\nFile: {scaler_file.name}")
    print(f"Type: {type(scaler).__name__}")
    print(f"Number of features: {len(scaler.mean_)}")
    
    print(f"\n{'─'*80}")
    print(f"FEATURE SCALING PARAMETERS (from training data)")
    print(f"{'─'*80}")
    print(f"{'Feature':<25} {'Mean':>12} {'Std Dev':>12}")
    print(f"{'─'*80}")
    
    for i, feature in enumerate(ALL_FEATURES):
        print(f"{feature:<25} {scaler.mean_[i]:>12.6f} {scaler.scale_[i]:>12.6f}")


def inspect_linear_models(ticker):
    """Show linear model details (Linear, Ridge, Lasso)."""
    
    models = {
        'Linear Regression': f"{ticker}_regression_linear_regression.pkl",
        'Ridge Regression': f"{ticker}_regression_ridge_regression.pkl",
        'Lasso Regression': f"{ticker}_regression_lasso_regression.pkl"
    }
    
    for name, filename in models.items():
        filepath = ML_MODELS_DIR / filename
        if not filepath.exists():
            continue
            
        model = joblib.load(filepath)
        
        print(f"\n{'='*80}")
        print(f"📈 {name.upper()}")
        print(f"{'='*80}")
        print(f"\nFile: {filename}")
        print(f"Type: {type(model).__name__}")
        
        # Model parameters
        if hasattr(model, 'alpha'):
            print(f"Alpha (regularization): {model.alpha}")
        
        # Coefficients
        print(f"\nIntercept: {model.intercept_:.6f}")
        print(f"Number of coefficients: {len(model.coef_)}")
        
        # Show coefficients
        print(f"\n{'─'*80}")
        print(f"FEATURE COEFFICIENTS (weights)")
        print(f"{'─'*80}")
        print(f"{'Feature':<25} {'Coefficient':>15} {'Abs Value':>15}")
        print(f"{'─'*80}")
        
        # Sort by absolute value
        coef_with_features = list(zip(ALL_FEATURES, model.coef_))
        coef_with_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in coef_with_features:
            print(f"{feature:<25} {coef:>15.6f} {abs(coef):>15.6f}")
        
        # Count non-zero coefficients (for Lasso)
        non_zero = np.sum(model.coef_ != 0)
        print(f"\n{'─'*80}")
        print(f"Non-zero coefficients: {non_zero} / {len(model.coef_)}")
        
        if non_zero < len(model.coef_):
            zero_features = [ALL_FEATURES[i] for i in range(len(model.coef_)) if model.coef_[i] == 0]
            print(f"Eliminated features: {', '.join(zero_features)}")


def inspect_tree_models(ticker):
    """Show tree-based model details (Random Forest, Gradient Boosting)."""
    
    models = {
        'Random Forest': f"{ticker}_regression_random_forest.pkl",
        'Gradient Boosting': f"{ticker}_regression_gradient_boosting.pkl"
    }
    
    for name, filename in models.items():
        filepath = ML_MODELS_DIR / filename
        if not filepath.exists():
            continue
            
        model = joblib.load(filepath)
        
        print(f"\n{'='*80}")
        print(f"🌲 {name.upper()}")
        print(f"{'='*80}")
        print(f"\nFile: {filename}")
        print(f"Type: {type(model).__name__}")
        
        # Model parameters
        print(f"\nModel Parameters:")
        print(f"  Number of estimators: {model.n_estimators}")
        print(f"  Max depth: {model.max_depth}")
        print(f"  Min samples split: {model.min_samples_split}")
        print(f"  Min samples leaf: {model.min_samples_leaf}")
        
        if hasattr(model, 'learning_rate'):
            print(f"  Learning rate: {model.learning_rate}")
        
        print(f"  Random state: {model.random_state}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"\n{'─'*80}")
            print(f"FEATURE IMPORTANCE")
            print(f"{'─'*80}")
            print(f"{'Feature':<25} {'Importance':>15} {'Percentage':>15}")
            print(f"{'─'*80}")
            
            # Sort by importance
            importances = model.feature_importances_
            feature_importance = list(zip(ALL_FEATURES, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in feature_importance:
                pct = importance * 100
                print(f"{feature:<25} {importance:>15.6f} {pct:>14.2f}%")
            
            # Summary
            top_5_importance = sum([imp for _, imp in feature_importance[:5]])
            print(f"\n{'─'*80}")
            print(f"Top 5 features account for: {top_5_importance*100:.2f}% of total importance")


def show_how_to_use(ticker):
    """Show example code for using the models."""
    
    print(f"\n{'='*80}")
    print(f"💡 HOW TO USE THESE MODELS")
    print(f"{'='*80}")
    
    print(f"""
To make predictions with the saved models:

```python
import joblib
import numpy as np

# 1. Load scaler and model
scaler = joblib.load('ML/models/{ticker}_regression_scaler.pkl')
model = joblib.load('ML/models/{ticker}_regression_lasso_regression.pkl')

# 2. Prepare features for prediction (21 features)
# Example: Features for one (date, MA_pair) combination
X_new = np.array([[
    0.0015,   # ret_1d
    0.0050,   # ret_5d
    0.0200,   # ret_20d
    0.0300,   # momentum_1m
    0.0500,   # momentum_3m
    0.0250,   # vol_20d
    50000000, # volume_20d_avg
    1.2,      # volume_ratio
    0.15,     # price_over_ma200
    0.0040,   # spy_ret_5d
    0.0180,   # spy_ret_20d
    0.0150,   # spy_vol_20d
    1.02,     # spy_ma_ratio_20_50
    0.05,     # spy_autocorr_1d
    150.5,    # ma_short_t
    148.2,    # ma_long_t
    2.3,      # ma_diff_t
    1.015,    # ma_ratio_t
    1.0,      # signal_t (1=long, 0=short)
    5.0,      # short_window
    20.0      # long_window
]])

# 3. Scale features (MUST use training scaler!)
X_scaled = scaler.transform(X_new)

# 4. Predict
predicted_return = model.predict(X_scaled)[0]

print(f"Predicted 3-day strategy return: {{predicted_return:.4f}}")
print(f"Predicted return: {{predicted_return*100:.2f}}%")
```

For each trading day:
1. Calculate features for all 12 MA pairs
2. Predict strategy_ret_3d for each pair
3. Select MA pair with highest predicted return
4. Trade using that pair's signal
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect saved ML models')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker to inspect models for')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🔍 INSPECTING ML MODELS FOR {args.ticker}")
    print(f"{'='*80}")
    
    # Inspect scaler
    inspect_scaler(args.ticker)
    
    # Inspect linear models
    inspect_linear_models(args.ticker)
    
    # Inspect tree models
    inspect_tree_models(args.ticker)
    
    # Show usage example
    show_how_to_use(args.ticker)
    
    print(f"\n{'='*80}")
    print(f"✅ INSPECTION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
