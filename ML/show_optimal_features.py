"""
Show which features the optimal Lasso model selected
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Use project root directory for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Load the regularization analysis results
results_file = PROJECT_ROOT / "data" / "ML" / "regularization_analysis" / "AAPL_lasso_regularization_analysis.csv"
results_df = pd.read_csv(results_file)

# Find optimal alpha (best test R²)
best_idx = results_df['test_r2'].idxmax()
optimal_alpha = results_df.loc[best_idx, 'alpha']
best_r2 = results_df.loc[best_idx, 'test_r2']

print(f"\n{'='*80}")
print(f"🎯 OPTIMAL LASSO MODEL FEATURES (α = {optimal_alpha:.2e})")
print(f"{'='*80}")
print(f"\nTest R²: {best_r2:.6f}")

# Load ML data
ml_data = pd.read_csv(PROJECT_ROOT / "data" / "ML" / "AAPL_ml_data.csv")
ml_data['Date'] = pd.to_datetime(ml_data['Date'])
ml_data = ml_data.sort_values('Date').reset_index(drop=True)

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

# Split data (70/30 chronologically)
unique_dates = sorted(ml_data['Date'].unique())
split_idx = int(len(unique_dates) * 0.7)
split_date = unique_dates[split_idx]

train_df = ml_data[ml_data['Date'] < split_date].copy()
test_df = ml_data[ml_data['Date'] >= split_date].copy()

# Prepare data
X_train = train_df[ALL_FEATURES].values
y_train = train_df[TARGET].values
X_test = test_df[ALL_FEATURES].values
y_test = test_df[TARGET].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with optimal alpha
model = Lasso(alpha=optimal_alpha, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

# Get non-zero coefficients
feature_importance = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
})

# Sort by absolute coefficient
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

# Show selected features
selected_features = feature_importance[feature_importance['Coefficient'] != 0].copy()
n_selected = len(selected_features)

print(f"\n📊 SELECTED FEATURES ({n_selected} out of {len(ALL_FEATURES)}):")
print(f"{'─'*80}\n")

for i, row in selected_features.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    
    # Get feature category
    if feature in GLOBAL_FEATURES:
        category = "🌐 Global Market"
    elif feature in MA_SPECIFIC_FEATURES:
        category = "📈 MA-Specific"
    else:
        category = "⚙️ MA Parameters"
    
    # Feature description
    descriptions = {
        'ret_1d': '1-day return',
        'ret_5d': '5-day return',
        'ret_20d': '20-day return',
        'momentum_1m': '1-month momentum',
        'momentum_3m': '3-month momentum',
        'vol_20d': '20-day volatility',
        'volume_20d_avg': '20-day average volume',
        'volume_ratio': 'Current volume / 20-day avg',
        'price_over_ma200': 'Price relative to MA200',
        'spy_ret_5d': 'SPY 5-day return',
        'spy_ret_20d': 'SPY 20-day return',
        'spy_vol_20d': 'SPY 20-day volatility',
        'spy_ma_ratio_20_50': 'SPY MA20/MA50 ratio',
        'spy_autocorr_1d': 'SPY 1-day autocorrelation',
        'ma_short_t': 'Short MA value at time t',
        'ma_long_t': 'Long MA value at time t',
        'ma_diff_t': 'MA short - MA long',
        'ma_ratio_t': 'MA short / MA long',
        'signal_t': 'Trading signal (-1, 0, +1)',
        'short_window': 'Short MA period',
        'long_window': 'Long MA period'
    }
    
    desc = descriptions.get(feature, feature)
    sign = "📈 Positive" if coef > 0 else "📉 Negative"
    
    print(f"{i+1}. {feature:20s} {category:20s}")
    print(f"   Coefficient: {coef:+.6f}  [{sign}]")
    print(f"   Description: {desc}")
    print()

# Show dropped features
dropped_features = feature_importance[feature_importance['Coefficient'] == 0].copy()
n_dropped = len(dropped_features)

print(f"\n🚫 DROPPED FEATURES ({n_dropped} out of {len(ALL_FEATURES)}):")
print(f"{'─'*80}")
print(f"These features had coefficient = 0 (eliminated by Lasso):\n")

for i, row in dropped_features.iterrows():
    feature = row['Feature']
    if feature in GLOBAL_FEATURES:
        category = "🌐 Global"
    elif feature in MA_SPECIFIC_FEATURES:
        category = "📈 MA-Specific"
    else:
        category = "⚙️ Parameters"
    print(f"  • {feature:20s} ({category})")

print(f"\n{'='*80}")
print(f"💡 INTERPRETATION:")
print(f"{'─'*80}")
print(f"The model uses only {n_selected} features, showing Lasso's automatic feature")
print(f"selection keeps only the most predictive variables and eliminates noise.")
print(f"This prevents overfitting and improves generalization!")
print(f"{'='*80}\n")
