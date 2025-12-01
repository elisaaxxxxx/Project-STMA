# Flexible Configuration for Trading Project

## 📁 Project Structure

```
Project/
├── project_config.py           # 🔧 Centralized configuration
├── run_pipeline.py            # 🚀 Traditional pipeline script
├── data/                      # 📊 Data
│   ├── SRC/                   # Raw and processed data
│   │   ├── raw/              # Downloaded prices
│   │   ├── processed/        # With moving averages and signals
│   │   └── results/          # Backtest results
│   └── ML/                    # 🤖 ML data
│       └── *_ml_data.csv     # Machine learning datasets
├── src/                       # 📈 Traditional strategy scripts
│   ├── data_loader.py
│   ├── calculate_moving_averages.py
│   ├── generate_signals.py
│   ├── backtest_signal_strategy.py
│   └── test_signal_variations.py
├── ML/                        # 🤖 Machine Learning
│   ├── create_ml_data.py     # Create ML dataset
│   ├── train_regression_model.py  # Train models
│   ├── inspect_models.py     # Inspect models
│   ├── verify_data_quality.py     # Verify no look-ahead bias
│   └── models/               # Trained models (.pkl)
└── README_CONFIG.md           # 📖 This file
```

## 🔧 Centralized Configuration

All project parameters are now centralized in `project_config.py`.

### How to modify the configuration:

1. **Open `project_config.py`**
2. **Modify values according to your needs:**

```python
# Change tickers
TICKERS = ['AAPL', 'SPY', 'MSFT', 'GOOGL']  # Add your stocks

# Change dates
START_DATE = '2020-01-01'  # New start date
END_DATE = '2024-12-31'    # New end date

# Other parameters...
TRANSACTION_COST = 0.002   # 0.2% cost
MA_PERIODS = [5, 10, 20, 50, 100, 200]  # Moving averages
```

3. **Save the file**
4. **All scripts will automatically use the new configuration!**

## 🚀 Usage

### Traditional Strategy Pipeline

```bash
# Run complete traditional pipeline
python run_pipeline.py --all

# Interactive menu
python run_pipeline.py

# Individual steps
python run_pipeline.py --ma        # Moving averages only
python run_pipeline.py --signals   # Signals only
python run_pipeline.py --backtest  # Backtest only
```

### 🤖 Machine Learning Pipeline (NEW!)

#### Step 1: Create ML Dataset
```bash
# Create ML training data for one ticker
python ML/create_ml_data.py --ticker AAPL

# Create for all tickers
python ML/create_ml_data.py --all
```

**What it does:**
- Creates one row per (date, MA_pair) combination
- 21 features: 14 global market features + 5 MA-specific + 2 MA parameters
- Target: `strategy_ret_3d` (3-day strategy returns)
- Saves to `data/ML/TICKER_ml_data.csv`

#### Step 2: Verify Data Quality
```bash
# Check for look-ahead bias and data integrity
python ML/verify_data_quality.py --ticker AAPL
```

**What it checks:**
- ✅ No future data in features
- ✅ Chronological ordering
- ✅ Proper target alignment
- ✅ No NaN values
- ✅ Proper 30/70 target distribution

#### Step 3: Train Regression Models
```bash
# Train models with 70/30 split
python ML/train_regression_model.py --ticker AAPL

# Train with walk-forward validation
python ML/train_regression_model.py --ticker AAPL --walk-forward

# Custom train/test ratio
python ML/train_regression_model.py --ticker AAPL --train-ratio 0.8
```

**What it trains:**
- Linear Regression
- Ridge Regression  
- Lasso Regression (usually best)
- Random Forest Regressor
- Gradient Boosting Regressor

**Models saved to:** `ML/models/TICKER_regression_*.pkl`

#### Step 4: Inspect Trained Models
```bash
# View detailed model information
python ML/inspect_models.py --ticker AAPL
```

**What you see:**
- Model coefficients/weights
- Feature importance
- Scaler parameters
- How to use models for prediction

### Individual Traditional Scripts
```bash
# In order:
python src/calculate_moving_averages.py
python src/generate_signals.py  
python src/backtest_signal_strategy.py
python src/test_signal_variations.py
```

## 📊 Configuration Examples

### To analyze cryptocurrencies:
```python
TICKERS = ['BTC-USD', 'ETH-USD', 'ADA-USD']
START_DATE = '2021-01-01'
END_DATE = '2024-12-31'
```

### For a 5-year analysis:
```python
TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA']
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
```

### For different moving averages:
```python
MA_PERIODS = [3, 7, 14, 30, 60, 120]  # Shorter term
# or
MA_PERIODS = [10, 25, 50, 100, 200, 300]  # Longer term
```

### ML-specific configurations:
The ML pipeline uses 12 MA pair combinations automatically:
- (5,10), (5,20), (5,50)
- (10,20), (10,50), (10,100)
- (20,50), (20,100), (20,200)
- (50,100), (50,200), (100,200)

## 🔍 Automatic Validation

The system automatically validates your configuration:
- ✅ Date format
- ✅ Parameter consistency
- ✅ Required file existence  
- ❌ Clearly displays errors

## 📈 Results

Results are saved in:
- **Traditional Backtests**: `data/SRC/results/backtest/`
- **Variations**: `data/SRC/results/variations/`
- **ML Data**: `data/ML/`
- **ML Models**: `ML/models/`

File names automatically adapt to your configuration!

## 🛠️ Advantages of the System

### Traditional Pipeline:
1. **No broken code**: Change config, everything works
2. **Automatic validation**: Errors detected before execution
3. **Consistent file names**: Everything adapts automatically
4. **Orchestrated pipeline**: Single script to do everything
5. **Total flexibility**: Tickers, dates, parameters all modifiable

### Machine Learning Pipeline:
1. **No look-ahead bias**: Strict chronological splits
2. **Proper feature engineering**: 21 carefully designed features
3. **Multiple models**: Compare Linear, Ridge, Lasso, RF, GBM
4. **Feature importance**: Understand what drives performance
5. **Reproducible**: Saved models (.pkl) for consistent predictions
6. **Walk-forward validation**: Test across different time periods

## 🚨 Important Notes

1. **Data download**: Make sure you have data for your new tickers
2. **yfinance compatibility**: Check that your tickers are supported
3. **Disk space**: More tickers = more files generated
4. **Processing time**: More data = longer processing time
5. **ML requirements**: 
   - Need sufficient historical data (recommended: 10+ years)
   - Training can take several minutes for large datasets
   - Models are saved and can be reused without retraining

## 🤖 ML Features Explained

### 21 Features per Observation:

**Global Market Features (14):**
1. `ret_1d` - 1-day return
2. `ret_5d` - 5-day return  
3. `ret_20d` - 20-day return
4. `momentum_1m` - 1-month momentum
5. `momentum_3m` - 3-month momentum
6. `vol_20d` - 20-day volatility
7. `volume_20d_avg` - 20-day average volume
8. `volume_ratio` - Current volume / average
9. `price_over_ma200` - Price vs 200-day MA
10. `spy_ret_5d` - SPY 5-day return (market benchmark)
11. `spy_ret_20d` - SPY 20-day return
12. `spy_vol_20d` - SPY 20-day volatility
13. `spy_ma_ratio_20_50` - SPY MA(20)/MA(50) ratio
14. `spy_autocorr_1d` - SPY 1-day autocorrelation

**MA-Specific Features (5):**
15. `ma_short_t` - Short MA value at time t
16. `ma_long_t` - Long MA value at time t
17. `ma_diff_t` - Short MA - Long MA
18. `ma_ratio_t` - Short MA / Long MA
19. `signal_t` - Trading signal (1=long, 0=short)

**MA Parameters (2):**
20. `short_window` - Short MA period (5, 10, 20, 50, 100)
21. `long_window` - Long MA period (10, 20, 50, 100, 200)

**Target:**
- `strategy_ret_3d` - 3-day strategy return (what we predict)

### Model Performance Metrics:

- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Best Model**: Usually Lasso Regression (R² ~ 0.01-0.02)

**Note**: Low R² (~1-2%) is normal for financial data! Even small predictive power is valuable when selecting best MA pairs.

## 🆘 Troubleshooting

### "File not found"
- Check that data exists in `data/SRC/`
- Use `data_loader.py` to download data

### "Configuration error"
- Script shows exactly what to fix
- Verify date format (YYYY-MM-DD)
- Check that TICKERS is not empty

### "Import Error"
- Run from project root directory
- Verify that `project_config.py` exists

### ML-specific issues:

**"Target shows 100% class 1"**
- Fixed! Ranking uses `method='first'` to break ties
- Target should show ~33% class 1, ~67% class 0

**"Negative R² score"**
- Normal for some models (means worse than baseline)
- Use Lasso Regression (usually best)
- Low R² (~0.01) is expected for financial data

**"Models not found"**
- Run training script first: `python ML/train_regression_model.py --ticker AAPL`
- Check `ML/models/` directory exists

**"Not enough data"**
- Need at least 200+ days of data
- Recommended: 10+ years for robust training
- Adjust START_DATE in config

## 📞 Support

Modify `project_config.py` and rerun!
Everything else updates automatically. 🎉

## 🎯 Quick Start Guide

### For Traditional Strategy:
```bash
python run_pipeline.py --all
```

### For Machine Learning:
```bash
# 1. Create ML data
python ML/create_ml_data.py --ticker AAPL

# 2. Verify quality
python ML/verify_data_quality.py --ticker AAPL

# 3. Train models
python ML/train_regression_model.py --ticker AAPL

# 4. Inspect results
python ML/inspect_models.py --ticker AAPL
```

That's it! 🚀