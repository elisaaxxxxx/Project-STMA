# 📊 Trading Strategy Project - Moving Averages + Machine Learning

## 🏗️ Project Structure

```
Project/
├── 📁 src/                                    # 🔧 TRADITIONAL STRATEGY SCRIPTS
│   ├── 📄 data_loader.py                     # Download data from yfinance
│   ├── 📄 calculate_moving_averages.py       # Calculate moving averages
│   ├── 📄 generate_signals.py                # Generate trading signals
│   ├── 📄 backtest_signal_strategy.py        # Backtest strategies
│   └── 📄 test_signal_variations.py          # Walk-forward tests (no bias)
│
├── 📁 ML/                                     # 🤖 MACHINE LEARNING PIPELINE
│   ├── � create_ml_data.py                  # Create ML training dataset
│   ├── 📄 train_regression_model.py          # Train regression models
│   ├── 📄 inspect_models.py                  # Inspect trained models
│   ├── 📄 verify_data_quality.py             # Verify no look-ahead bias
│   └── 📁 models/                            # Saved models (.pkl files)
│       ├── AAPL_regression_scaler.pkl
│       ├── AAPL_regression_lasso_regression.pkl
│       ├── AAPL_regression_random_forest.pkl
│       └── ... (other models)
│
├── 📁 data/                                   # 📊 DATA AND RESULTS
│   ├── 📁 SRC/                               # Traditional strategy data
│   │   ├── 📁 raw/                           # Raw downloaded data
│   │   │   ├── AAPL_2000-01-01_2025-11-01.csv
│   │   │   └── ... (other tickers)
│   │   │
│   │   ├── 📁 processed/                     # Enriched data
│   │   │   ├── AAPL_*_with_MAs.csv          # With moving averages
│   │   │   └── AAPL_*_with_signals.csv      # With trading signals
│   │   │
│   │   └── 📁 results/                       # Analysis results
│   │       ├── 📁 backtest/                  # Backtest results
│   │       └── 📁 variations/                # Walk-forward test results
│   │
│   └── 📁 ML/                                # 🤖 ML training data
│       ├── AAPL_ml_data.csv                  # ML dataset (75K+ rows)
│       └── ... (other tickers)
│
├── ⚙️ project_config.py                      # CENTRAL CONFIGURATION
├── 🚀 run_pipeline.py                        # MAIN SCRIPT (traditional)
├── 📖 README.md                              # This documentation
└── 📋 README_CONFIG.md                       # Configuration guide
```

---

## 🚀 Quick Start

### Traditional Strategy Pipeline

#### 1️⃣ **Modify Configuration**
```python
# Edit project_config.py
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START_DATE = '2000-01-01'
END_DATE = '2025-11-01'
```

#### 2️⃣ **Run Complete Pipeline**
```bash
python run_pipeline.py --all
```

#### 3️⃣ **View Results**
- **Backtests**: `data/SRC/results/backtest/`
- **Walk-forward analysis**: `data/SRC/results/variations/`

---

### 🤖 Machine Learning Pipeline (NEW!)

#### 1️⃣ **Create ML Dataset**
```bash
# For one ticker
python ML/create_ml_data.py --ticker AAPL

# For all tickers
python ML/create_ml_data.py --all
```

**What it does:**
- Creates 21 features per (date, MA_pair)
- One row per date × 12 MA pair combinations
- Saves to `data/ML/TICKER_ml_data.csv`

#### 2️⃣ **Verify Data Quality**
```bash
python ML/verify_data_quality.py --ticker AAPL
```

**Checks for:**
- ✅ No future data in features
- ✅ Proper chronological split
- ✅ Correct target distribution (30/70)

#### 3️⃣ **Train Models**
```bash
# Basic training (70/30 split)
python ML/train_regression_model.py --ticker AAPL

# With walk-forward validation
python ML/train_regression_model.py --ticker AAPL --walk-forward
```

**Models trained:**
- Linear Regression
- Ridge Regression
- Lasso Regression ⭐ (usually best)
- Random Forest
- Gradient Boosting

#### 4️⃣ **Inspect Models**
```bash
python ML/inspect_models.py --ticker AAPL
```

**Shows:**
- Model coefficients
- Feature importance
- How to use for predictions

---

## 🔧 Available Commands

### Traditional Pipeline

| Command | Description |
|----------|-------------|
| `python run_pipeline.py --all` | 🔄 Complete pipeline (everything) |
| `python run_pipeline.py --config` | ⚙️ Display configuration |
| `python run_pipeline.py --ma` | 📊 Calculate moving averages |
| `python run_pipeline.py --signals` | 📈 Generate signals |
| `python run_pipeline.py --backtest` | 🎯 Backtest only |
| `python run_pipeline.py --variations` | 🔬 Variations tests |
| `python src/data_loader.py` | 📥 Download data |

### Machine Learning Pipeline

| Command | Description |
|----------|-------------|
| `python ML/create_ml_data.py --ticker AAPL` | 🤖 Create ML dataset |
| `python ML/create_ml_data.py --all` | 🤖 Create for all tickers |
| `python ML/verify_data_quality.py --ticker AAPL` | ✅ Verify data quality |
| `python ML/train_regression_model.py --ticker AAPL` | 🎓 Train models |
| `python ML/train_regression_model.py --ticker AAPL --walk-forward` | 🔄 Train with validation |
| `python ML/inspect_models.py --ticker AAPL` | 🔍 Inspect trained models |

---

## 📈 Strategies Implemented

### **Traditional Moving Average Strategies**

**Moving Averages Used:**
- **MA 5, 10, 20**: Short term
- **MA 50, 100**: Medium term
- **MA 200**: Long term

**Signals Generated:**
1. **Short Signal (5 vs 20)**: `Signal_5_20_short`
2. **Medium Signal (10 vs 50)**: `Signal_10_50_medium`
3. **Long Signal (20 vs 100)**: `Signal_20_100_long`
4. **Very Long Signal (50 vs 200)**: `Signal_50_200_vlong`

**Strategies Tested:**
- ✅ **Original**: ≥2 signals out of 4
- 📊 **Short term only**: Signal 5 vs 20
- 📈 **Medium term only**: Signal 10 vs 50
- 📉 **Long term only**: Signal 50 vs 200
- 🔄 **Short OR Long**: Short signal OR long signal
- ⚡ **Short AND Medium**: Short signal AND medium signal
- 🎯 **Long AND Very Long**: Long signal AND very long signal
- 🧮 **≥3 signals**: At least 3 out of 4
- 💎 **All signals**: All 4 signals positive

---

### � **Machine Learning Strategy (NEW!)**

**Approach:**
- Predict `strategy_ret_3d` (3-day returns) for each MA pair
- Select best MA pair each day based on ML predictions
- Trade using that pair's signal

**Features (21 total):**

**Global Market Features (14):**
1. `ret_1d`, `ret_5d`, `ret_20d` - Price returns
2. `momentum_1m`, `momentum_3m` - Momentum indicators
3. `vol_20d` - Volatility
4. `volume_20d_avg`, `volume_ratio` - Volume indicators
5. `price_over_ma200` - Long-term trend
6. `spy_ret_5d`, `spy_ret_20d`, `spy_vol_20d` - Market benchmark
7. `spy_ma_ratio_20_50`, `spy_autocorr_1d` - Market regime

**MA-Specific Features (5):**
8. `ma_short_t`, `ma_long_t` - MA values
9. `ma_diff_t`, `ma_ratio_t` - MA relationships
10. `signal_t` - Current trading signal

**MA Parameters (2):**
11. `short_window`, `long_window` - MA periods

**Target:**
- `strategy_ret_3d` - 3-day strategy return to predict

**12 MA Pairs Tested:**
- (5,10), (5,20), (5,50)
- (10,20), (10,50), (10,100)
- (20,50), (20,100), (20,200)
- (50,100), (50,200), (100,200)

---

## 📊 Example Results

### Traditional Strategy
```
========================================================================================================================
FINAL SUMMARY: Walk-Forward vs Traditional Analysis
========================================================================================================================

Ticker | Method                    | CAGR     | Sharpe  | MaxDD    | Notes
------------------------------------------------------------------------------------------------------------------------
AAPL   | Walk-Forward (Clean)      |  20.92% |   0.79 | -55.38% | No look-ahead bias
AAPL   | Best Traditional          |  27.78% |   0.86 | -54.85% | Short OR Long
AAPL   | Buy & Hold                |  25.10% |   0.65 | -81.80% | Benchmark
```

### Machine Learning Models
```
================================================================================
📊 RESULTS SUMMARY - ML Regression Models
================================================================================

Model              | Test R²  | Test RMSE | Test MAE | Notes
----------------------------------------------------------------------------
Lasso Regression   |  0.0106  |  0.0325   | 0.0237   | ⭐ Best (simplest, no overfit)
Linear Regression  | -0.2776  |  0.0370   | 0.0282   | Poor generalization
Ridge Regression   | -0.2776  |  0.0370   | 0.0282   | Poor generalization
Random Forest      | -0.2553  |  0.0366   | 0.0270   | Overfits (Train R²=0.33)
Gradient Boosting  | -0.2738  |  0.0369   | 0.0274   | Overfits (Train R²=0.34)
```

**Note:** Low R² (~1%) is normal for financial data - even small predictive power helps select best MA pairs!

---

## 🎯 Key Points

### ✅ **Advantages of This Structure**
- **🗂️ Clear organization**: Programs separated from data
- **🔧 Easy maintenance**: All code in `src/` and `ML/`
- **📊 Organized data**: Raw → Processed → Results
- **⚙️ Centralized configuration**: Single file to modify
- **🤖 ML integration**: Complete pipeline from data to trained models

### 🧠 **Walk-Forward Analysis (No Look-Ahead Bias)**
- **Eliminates look-ahead bias**: Strategy selection based only on past data
- **More realistic**: Performance without "seeing the future"
- **Rolling window**: 36 months training + 6 months test

### 🤖 **Machine Learning Approach**
- **Regression task**: Predicts continuous returns (not just classification)
- **21 features**: Market conditions + MA characteristics
- **Chronological split**: 70% train (2000-2018), 30% test (2018-2025)
- **No look-ahead bias**: All features use only past data
- **Best model**: Lasso Regression (R² = 0.0106)
- **Feature importance**: 
  - Top 3: `price_over_ma200`, `vol_20d`, `spy_ma_ratio_20_50`
  - Lasso keeps only 2 features: `signal_t` and `spy_ret_20d`

### 💰 **Financial Parameters**
- **Transaction costs**: 0.1% per trade
- **252 trading days** per year
- **Profit reinvestment**

---

## 🔄 Typical Workflow

### Traditional Strategy
1. **📥 Download** → `data/SRC/raw/`
2. **📊 Moving averages** → `data/SRC/processed/*_MAs.csv`
3. **📈 Signals** → `data/SRC/processed/*_signals.csv`
4. **🎯 Backtests** → `data/SRC/results/backtest/`
5. **🔬 Walk-Forward** → `data/SRC/results/variations/`

### Machine Learning Pipeline
1. **📥 Load processed data** → From `data/SRC/processed/`
2. **🔧 Feature engineering** → Create 21 features per (date, MA_pair)
3. **💾 Save ML dataset** → `data/ML/*_ml_data.csv`
4. **✅ Verify quality** → Check for look-ahead bias
5. **🎓 Train models** → 5 regression models
6. **💾 Save models** → `ML/models/*.pkl`
7. **🔍 Inspect** → View coefficients, importance
8. **📈 Predict & backtest** → (Next step: backtest ML strategy)

---

## 🛠️ Technologies Used

- **Python 3.13+**
- **pandas**: Data manipulation
- **yfinance**: Financial data download
- **matplotlib**: Graphs and visualizations
- **numpy**: Mathematical calculations
- **scikit-learn**: Machine learning models
- **joblib**: Model persistence (.pkl files)

---

## 📚 Documentation

- **README.md** (this file): Project overview
- **README_CONFIG.md**: Detailed configuration guide with ML instructions
- See individual script docstrings for detailed usage

---

*Created by Elisa - December 2025* 🚀