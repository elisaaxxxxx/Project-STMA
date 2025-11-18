# 📊 Projet de Stratégie de Trading - Moving Averages

## 🏗️ Structure du Projet

```
Project/
├── 📁 src/                                    # 🔧 PROGRAMMES PYTHON
│   ├── 📄 data_loader.py                     # Téléchargement des données yfinance
│   ├── 📄 calculate_moving_averages.py       # Calcul des moyennes mobiles
│   ├── 📄 generate_signals.py                # Génération des signaux de trading
│   ├── 📄 backtest_signal_strategy.py        # Backtesting des stratégies
│   └── 📄 test_signal_variations.py          # Tests walk-forward (sans biais)
│
├── 📁 data/                                   # 📊 DONNÉES ET RÉSULTATS
│   ├── 📁 raw/                               # Données brutes téléchargées
│   │   ├── AAPL_2000-01-01_2025-11-01.csv
│   │   ├── MSFT_2000-01-01_2025-11-01.csv
│   │   └── ... (autres tickers)
│   │
│   ├── 📁 processed/                         # Données enrichies
│   │   ├── AAPL_*_with_MAs.csv              # Avec moyennes mobiles
│   │   ├── AAPL_*_with_signals.csv          # Avec signaux de trading
│   │   └── ... (autres tickers)
│   │
│   └── 📁 results/                           # Résultats des analyses
│       ├── 📁 backtest/                      # Résultats de backtests
│       │   ├── AAPL_*_backtest_results.csv
│       │   └── AAPL_*_backtest_plot.png
│       │
│       └── 📁 variations/                    # Tests de variations walk-forward
│           ├── AAPL_walk_forward_detailed.csv
│           ├── AAPL_strategy_selections.csv
│           └── AAPL_signal_variations_equity_curves.png
│
├── ⚙️ project_config.py                      # CONFIGURATION CENTRALE
├── 🚀 run_pipeline.py                        # SCRIPT PRINCIPAL
├── 📖 README.md                              # Cette documentation
└── 📋 README_CONFIG.md                       # Guide de configuration
```

---

## 🚀 Utilisation Rapide

### 1️⃣ **Modifier la Configuration**
```python
# Éditez project_config.py
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
START_DATE = '2000-01-01'
END_DATE = '2025-11-01'
```

### 2️⃣ **Lancer le Pipeline Complet**
```bash
python run_pipeline.py --all
```

### 3️⃣ **Consulter les Résultats**
- **Backtests** : `data/results/backtest/`
- **Analyses walk-forward** : `data/results/variations/`

---

## 🔧 Commandes Disponibles

| Commande | Description |
|----------|-------------|
| `python run_pipeline.py --all` | 🔄 Pipeline complet (tout) |
| `python run_pipeline.py --config` | ⚙️ Afficher la configuration |
| `python run_pipeline.py --ma` | 📊 Calculer moyennes mobiles |
| `python run_pipeline.py --signals` | 📈 Générer signaux |
| `python run_pipeline.py --backtest` | 🎯 Backtest seulement |
| `python run_pipeline.py --variations` | 🔬 Tests variations |
| `python src/data_loader.py` | 📥 Télécharger données |

---

## 📈 Stratégies Implémentées

### **Moyennes Mobiles Utilisées**
- **MA 5, 10, 20** : Court terme
- **MA 50, 100** : Moyen terme  
- **MA 200** : Long terme

### **Signaux Générés**
1. **Signal Court (5 vs 20)** : `Signal_5_20_short`
2. **Signal Moyen (10 vs 50)** : `Signal_10_50_medium`
3. **Signal Long (20 vs 100)** : `Signal_20_100_long`
4. **Signal Très Long (50 vs 200)** : `Signal_50_200_vlong`

### **Stratégies Testées**
- ✅ **Original** : ≥2 signaux sur 4
- 📊 **Court terme uniquement** : Signal 5 vs 20
- 📈 **Moyen terme uniquement** : Signal 10 vs 50
- 📉 **Long terme uniquement** : Signal 50 vs 200
- 🔄 **Court OU Long** : Signal court OU long
- ⚡ **Court ET Moyen** : Signal court ET moyen
- 🎯 **Long ET Très Long** : Signal long ET très long
- 🧮 **≥3 signaux** : Au moins 3 sur 4
- 💎 **Tous les signaux** : Les 4 signaux positifs

---

## 📊 Exemple de Résultats

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

---

## 🎯 Points Clés

### ✅ **Avantages de cette Structure**
- **🗂️ Organisation claire** : Programmes séparés des données
- **🔧 Maintenance facile** : Tout le code dans `src/`
- **📊 Données organisées** : Raw → Processed → Results
- **⚙️ Configuration centralisée** : Un seul fichier à modifier

### 🧠 **Walk-Forward Analysis**
- **Élimine le biais de look-ahead** : Sélection des stratégies basée seulement sur les données passées
- **Plus réaliste** : Performance obtenue sans "voir l'avenir"
- **Fenêtre glissante** : 36 mois training + 6 mois test

### 💰 **Paramètres Financiers**
- **Coûts de transaction** : 0.1% par trade
- **252 jours de trading** par an
- **Réinvestissement des profits**

---

## 🔄 Workflow Type

1. **📥 Téléchargement** → `data/raw/`
2. **📊 Moyennes mobiles** → `data/processed/*_MAs.csv`
3. **📈 Signaux** → `data/processed/*_signals.csv`
4. **🎯 Backtests** → `data/results/backtest/`
5. **🔬 Walk-Forward** → `data/results/variations/`

---

## 🛠️ Technologies Utilisées

- **Python 3.13+**
- **pandas** : Manipulation de données
- **yfinance** : Téléchargement de données financières
- **matplotlib** : Graphiques et visualisations
- **numpy** : Calculs mathématiques

---

*Créé par Elisa - Novembre 2025* 🚀

Mathieu est trop fort 