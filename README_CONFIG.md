# Configuration Flexible pour le Projet de Trading

## 📁 Structure du Projet

```
Project/
├── project_config.py           # 🔧 Configuration centralisée
├── run_pipeline.py            # 🚀 Script principal
├── Data/src/                  # 📊 Données brutes
├── MA_strategy/               # 📈 Scripts de stratégie
└── README_CONFIG.md           # 📖 Ce fichier
```

## 🔧 Configuration Centralisée

Tous les paramètres du projet sont maintenant centralisés dans `project_config.py`. 

### Comment modifier la configuration :

1. **Ouvrez `project_config.py`**
2. **Modifiez les valeurs selon vos besoins :**

```python
# Changez les tickers
TICKERS = ['AAPL', 'SPY', 'MSFT', 'GOOGL']  # Ajoutez vos actions

# Changez les dates
START_DATE = '2020-01-01'  # Nouvelle date de début
END_DATE = '2024-12-31'    # Nouvelle date de fin

# Autres paramètres...
TRANSACTION_COST = 0.002   # 0.2% de coût
MA_PERIODS = [5, 10, 20, 50, 100, 200]  # Moyennes mobiles
```

3. **Sauvegardez le fichier**
4. **Tous les scripts utiliseront automatiquement la nouvelle configuration !**

## 🚀 Utilisation

### Option 1: Script Principal (Recommandé)
```bash
# Exécuter tout le pipeline
python run_pipeline.py --all

# Menu interactif
python run_pipeline.py

# Étapes individuelles
python run_pipeline.py --ma        # Moyennes mobiles seulement
python run_pipeline.py --signals   # Signaux seulement
python run_pipeline.py --backtest  # Backtest seulement
```

### Option 2: Scripts Individuels
```bash
# Dans l'ordre :
python MA_strategy/calculate_moving_averages.py
python MA_strategy/generate_signals.py  
python MA_strategy/backtest_signal_strategy.py
python MA_strategy/test_signal_variations.py
```

## 📊 Exemples de Configuration

### Pour analyser des crypto-monnaies :
```python
TICKERS = ['BTC-USD', 'ETH-USD', 'ADA-USD']
START_DATE = '2021-01-01'
END_DATE = '2024-12-31'
```

### Pour une analyse sur 5 ans :
```python
TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA']
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
```

### Pour des moyennes mobiles différentes :
```python
MA_PERIODS = [3, 7, 14, 30, 60, 120]  # Plus court terme
# ou
MA_PERIODS = [10, 25, 50, 100, 200, 300]  # Plus long terme
```

## 🔍 Validation Automatique

Le système valide automatiquement votre configuration :
- ✅ Format des dates
- ✅ Cohérence des paramètres  
- ✅ Existence des fichiers requis
- ❌ Affiche les erreurs clairement

## 📈 Résultats

Les résultats sont sauvegardés dans :
- **Backtests** : `MA_strategy/backtest_results/`
- **Variations** : `MA_strategy/signal_variations_test/`

Les noms de fichiers s'adaptent automatiquement à votre configuration !

## 🛠️ Avantages du Nouveau Système

1. **Plus de code cassé** : Changez la config, tout fonctionne
2. **Validation automatique** : Erreurs détectées avant l'exécution  
3. **Noms de fichiers cohérents** : Tout s'adapte automatiquement
4. **Pipeline orchestré** : Un seul script pour tout faire
5. **Flexibilité totale** : Tickers, dates, paramètres modifiables

## 🚨 Notes Importantes

1. **Téléchargement des données** : Assurez-vous d'avoir les données pour vos nouveaux tickers
2. **Compatibilité yfinance** : Vérifiez que vos tickers sont supportés
3. **Espace disque** : Plus de tickers = plus de fichiers générés
4. **Temps de calcul** : Plus de données = plus de temps de traitement

## 🆘 Dépannage

### "Fichier non trouvé"
- Vérifiez que les données existent dans `Data/src/`
- Utilisez `data_loader.py` pour télécharger les données

### "Erreur de configuration"  
- Le script affiche exactement quoi corriger
- Vérifiez le format des dates (YYYY-MM-DD)
- Vérifiez que TICKERS n'est pas vide

### "Import Error"
- Exécutez depuis le dossier racine du projet
- Vérifiez que `project_config.py` existe

## 📞 Support

Modifiez `project_config.py` et relancez ! 
Tout le reste se met à jour automatiquement. 🎉