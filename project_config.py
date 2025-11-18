"""
Configuration centralisée pour le projet de stratégie de trading
================================================================

Modifiez les valeurs ci-dessous pour changer les tickers, dates, 
et autres paramètres du projet.

Tous les scripts utiliseront automatiquement ces paramètres.
"""

# ===== TICKERS À ANALYSER =====
# Liste des actions à analyser
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

# Exemples d'autres tickers intéressants à tester :
# Actions tech (FAANG+) :
# TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

# ETFs diversifiés :
# TICKERS = ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA', 'EFA', 'VWO']

# Secteurs spécifiques :
# TICKERS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI']  # Énergie, Finance, Tech, Santé, Industrie

# Matières premières :
# TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']  # Or, Argent, Pétrole, Gaz, Agriculture

# Actions défensives :
# TICKERS = ['JNJ', 'PG', 'KO', 'WMT', 'PFE']  # Consumer staples & healthcare

# Crypto (si supporté par yfinance) :
# TICKERS = ['BTC-USD', 'ETH-USD']

# ===== PÉRIODE D'ANALYSE =====
# Format: 'AAAA-MM-JJ'
START_DATE = '2000-01-01'
END_DATE = '2025-11-01'  # Jusqu'au 1er novembre 2025

# Exemples d'autres périodes :
# START_DATE = '2020-01-01'  # Dernières 5 années
# START_DATE = '2010-01-01'  # Dernières 15 années
# END_DATE = '2024-12-31'    # Jusqu'à fin 2024

# ===== PARAMÈTRES DES MOYENNES MOBILES =====
# Périodes des moyennes mobiles à calculer
MA_PERIODS = [5, 10, 20, 50, 100, 200]

# Comparaisons pour générer les signaux (court terme vs long terme)
MA_COMPARISONS = [
    {'short': 5, 'long': 20, 'name': 'Signal_5_20_short'},      # Court terme
    {'short': 10, 'long': 50, 'name': 'Signal_10_50_medium'},   # Moyen terme  
    {'short': 20, 'long': 100, 'name': 'Signal_20_100_long'},   # Long terme
    {'short': 50, 'long': 200, 'name': 'Signal_50_200_vlong'}   # Très long terme
]

# ===== PARAMÈTRES DE BACKTEST =====
# Coût de transaction par trade (en pourcentage)
TRANSACTION_COST = 0.001  # 0.1% par transaction

# Nombre de jours de trading par an (pour l'annualisation)
TRADING_DAYS_PER_YEAR = 252

# ===== PARAMÈTRES WALK-FORWARD =====
# Période d'entraînement en mois
TRAINING_MONTHS = 36  # 3 ans

# Période de test en mois
TEST_MONTHS = 6  # 6 mois

# ===== RÉPERTOIRES =====
# Nouvelle structure : séparer programmes et données
DATA_RAW_DIR = 'PROJECT/data/raw'                    # Données brutes (CSV téléchargés)
DATA_PROCESSED_DIR = 'PROJECT/data/processed'        # Données avec MA et signaux  
RESULTS_BACKTEST_DIR = 'PROJECT/data/results/backtest'     # Résultats des backtests
RESULTS_VARIATIONS_DIR = 'PROJECT/data/results/variations'  # Tests de variations

# Anciens noms pour compatibilité (DEPRECATED)
DATA_DIR = DATA_RAW_DIR
STRATEGY_DIR = DATA_PROCESSED_DIR
RESULTS_DIR = RESULTS_BACKTEST_DIR
VARIATIONS_DIR = RESULTS_VARIATIONS_DIR

# ===== FONCTIONS UTILITAIRES =====

def get_data_file_path(ticker, start_date=None, end_date=None):
    """Génère le chemin vers le fichier de données brutes."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_RAW_DIR}/{ticker}_{start_date}_{end_date}.csv"

def get_ma_file_path(ticker, start_date=None, end_date=None):
    """Génère le chemin vers le fichier avec moyennes mobiles."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_PROCESSED_DIR}/{ticker}_{start_date}_{end_date}_with_MAs.csv"

def get_signals_file_path(ticker, start_date=None, end_date=None):
    """Génère le chemin vers le fichier avec signaux."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{DATA_PROCESSED_DIR}/{ticker}_{start_date}_{end_date}_with_signals.csv"

def get_backtest_file_path(ticker, start_date=None, end_date=None):
    """Génère le chemin vers le fichier de résultats de backtest."""
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    return f"{RESULTS_BACKTEST_DIR}/{ticker}_{start_date}_{end_date}_backtest_results.csv"

def print_config():
    """Affiche la configuration actuelle."""
    print("=" * 60)
    print("CONFIGURATION DU PROJET")
    print("=" * 60)
    print(f"Tickers: {TICKERS}")
    print(f"Période: {START_DATE} à {END_DATE}")
    print(f"Moyennes mobiles: {MA_PERIODS}")
    print(f"Coût de transaction: {TRANSACTION_COST:.4f}")
    print(f"Walk-Forward: {TRAINING_MONTHS} mois training, {TEST_MONTHS} mois test")
    print("=" * 60)

def validate_config():
    """Valide la configuration."""
    errors = []
    
    if not TICKERS:
        errors.append("TICKERS ne peut pas être vide")
    
    try:
        from datetime import datetime
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
        if start_dt >= end_dt:
            errors.append("START_DATE doit être antérieure à END_DATE")
    except ValueError:
        errors.append("Format de date invalide (utilisez AAAA-MM-JJ)")
    
    if not MA_PERIODS or not all(isinstance(p, int) and p > 0 for p in MA_PERIODS):
        errors.append("MA_PERIODS doit contenir des entiers positifs")
    
    if not (0 <= TRANSACTION_COST <= 1):
        errors.append("TRANSACTION_COST doit être entre 0 et 1")
    
    return len(errors) == 0, errors

# Validation automatique à l'import
if __name__ == "__main__":
    print_config()
    is_valid, errors = validate_config()
    
    if is_valid:
        print("\n✅ Configuration valide !")
    else:
        print("\n❌ Erreurs de configuration :")
        for error in errors:
            print(f"  - {error}")