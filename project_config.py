"""
Configuration centralisée pour le projet de stratégie de trading
================================================================

Modifiez les valeurs ci-dessous pour changer les tickers, dates, 
et autres paramètres du projet.

Tous les scripts utiliseront automatiquement ces paramètres.
"""

# ===== TICKERS À ANALYSER =====
# Tickers à trader (changez cette liste selon vos besoins)
TICKERS = [
    # Tech (meilleurs performers ML)
    'AAPL',   # 📱 Apple - ML +2.19% vs B&H
    'NVDA',   # 🎮 Nvidia - ML +21.49% vs B&H
    
    # Finance (stable, prévisible)
    'JPM',    # 🏦 JP Morgan - Banque leader
    'BAC',    # 🏦 Bank of America
    
    # Consumer Staples (défensif, stable)
    'PG',     # 🧼 Procter & Gamble - Consumer goods
    'KO',     # 🥤 Coca-Cola - Beverages
    
    # Healthcare (croissance stable)
    'JNJ',    # � Johnson & Johnson - Pharma
]

# SPY comme benchmark uniquement (pour features ML)
BENCHMARK_TICKER = 'SPY'  # 📊 S&P 500 ETF - Benchmark uniquement

# Liste complète (tickers + benchmark) pour téléchargement des données
ALL_TICKERS = TICKERS + [BENCHMARK_TICKER]

# Caractéristiques de chaque ticker:
# - AAPL: Tech leader, forte croissance, high volatility
# - NVDA: Semiconducteur, très forte croissance, très volatile (AI boom)
# - JPM: Banque, cyclique, corrélé aux taux d'intérêt
# - JNJ:  Pharma/Healthcare, défensif, faible volatilité
# - XOM:  Énergie, cyclique, corrélé au pétrole

# Note: SPY est utilisé uniquement comme BENCHMARK (dans les features ML)
# mais n'est PAS tradé directement

# Exemples d'autres tickers intéressants à tester :
# Actions tech (FAANG+) :
# TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', ]

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
# Utilise des chemins ABSOLUS basés sur l'emplacement de ce fichier
# Cela garantit que les données sont toujours créées au bon endroit,
# peu importe d'où on lance le script
import os
from pathlib import Path

# Trouve le dossier racine du projet (là où se trouve ce fichier)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Structure organisée : SRC pour pipeline traditionnel, ML pour machine learning
DATA_RAW_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'raw')                    # Données brutes (CSV téléchargés)
DATA_PROCESSED_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'processed')        # Données avec MA et signaux  
RESULTS_BACKTEST_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'results' / 'backtest')     # Résultats des backtests
RESULTS_VARIATIONS_DIR = str(PROJECT_ROOT / 'data' / 'SRC' / 'results' / 'variations')  # Tests de variations

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

# ===== FONCTIONS DE GESTION DE CONFIGURATION =====

def update_tickers(new_tickers):
    """Met à jour la liste des tickers dans le fichier de configuration."""
    import os
    import shutil
    
    print(f"🔄 Mise à jour des tickers: {new_tickers}")
    
    # Lire le fichier
    with open('project_config.py', 'r') as f:
        content = f.read()
    
    # Construire la nouvelle ligne TICKERS
    tickers_list = [f"'{ticker.strip()}'" for ticker in new_tickers]
    new_tickers_line = f"TICKERS = [{', '.join(tickers_list)}]"
    
    # Remplacer la ligne TICKERS
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('TICKERS = [') and not line.strip().startswith('#'):
            lines[i] = new_tickers_line
            break
    
    # Sauvegarder
    with open('project_config.py', 'w') as f:
        f.write('\n'.join(lines))
    
    # Nettoyer le cache
    clear_cache()
    
    # Vérifier quelles données manquent et proposer le téléchargement
    check_and_download_missing_data(new_tickers)
    
    print("✅ Tickers mis à jour!")

def check_and_download_missing_data(tickers):
    """Vérifie et télécharge automatiquement les données manquantes."""
    import os
    import sys
    from pathlib import Path
    
    missing_tickers = []
    
    # Vérifier quels fichiers manquent
    for ticker in tickers:
        data_file = f"data/raw/{ticker}_{START_DATE}_{END_DATE}.csv"
        if not os.path.exists(data_file):
            missing_tickers.append(ticker)
    
    if missing_tickers:
        print(f"\n📥 Données manquantes pour: {', '.join(missing_tickers)}")
        print("🔄 Téléchargement automatique en cours...")
        
        try:
            # Lancer le data_loader via subprocess pour éviter les problèmes d'import
            import subprocess
            result = subprocess.run([
                sys.executable, 'src/data_loader.py'
            ], cwd='.', capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Données téléchargées avec succès!")
                
                # Traitement automatique complet pour les nouvelles données
                print("🔄 Traitement automatique des nouvelles données...")
                
                # Calcul des moyennes mobiles
                ma_result = subprocess.run([
                    sys.executable, 'run_pipeline.py', '--ma'
                ], cwd='.', capture_output=True, text=True)
                
                if ma_result.returncode == 0:
                    print("✅ Moyennes mobiles calculées!")
                    
                    # Génération des signaux
                    signals_result = subprocess.run([
                        sys.executable, 'run_pipeline.py', '--signals'
                    ], cwd='.', capture_output=True, text=True)
                    
                    if signals_result.returncode == 0:
                        print("✅ Signaux générés!")
                        print("🎉 Nouvelles données complètement traitées!")
                    else:
                        print("⚠️  Erreur génération signaux, mais données téléchargées")
                else:
                    print("⚠️  Erreur calcul MA, mais données téléchargées")
                    
            else:
                print(f"❌ Erreur lors du téléchargement: {result.stderr}")
                print("💡 Vous pouvez télécharger manuellement avec: python src/data_loader.py")
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            print("💡 Vous pouvez télécharger manuellement avec: python src/data_loader.py")
    else:
        print("✅ Toutes les données sont disponibles!")

def update_dates(start_date, end_date):
    """Met à jour les dates dans le fichier de configuration."""
    print(f"🔄 Mise à jour des dates: {start_date} → {end_date}")
    
    # Lire le fichier
    with open('project_config.py', 'r') as f:
        content = f.read()
    
    # Remplacer les dates
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('START_DATE = ') and not line.strip().startswith('#'):
            lines[i] = f"START_DATE = '{start_date}'"
        elif line.strip().startswith('END_DATE = ') and not line.strip().startswith('#'):
            lines[i] = f"END_DATE = '{end_date}'"
    
    # Sauvegarder
    with open('project_config.py', 'w') as f:
        f.write('\n'.join(lines))
    
    # Nettoyer le cache
    clear_cache()
    print("✅ Dates mises à jour!")

def clear_cache():
    """Nettoie les caches Python pour forcer le rechargement."""
    import os
    import shutil
    import sys
    
    print("🧹 Nettoyage des caches Python...")
    
    # Supprimer __pycache__
    cache_dirs = ['__pycache__', 'src/__pycache__']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"   Supprimé: {cache_dir}")
    
    # Supprimer le module des modules chargés
    if 'project_config' in sys.modules:
        del sys.modules['project_config']
        print("   Module project_config rechargé")

def manage_config():
    """Interface interactive pour gérer la configuration."""
    import sys
    
    if len(sys.argv) == 1:
        # Mode interactif
        print("\n🎛️  GESTIONNAIRE DE CONFIGURATION")
        print("="*50)
        print("1. Afficher la configuration actuelle")
        print("2. Modifier les tickers")
        print("3. Modifier les dates")
        print("4. Nettoyer les caches")
        print("5. Quitter")
        
        while True:
            try:
                choice = input("\nChoisissez une option (1-5): ").strip()
                
                if choice == '1':
                    print_config()
                elif choice == '2':
                    current_tickers = ', '.join(TICKERS)
                    print(f"Tickers actuels: {current_tickers}")
                    new_tickers = input("Nouveaux tickers (séparés par virgules): ").strip()
                    if new_tickers:
                        tickers = [t.strip().upper() for t in new_tickers.split(',')]
                        update_tickers(tickers)
                elif choice == '3':
                    print(f"Dates actuelles: {START_DATE} → {END_DATE}")
                    start = input("Nouvelle date de début (AAAA-MM-JJ): ").strip()
                    end = input("Nouvelle date de fin (AAAA-MM-JJ): ").strip()
                    if start and end:
                        update_dates(start, end)
                elif choice == '4':
                    clear_cache()
                elif choice == '5':
                    print("Au revoir!")
                    break
                else:
                    print("❌ Option invalide, choisissez 1-5")
            
            except KeyboardInterrupt:
                print("\n\nAu revoir!")
                break
    else:
        # Mode ligne de commande
        import argparse
        
        parser = argparse.ArgumentParser(description="Gestion de la configuration")
        parser.add_argument('--show', action='store_true', help='Affiche la configuration')
        parser.add_argument('--tickers', type=str, help='Nouveaux tickers (ex: AAPL,MSFT,SPY)')
        parser.add_argument('--dates', nargs=2, help='Nouvelles dates (START END)')
        parser.add_argument('--clear', action='store_true', help='Nettoie les caches')
        
        args = parser.parse_args()
        
        if args.show:
            print_config()
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(',')]
            update_tickers(tickers)
        if args.dates:
            update_dates(args.dates[0], args.dates[1])
        if args.clear:
            clear_cache()

# Validation automatique à l'import
if __name__ == "__main__":
    manage_config()