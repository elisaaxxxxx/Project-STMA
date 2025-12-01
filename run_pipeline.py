#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline complet de stratégie de trading
=========================================================================

Ce script utilise la configuration centralisée pour :
1. Télécharger les données (optionnel)
2. Calculer les moyennes mobiles
3. Générer les signaux de trading
4. Effectuer les backtests
5. Tester les variations de signaux avec walk-forward

Usage:
    python run_pipeline.py --all          # Exécute tout le pipeline
    python run_pipeline.py --config       # Affiche la configuration
    python run_pipeline.py --ma           # Calcule seulement les moyennes mobiles
    python run_pipeline.py --signals      # Génère seulement les signaux
    python run_pipeline.py --backtest     # Backtest seulement
    python run_pipeline.py --variations   # Test des variations seulement

Pour modifier la configuration, éditez project_config.py
"""

import sys
import argparse
from pathlib import Path
import os

# Import de la configuration
from project_config import (TICKERS, START_DATE, END_DATE, print_config, 
                           validate_config, get_data_file_path)

def check_data_files():
    """Vérifie si les fichiers de données existent et propose de les télécharger."""
    missing_files = []
    missing_tickers = []
    
    for ticker in TICKERS:
        data_file = get_data_file_path(ticker)
        if not Path(data_file).exists():
            missing_files.append(data_file)
            missing_tickers.append(ticker)
    
    if missing_tickers:
        print(f"\n⚠️  DONNÉES MANQUANTES pour: {', '.join(missing_tickers)}")
        print("📥 Téléchargement automatique des données manquantes...")
        
        # Import et exécution automatique du data loader
        try:
            sys.path.append(str(Path(__file__).parent / "src"))
            import data_loader
            data_loader.main()
            print("✅ Données téléchargées avec succès!")
            return []  # Plus de fichiers manquants après téléchargement
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return missing_files
    
    return missing_files

def run_moving_averages():
    """Exécute le calcul des moyennes mobiles."""
    print("\n" + "="*60)
    print("ÉTAPE 1: Calcul des moyennes mobiles")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        import calculate_moving_averages
        calculate_moving_averages.main()
        return True
    except Exception as e:
        print(f"❌ Erreur lors du calcul des moyennes mobiles: {e}")
        return False

def run_signal_generation():
    """Exécute la génération des signaux."""
    print("\n" + "="*60)
    print("ÉTAPE 2: Génération des signaux de trading")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        import generate_signals
        generate_signals.main()
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la génération des signaux: {e}")
        return False

def run_backtest():
    """Exécute les backtests."""
    print("\n" + "="*60)
    print("ÉTAPE 3: Backtest des stratégies")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        import backtest_signal_strategy
        backtest_signal_strategy.main()
        return True
    except Exception as e:
        print(f"❌ Erreur lors du backtest: {e}")
        return False

def run_signal_variations():
    """Exécute les tests de variations de signaux."""
    print("\n" + "="*60)
    print("ÉTAPE 4: Test des variations de signaux (Walk-Forward)")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        import test_signal_variations
        test_signal_variations.main()
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test des variations: {e}")
        return False



def run_full_pipeline():
    """Exécute le pipeline complet."""
    print("🚀 DÉMARRAGE DU PIPELINE COMPLET")
    print_config()
    
    # Vérification et téléchargement automatique des fichiers de données manquants
    missing_files = check_data_files()
    if missing_files:
        print(f"\n❌ Impossible de télécharger certaines données:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n💡 Vérifiez votre connexion internet et les noms des tickers")
        return False
    
    # Exécution séquentielle
    steps = [
        ("Moyennes mobiles", run_moving_averages),
        ("Signaux", run_signal_generation), 
        ("Backtest", run_backtest),
        ("Variations", run_signal_variations)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Échec à l'étape: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"📊 Résultats disponibles dans:")
    print(f"  - Backtests: MA_strategy/backtest_results/")
    print(f"  - Variations: MA_strategy/signal_variations_test/")
    print("="*60)
    
    return True

def main():
    """Fonction principale avec gestion des arguments."""
    
    # Validation de base de la configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ ERREURS DE CONFIGURATION:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Corrigez les erreurs dans project_config.py")
        return 1
    
    parser = argparse.ArgumentParser(
        description="Pipeline de stratégie de trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run_pipeline.py --all          # Pipeline complet
  python run_pipeline.py --config       # Voir la configuration
  python run_pipeline.py --ma --signals # Moyennes mobiles + signaux
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Exécute tout le pipeline complet')
    parser.add_argument('--config', action='store_true', 
                       help='Affiche la configuration actuelle')
    parser.add_argument('--ma', action='store_true', 
                       help='Calcule les moyennes mobiles')
    parser.add_argument('--signals', action='store_true', 
                       help='Génère les signaux de trading')
    parser.add_argument('--backtest', action='store_true', 
                       help='Effectue les backtests')
    parser.add_argument('--variations', action='store_true', 
                       help='Test les variations de signaux')

    
    args = parser.parse_args()
    
    # Si aucun argument, afficher le menu interactif
    if not any(vars(args).values()):
        print("\n🎯 PIPELINE DE STRATÉGIE DE TRADING")
        print_config()
        print("Options disponibles:")
        print("1. Exécuter le pipeline complet")
        print("2. Calculer les moyennes mobiles")
        print("3. Générer les signaux")
        print("4. Exécuter les backtests")
        print("5. Tester les variations")
        print("6. Afficher la configuration")
        print("7. Quitter")
        
        while True:
            try:
                choice = input("\nChoisissez une option (1-7): ").strip()
                
                if choice == '1':
                    run_full_pipeline()
                    break
                elif choice == '2':
                    run_moving_averages()
                    break
                elif choice == '3':
                    run_signal_generation()
                    break
                elif choice == '4':
                    run_backtest()
                    break
                elif choice == '5':
                    run_signal_variations()
                    break
                elif choice == '6':
                    print_config()
                elif choice == '7':
                    print("Au revoir!")
                    break
                else:
                    print("❌ Option invalide, choisissez 1-7")
            
            except KeyboardInterrupt:
                print("\n\nAu revoir!")
                break
        
        return 0
    
    # Exécution selon les arguments
    success = True
    
    if args.config:
        print_config()
    
    if args.all:
        success = run_full_pipeline()
    else:
        # Pipeline traditionnel
        if args.ma:
            success &= run_moving_averages()
        if args.signals:
            success &= run_signal_generation()
        if args.backtest:
            success &= run_backtest()
        if args.variations:
            success &= run_signal_variations()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())