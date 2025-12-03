#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline complet de stratégie de trading
=========================================================================

Ce script utilise la configuration centralisée pour :

TRADITIONAL PIPELINE:
1. Télécharger les données (optionnel)
2. Calculer les moyennes mobiles
3. Générer les signaux de trading
4. Effectuer les backtests
5. Tester les variations de signaux avec walk-forward

MACHINE LEARNING PIPELINE:
6. Créer les datasets ML
7. Entraîner les modèles de régression
8. Analyser la régularisation Lasso
9. Backtester les stratégies ML

Usage:
    python run_pipeline.py --all          # Pipeline complet (Traditional + ML)
    python run_pipeline.py --traditional  # Pipeline traditionnel seulement
    python run_pipeline.py --ml           # Pipeline ML seulement
    python run_pipeline.py --config       # Affiche la configuration
    
    # Traditional steps:
    python run_pipeline.py --ma           # Moyennes mobiles
    python run_pipeline.py --signals      # Signaux
    python run_pipeline.py --backtest     # Backtest
    python run_pipeline.py --variations   # Variations
    
    # ML steps:
    python run_pipeline.py --ml-data      # Créer datasets ML
    python run_pipeline.py --ml-train     # Entraîner modèles
    python run_pipeline.py --ml-analyze   # Analyser régularisation
    python run_pipeline.py --ml-backtest  # Backtest ML

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
        
        # Exécution du data loader via subprocess
        try:
            import subprocess
            data_loader_path = Path(__file__).parent / "src" / "data_loader.py"
            result = subprocess.run(
                [sys.executable, str(data_loader_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Données téléchargées avec succès!")
                return []  # Plus de fichiers manquants après téléchargement
            else:
                print(f"❌ Erreur lors du téléchargement:")
                print(result.stderr)
                return missing_files
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

# ============================================================================
# MACHINE LEARNING PIPELINE FUNCTIONS
# ============================================================================

def run_ml_data_creation():
    """Crée les datasets ML pour tous les tickers."""
    print("\n" + "="*60)
    print("ÉTAPE ML-1: Création des datasets ML")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "ML"))
        import create_ml_data
        
        # Créer les données ML pour chaque ticker
        for ticker in TICKERS:
            print(f"\n📊 Création dataset ML pour {ticker}...")
            # Simuler les arguments en utilisant sys.argv
            original_argv = sys.argv.copy()
            sys.argv = ['create_ml_data.py', '--ticker', ticker]
            try:
                create_ml_data.main()
            except SystemExit:
                pass
            sys.argv = original_argv
        
        print("\n✅ Datasets ML créés pour tous les tickers!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création des datasets ML: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_ml_training():
    """Entraîne les modèles ML pour tous les tickers."""
    print("\n" + "="*60)
    print("ÉTAPE ML-2: Entraînement des modèles ML")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "ML"))
        import train_regression_model
        
        # Entraîner les modèles pour chaque ticker
        for ticker in TICKERS:
            print(f"\n🤖 Entraînement des modèles pour {ticker}...")
            original_argv = sys.argv.copy()
            sys.argv = ['train_regression_model.py', '--ticker', ticker]
            try:
                train_regression_model.main()
            except SystemExit:
                pass
            sys.argv = original_argv
        
        print("\n✅ Modèles ML entraînés pour tous les tickers!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement des modèles: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_ml_regularization_analysis():
    """Analyse la régularisation Lasso pour tous les tickers."""
    print("\n" + "="*60)
    print("ÉTAPE ML-3: Analyse de régularisation Lasso")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "ML"))
        import analyze_lasso_regularization
        
        # Analyser la régularisation pour chaque ticker
        for ticker in TICKERS:
            print(f"\n📊 Analyse de régularisation pour {ticker}...")
            original_argv = sys.argv.copy()
            sys.argv = ['analyze_lasso_regularization.py', '--ticker', ticker, '--n-alphas', '50']
            try:
                analyze_lasso_regularization.main()
            except SystemExit:
                pass
            sys.argv = original_argv
        
        print("\n✅ Analyses de régularisation terminées pour tous les tickers!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de régularisation: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_ml_backtest():
    """Backtest les stratégies ML pour tous les tickers."""
    print("\n" + "="*60)
    print("ÉTAPE ML-4: Backtest des stratégies ML")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "ML"))
        import backtest_ml_strategy
        
        # Backtester pour chaque ticker
        for ticker in TICKERS:
            print(f"\n📈 Backtest ML pour {ticker}...")
            original_argv = sys.argv.copy()
            sys.argv = ['backtest_ml_strategy.py', '--ticker', ticker, '--model', 'lasso_regression']
            try:
                backtest_ml_strategy.main()
            except SystemExit:
                pass
            sys.argv = original_argv
        
        print("\n✅ Backtests ML terminés pour tous les tickers!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du backtest ML: {e}")
        import traceback
        traceback.print_exc()
        return False



def run_traditional_pipeline():
    """Exécute le pipeline traditionnel complet."""
    print("\n🚀 DÉMARRAGE DU PIPELINE TRADITIONNEL")
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
    print("✅ PIPELINE TRADITIONNEL TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"📊 Résultats disponibles dans:")
    print(f"  - Données: data/SRC/processed/")
    print(f"  - Backtests: data/SRC/results/backtest/")
    print(f"  - Variations: data/SRC/results/variations/")
    print("="*60)
    
    return True

def run_ml_pipeline():
    """Exécute le pipeline Machine Learning complet."""
    print("\n🤖 DÉMARRAGE DU PIPELINE MACHINE LEARNING")
    print_config()
    
    # S'assurer que les données processed existent
    print("\n📋 Vérification des données processed...")
    processed_files_exist = True
    for ticker in TICKERS:
        signals_file = Path(f"data/SRC/processed/{ticker}_{START_DATE}_{END_DATE}_with_signals.csv")
        if not signals_file.exists():
            print(f"⚠️  Fichier manquant: {signals_file}")
            processed_files_exist = False
    
    if not processed_files_exist:
        print("\n📊 Exécution du pipeline traditionnel d'abord...")
        if not run_traditional_pipeline():
            print("❌ Impossible de continuer sans données processed")
            return False
    
    # Exécution séquentielle du pipeline ML
    ml_steps = [
        ("Création datasets ML", run_ml_data_creation),
        ("Entraînement modèles", run_ml_training),
        ("Analyse régularisation", run_ml_regularization_analysis),
        ("Backtest ML", run_ml_backtest)
    ]
    
    for step_name, step_func in ml_steps:
        if not step_func():
            print(f"\n❌ Échec à l'étape ML: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("✅ PIPELINE MACHINE LEARNING TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"📊 Résultats ML disponibles dans:")
    print(f"  - Datasets: data/ML/")
    print(f"  - Modèles: ML/models/")
    print(f"  - Analyses: data/ML/regularization_analysis/")
    print(f"  - Backtests: data/ML/backtest_results/")
    print("="*60)
    
    return True

def run_full_pipeline():
    """Exécute le pipeline complet (Traditional + ML)."""
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DU PIPELINE COMPLET (TRADITIONAL + MACHINE LEARNING)")
    print("="*80)
    print_config()
    
    # Étape 1: Pipeline traditionnel
    print("\n" + "="*80)
    print("PHASE 1: PIPELINE TRADITIONNEL")
    print("="*80)
    
    if not run_traditional_pipeline():
        print("\n❌ Échec du pipeline traditionnel")
        return False
    
    # Étape 2: Pipeline ML
    print("\n" + "="*80)
    print("PHASE 2: PIPELINE MACHINE LEARNING")
    print("="*80)
    
    if not run_ml_pipeline():
        print("\n❌ Échec du pipeline ML")
        return False
    
    # Résumé final
    print("\n" + "="*80)
    print("✅✅✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS! ✅✅✅")
    print("="*80)
    print(f"\n📊 RÉSUMÉ DES RÉSULTATS:")
    print(f"\n  TRADITIONAL PIPELINE:")
    print(f"    • Données processed: data/SRC/processed/")
    print(f"    • Backtests: data/SRC/results/backtest/")
    print(f"    • Walk-forward: data/SRC/results/variations/")
    print(f"\n  MACHINE LEARNING PIPELINE:")
    print(f"    • Datasets ML: data/ML/")
    print(f"    • Modèles entraînés: ML/models/")
    print(f"    • Analyses régularisation: data/ML/regularization_analysis/")
    print(f"    • Backtests ML: data/ML/backtest_results/")
    print(f"\n  TICKERS TRAITÉS: {', '.join(TICKERS)}")
    print(f"  PÉRIODE: {START_DATE} → {END_DATE}")
    print("="*80 + "\n")
    
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
        description="Pipeline de stratégie de trading (Traditional + ML)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run_pipeline.py --all              # Pipeline complet (Traditional + ML)
  python run_pipeline.py --traditional      # Pipeline traditionnel seulement
  python run_pipeline.py --ml               # Pipeline ML seulement
  python run_pipeline.py --config           # Voir la configuration
  
  # Traditional steps:
  python run_pipeline.py --ma --signals     # Moyennes mobiles + signaux
  
  # ML steps:
  python run_pipeline.py --ml-data --ml-train   # Créer data + entraîner
        """
    )
    
    # Pipeline complet
    parser.add_argument('--all', action='store_true', 
                       help='Pipeline complet (Traditional + ML)')
    parser.add_argument('--traditional', action='store_true',
                       help='Pipeline traditionnel seulement')
    parser.add_argument('--ml', action='store_true',
                       help='Pipeline Machine Learning seulement')
    parser.add_argument('--config', action='store_true', 
                       help='Affiche la configuration actuelle')
    
    # Traditional pipeline steps
    parser.add_argument('--ma', action='store_true', 
                       help='Calcule les moyennes mobiles')
    parser.add_argument('--signals', action='store_true', 
                       help='Génère les signaux de trading')
    parser.add_argument('--backtest', action='store_true', 
                       help='Effectue les backtests')
    parser.add_argument('--variations', action='store_true', 
                       help='Test les variations de signaux')
    
    # ML pipeline steps
    parser.add_argument('--ml-data', action='store_true',
                       help='Crée les datasets ML')
    parser.add_argument('--ml-train', action='store_true',
                       help='Entraîne les modèles ML')
    parser.add_argument('--ml-analyze', action='store_true',
                       help='Analyse la régularisation Lasso')
    parser.add_argument('--ml-backtest', action='store_true',
                       help='Backtest les stratégies ML')
    
    args = parser.parse_args()
    
    # Si aucun argument, afficher le menu interactif
    if not any(vars(args).values()):
        print("\n" + "="*70)
        print("🎯 PIPELINE DE STRATÉGIE DE TRADING (TRADITIONAL + ML)")
        print("="*70)
        print_config()
        print("\n📋 OPTIONS DISPONIBLES:")
        print("\n  PIPELINES COMPLETS:")
        print("    1. 🚀 Exécuter le pipeline COMPLET (Traditional + ML)")
        print("    2. 📊 Pipeline TRADITIONNEL seulement")
        print("    3. 🤖 Pipeline MACHINE LEARNING seulement")
        print("\n  ÉTAPES TRADITIONNELLES:")
        print("    4. Calculer les moyennes mobiles")
        print("    5. Générer les signaux")
        print("    6. Exécuter les backtests")
        print("    7. Tester les variations (walk-forward)")
        print("\n  ÉTAPES MACHINE LEARNING:")
        print("    8. Créer les datasets ML")
        print("    9. Entraîner les modèles ML")
        print("   10. Analyser la régularisation Lasso")
        print("   11. Backtest des stratégies ML")
        print("\n  AUTRES:")
        print("   12. Afficher la configuration")
        print("   13. Quitter")
        print("="*70)
        
        while True:
            try:
                choice = input("\n👉 Choisissez une option (1-13): ").strip()
                
                if choice == '1':
                    run_full_pipeline()
                    break
                elif choice == '2':
                    run_traditional_pipeline()
                    break
                elif choice == '3':
                    run_ml_pipeline()
                    break
                elif choice == '4':
                    run_moving_averages()
                    break
                elif choice == '5':
                    run_signal_generation()
                    break
                elif choice == '6':
                    run_backtest()
                    break
                elif choice == '7':
                    run_signal_variations()
                    break
                elif choice == '8':
                    run_ml_data_creation()
                    break
                elif choice == '9':
                    run_ml_training()
                    break
                elif choice == '10':
                    run_ml_regularization_analysis()
                    break
                elif choice == '11':
                    run_ml_backtest()
                    break
                elif choice == '12':
                    print_config()
                elif choice == '13':
                    print("👋 Au revoir!")
                    break
                else:
                    print("❌ Option invalide, choisissez 1-13")
            
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
        
        return 0
    
    # Exécution selon les arguments
    success = True
    
    if args.config:
        print_config()
    
    if args.all:
        success = run_full_pipeline()
    elif args.traditional:
        success = run_traditional_pipeline()
    elif args.ml:
        success = run_ml_pipeline()
    else:
        # Exécution d'étapes individuelles
        if args.ma:
            success &= run_moving_averages()
        if args.signals:
            success &= run_signal_generation()
        if args.backtest:
            success &= run_backtest()
        if args.variations:
            success &= run_signal_variations()
        if args.ml_data:
            success &= run_ml_data_creation()
        if args.ml_train:
            success &= run_ml_training()
        if args.ml_analyze:
            success &= run_ml_regularization_analysis()
        if args.ml_backtest:
            success &= run_ml_backtest()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())