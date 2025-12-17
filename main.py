#!/usr/bin/env python3
"""
Pipeline V2 - Corrigé pour la nouvelle structure de dossiers
============================================================

Structure des dossiers:
data/
  ├── raw/          # Données téléchargées
  ├── processed/    # Données avec MA et signaux
  ├── results/      # Résultats backtests traditionnels
  └── ML/           # Données et résultats ML

Usage:
    python main.py --all          # Pipeline complet
    python main.py --traditional  # Pipeline traditionnel
    python main.py --ml           # Pipeline ML
    python main.py --config       # Voir config
"""

import sys
import argparse
import subprocess
from pathlib import Path
import os

# Import configuration
from project_config import (TICKERS, ALL_TICKERS, BENCHMARK_TICKER, 
                           START_DATE, END_DATE, print_config, validate_config)

def ensure_directories():
    """Crée tous les dossiers nécessaires."""
    dirs = [
        'data/SRC/raw',
        'data/SRC/processed', 
        'data/SRC/results/backtest',
        'data/SRC/results/variations',
        'data/ML',
        'data/ML/backtest_results',
        'data/ML/regularization_analysis',
        'ML/models'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Tous les dossiers créés")

def check_data_files():
    """Vérifie si les fichiers raw existent."""
    missing = []
    for ticker in ALL_TICKERS:  # Vérifie tous les tickers incluant benchmark
        file = Path(f"data/SRC/raw/{ticker}_{START_DATE}_{END_DATE}.csv")
        if not file.exists():
            missing.append(ticker)
    
    if missing:
        print(f"\n⚠️  Données manquantes: {', '.join(missing)}")
        print("📥 Téléchargement automatique...")
        
        try:
            result = subprocess.run(
                [sys.executable, "src/data_loader.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Données téléchargées!")
                return []
            else:
                print(f"❌ Erreur: {result.stderr}")
                return missing
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return missing
    
    return []

def run_traditional_pipeline():
    """Exécute le pipeline traditionnel."""
    print("\n" + "="*70)
    print("🚀 PIPELINE TRADITIONNEL")
    print("="*70)
    print_config()
    
    # Vérifier les données
    ensure_directories()
    missing = check_data_files()
    if missing:
        print(f"\n❌ Impossible de télécharger: {', '.join(missing)}")
        return False
    
    # Étapes
    scripts = [
        ("Moyennes mobiles", "src/calculate_moving_averages.py"),
        ("Signaux", "src/generate_signals.py"),
        ("Backtest", "src/backtest_signal_strategy.py"),
        ("Variations", "src/test_signal_variations.py")
    ]
    
    for name, script in scripts:
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print("="*70)
        
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False  # Afficher la sortie en direct
        )
        
        if result.returncode != 0:
            print(f"\n❌ Échec: {name}")
            return False
    
    print("\n" + "="*70)
    print("✅ PIPELINE TRADITIONNEL TERMINÉ!")
    print("="*80 + "\n")
    print("📊 Résultats:")
    print("  • data/SRC/processed/ - Données avec MA et signaux")
    print("  • data/SRC/results/backtest/ - Backtests")
    print("  • data/SRC/results/variations/ - Walk-forward")
    print("="*70)
    
    return True

def run_ml_pipeline():
    """Exécute le pipeline ML."""
    print("\n" + "="*70)
    print("🤖 PIPELINE MACHINE LEARNING")
    print("="*70)
    print_config()
    
    ensure_directories()
    
    # Vérifier que les données processed existent
    print("\n📋 Vérification des données processed...")
    missing = []
    for ticker in TICKERS:
        file = Path(f"data/SRC/processed/{ticker}_{START_DATE}_{END_DATE}_with_signals.csv")
        if not file.exists():
            missing.append(ticker)
    
    if missing:
        print(f"⚠️  Données processed manquantes: {', '.join(missing)}")
        print("📊 Exécution du pipeline traditionnel d'abord...")
        if not run_traditional_pipeline():
            return False
    
    # Étapes ML
    ml_steps = [
        ("Création datasets ML", "ML/create_ml_data.py", None),
        ("Entraînement modèles", "ML/train_regression_model.py", None),
        ("Analyse régularisation", "ML/analyze_lasso_regularization.py", ["--n-alphas", "50"]),
        ("Backtest ML", "ML/backtest_ml_strategy.py", ["--model", "lasso_regression"])
    ]
    
    for name, script, extra_args in ml_steps:
        print(f"\n{'='*70}")
        print(f"🤖 {name}")
        print("="*70)
        
        for ticker in TICKERS:
            print(f"\n📊 {ticker}...")
            
            cmd = [sys.executable, script, "--ticker", ticker]
            if extra_args:
                cmd.extend(extra_args)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Erreur pour {ticker}:")
                print(result.stderr)
                return False
            else:
                # Afficher juste les lignes importantes
                for line in result.stdout.split('\n'):
                    if any(x in line for x in ['✅', '✓', 'CAGR', 'Sharpe', 'Test R²', 'Best']):
                        print(line)
    
    print("\n" + "="*70)
    print("✅ PIPELINE ML TERMINÉ!")
    print("="*70)
    print("📊 Résultats ML:")
    print("  • data/ML/ - Datasets ML")
    print("  • ML/models/ - Modèles entraînés")
    print("  • data/ML/regularization_analysis/ - Analyses")
    print("  • data/ML/backtest_results/ - Backtests ML")
    print("="*70)
    
    return True

def run_full_pipeline():
    """Pipeline complet."""
    print("\n" + "="*80)
    print("🚀 PIPELINE COMPLET (TRADITIONAL + ML)")
    print("="*80)
    
    # Phase 1: Traditional
    print("\n" + "="*80)
    print("PHASE 1: PIPELINE TRADITIONNEL")
    print("="*80)
    
    if not run_traditional_pipeline():
        print("\n❌ Échec phase 1")
        return False
    
    # Phase 2: ML
    print("\n" + "="*80)
    print("PHASE 2: PIPELINE ML")
    print("="*80)
    
    if not run_ml_pipeline():
        print("\n❌ Échec phase 2")
        return False
    
    # Résumé final
    print("\n" + "="*80)
    print("✅✅✅ PIPELINE COMPLET TERMINÉ! ✅✅✅")
    print("="*80)
    print(f"\n📊 RÉSUMÉ:")
    print(f"\n  PIPELINE TRADITIONNEL:")
    print(f"    • Données: data/SRC/processed/")
    print(f"    • Backtests: data/SRC/results/backtest/")
    print(f"    • Walk-forward: data/SRC/results/variations/")
    print(f"\n  PIPELINE ML:")
    print(f"    • Datasets: data/ML/")
    print(f"    • Modèles: ML/models/")
    print(f"    • Analyses: data/ML/regularization_analysis/")
    print(f"    • Backtests ML: data/ML/backtest_results/")
    print(f"  TICKERS: {', '.join(TICKERS)}")
    print(f"  BENCHMARK: {BENCHMARK_TICKER}")
    print(f"  PÉRIODE: {START_DATE} → {END_DATE}")
    print("="*80 + "\n")
    
    # Show comprehensive results
    show_results()
    
    return True

def show_results():
    """Display comprehensive results using show_results.py."""
    
    print("\n" + "="*80)
    print("📊 AFFICHAGE DES RÉSULTATS COMPLETS")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            ['python3', 'show_results.py'],
            check=True,
            capture_output=False  # Show output directly
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'affichage des résultats: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def main():
    """Fonction principale."""
    
    # Validation config
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ ERREURS DE CONFIGURATION:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    parser = argparse.ArgumentParser(
        description="Pipeline V2 - Structure corrigée",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Pipeline complet')
    parser.add_argument('--traditional', action='store_true',
                       help='Pipeline traditionnel')
    parser.add_argument('--ml', action='store_true',
                       help='Pipeline ML')
    parser.add_argument('--config', action='store_true',
                       help='Afficher config')
    
    args = parser.parse_args()
    
    # Si pas d'arguments
    if not any(vars(args).values()):
        print("\n" + "="*70)
        print("🎯 PIPELINE V2 - STRUCTURE CORRIGÉE")
        print("="*70)
        print_config()
        print("\n📋 OPTIONS:")
        print("  1. Pipeline complet (Traditional + ML)")
        print("  2. Pipeline traditionnel seulement")
        print("  3. Pipeline ML seulement")
        print("  4. Afficher configuration")
        print("  5. Quitter")
        
        try:
            choice = input("\n👉 Choisir (1-5): ").strip()
            
            if choice == '1':
                run_full_pipeline()
            elif choice == '2':
                run_traditional_pipeline()
            elif choice == '3':
                run_ml_pipeline()
            elif choice == '4':
                print_config()
            elif choice == '5':
                print("👋 Au revoir!")
            else:
                print("❌ Option invalide")
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
        
        return 0
    
    # Exécution selon arguments
    if args.config:
        print_config()
    
    success = True
    if args.all:
        success = run_full_pipeline()
    elif args.traditional:
        success = run_traditional_pipeline()
    elif args.ml:
        success = run_ml_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
