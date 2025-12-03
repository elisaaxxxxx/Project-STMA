#!/usr/bin/env python3
"""
Show All Results - Display Summary of Current Tickers
=====================================================

Displays a comprehensive summary of all results for tickers
configured in project_config.py

Usage:
    python show_results.py
"""

import sys
import pandas as pd
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent))
from project_config import TICKERS, BENCHMARK_TICKER, START_DATE, END_DATE

def load_traditional_results(ticker):
    """Load traditional strategy results - biased, walk-forward, and buy & hold."""
    
    # Walk-forward results
    wf_file = Path(f"data/SRC/results/variations/{ticker}_signal_variations_comparison.csv")
    
    if wf_file.exists():
        df = pd.read_csv(wf_file)
        
        # Walk-forward (no look-ahead)
        wf_row = df[df['Strategy'] == 'Walk-Forward (No Look-Ahead)'].iloc[0]
        
        # Buy & Hold
        bh_row = df[df['Strategy'] == 'Buy & Hold'].iloc[0]
        
        # Best traditional (with look-ahead bias) - exclude Walk-Forward and Buy & Hold
        exclude_strategies = ['Walk-Forward (No Look-Ahead)', 'Buy & Hold']
        best_trad_row = df[~df['Strategy'].isin(exclude_strategies)].sort_values('Sharpe', ascending=False).iloc[0]
        
        return {
            'ticker': ticker,
            'walk_forward': {
                'method': 'Walk-Forward',
                'cagr': wf_row['CAGR'] * 100,
                'sharpe': wf_row['Sharpe'],
                'maxdd': wf_row['MaxDD'] * 100,
                'strategy': 'Variable'
            },
            'best_biased': {
                'method': 'Best Traditional (Biased)',
                'cagr': best_trad_row['CAGR'] * 100,
                'sharpe': best_trad_row['Sharpe'],
                'maxdd': best_trad_row['MaxDD'] * 100,
                'strategy': best_trad_row['Strategy']
            },
            'buy_hold': {
                'method': 'Buy & Hold',
                'cagr': bh_row['CAGR'] * 100,
                'sharpe': bh_row['Sharpe'],
                'maxdd': bh_row['MaxDD'] * 100,
                'strategy': 'Passive'
            }
        }
    else:
        return None

def load_ml_results(ticker):
    """Load ML strategy results."""
    
    backtest_file = Path(f"data/ML/backtest_results/{ticker}_lasso_regression_backtest_results.csv")
    
    if backtest_file.exists():
        df = pd.read_csv(backtest_file)
        
        # Calculate metrics from equity curve
        final_equity = df['equity'].iloc[-1]
        initial_equity = 1.0
        days = len(df)
        years = days / 252
        
        # CAGR
        cagr = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
        
        # Sharpe (approximate from daily returns)
        daily_returns = df['strategy_return'].dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
        
        # Max Drawdown
        cummax = df['equity'].cummax()
        drawdown = (df['equity'] - cummax) / cummax
        maxdd = drawdown.min() * 100
        
        return {
            'method': 'ML (Lasso)',
            'cagr': cagr,
            'sharpe': sharpe,
            'maxdd': maxdd,
            'strategy': 'Auto-selection MA pairs'
        }
    else:
        return None

def load_regularization_results(ticker):
    """Load regularization analysis results."""
    
    reg_file = Path(f"data/ML/regularization_analysis/{ticker}_lasso_regularization_analysis.csv")
    
    if reg_file.exists():
        df = pd.read_csv(reg_file)
        best_idx = df['test_r2'].idxmax()
        best_row = df.iloc[best_idx]
        
        return {
            'ticker': ticker,
            'alpha': best_row['alpha'],
            'test_r2': best_row['test_r2'],
            'train_r2': best_row['train_r2'],
            'n_features': best_row['n_nonzero_coefs']
        }
    else:
        return None

def print_header():
    """Print header."""
    print("\n" + "="*100)
    print("📊 RÉSULTATS COMPLETS - STRATÉGIES DE TRADING")
    print("="*100)
    print(f"\n📋 Configuration actuelle:")
    print(f"   • Tickers: {', '.join(TICKERS)}")
    print(f"   • Benchmark: {BENCHMARK_TICKER}")
    print(f"   • Période: {START_DATE} → {END_DATE}")
    print("="*100)

def print_main_comparison(results):
    """Print main comparison of 4 methods."""
    
    if not results:
        print("\n⚠️  Aucun résultat trouvé")
        return
    
    print("\n" + "="*100)
    print("🏆 COMPARAISON DES 4 MÉTHODES")
    print("="*100)
    print("\nCette comparaison montre:")
    print("  1️⃣  Buy & Hold - Benchmark passif (acheter et garder)")
    print("  2️⃣  Meilleure BIAISÉE (look-ahead) - Performance artificielle")
    print("  3️⃣  Walk-Forward (SANS biais) - Performance réaliste")
    print("  4️⃣  Machine Learning - Sélection automatique")
    print("\n" + "="*100)
    
    for r in results:
        ticker = r['ticker']
        print(f"\n📊 {ticker}")
        print("-"*100)
        print(f"{'Méthode':<35} {'CAGR':>10} {'Sharpe':>10} {'Max DD':>10}")
        print("-"*100)
        
        # Buy & Hold
        bh = r['buy_hold']
        print(f"{'1️⃣  ' + bh['method']:<35} {bh['cagr']:>9.2f}% {bh['sharpe']:>10.2f} {bh['maxdd']:>9.2f}%")
        
        # Best biased
        biased = r['best_biased']
        print(f"{'2️⃣  ' + biased['method']:<35} {biased['cagr']:>9.2f}% {biased['sharpe']:>10.2f} {biased['maxdd']:>9.2f}%")
        
        # Walk-forward
        wf = r['walk_forward']
        print(f"{'3️⃣  ' + wf['method']:<35} {wf['cagr']:>9.2f}% {wf['sharpe']:>10.2f} {wf['maxdd']:>9.2f}%")
        
        # ML
        ml = r['ml']
        print(f"{'4️⃣  ' + ml['method']:<35} {ml['cagr']:>9.2f}% {ml['sharpe']:>10.2f} {ml['maxdd']:>9.2f}%")
        
        # Performance gaps
        print("\n" + "="*100)
        print("📈 Écarts vs Buy & Hold:")
        diff_biased_vs_bh = biased['cagr'] - bh['cagr']
        diff_wf_vs_bh = wf['cagr'] - bh['cagr']
        diff_ml_vs_bh = ml['cagr'] - bh['cagr']
        
        print(f"   • Best Biased vs B&H:      {diff_biased_vs_bh:+.2f}% {'🚀' if diff_biased_vs_bh > 0 else '🔴'}")
        print(f"   • Walk-Forward vs B&H:     {diff_wf_vs_bh:+.2f}% {'�' if diff_wf_vs_bh > 0 else '🔴'}")
        print(f"   • ML vs B&H:               {diff_ml_vs_bh:+.2f}% {'🚀' if diff_ml_vs_bh > 0 else '🔴'}")
        
        print("\n📈 Écarts de performance:")
        diff_biased_vs_wf = biased['cagr'] - wf['cagr']
        diff_ml_vs_wf = ml['cagr'] - wf['cagr']
        
        print(f"   • Biais du look-ahead:     {abs(diff_biased_vs_wf):.2f}% {'�' if diff_biased_vs_wf < 0 else '✅'}")
        print(f"   • ML vs Walk-Forward:      {diff_ml_vs_wf:+.2f}% {'🚀' if diff_ml_vs_wf > 0 else '🔴'}")
        print("="*100)

def print_summary(results):
    """Print overall summary."""
    
    print("\n" + "="*100)
    print("💡 RÉSUMÉ & INTERPRÉTATION")
    print("="*100)
    
    print("\n� BUY & HOLD:")
    print("   Stratégie passive - acheter et garder. Benchmark de référence.")
    
    print("\n�🔴 BIAIS DU LOOK-AHEAD:")
    print("   La 'meilleure stratégie biaisée' utilise des infos du futur.")
    print("   C'est comme tricher en regardant les réponses!")
    
    print("\n✅ WALK-FORWARD (SANS BIAIS):")
    print("   Sélection basée sur le passé, testée sur le futur.")
    print("   C'est la performance RÉALISTE que vous auriez obtenue.")
    
    print("\n🤖 MACHINE LEARNING:")
    print("   Le ML sélectionne automatiquement les meilleures MA pairs")
    print("   en utilisant 21 features (prix, volume, momentum, SPY, etc.).")
    
    # Calculate averages
    avg_bh = sum(r['buy_hold']['cagr'] for r in results) / len(results)
    avg_biased = sum(r['best_biased']['cagr'] for r in results) / len(results)
    avg_wf = sum(r['walk_forward']['cagr'] for r in results) / len(results)
    avg_ml = sum(r['ml']['cagr'] for r in results) / len(results)
    
    print("\n📊 MOYENNES SUR TOUS LES TICKERS:")
    print(f"   1️⃣  Buy & Hold:           {avg_bh:>8.2f}% CAGR")
    print(f"   2️⃣  Biased (look-ahead):  {avg_biased:>8.2f}% CAGR")
    print(f"   3️⃣  Walk-Forward:         {avg_wf:>8.2f}% CAGR")
    print(f"   4️⃣  Machine Learning:     {avg_ml:>8.2f}% CAGR")
    
    improvement_vs_wf = avg_ml - avg_wf
    improvement_vs_bh = avg_ml - avg_bh
    print(f"\n   🚀 Amélioration ML vs Walk-Forward: {improvement_vs_wf:+.2f}% CAGR")
    print(f"   🚀 Amélioration ML vs Buy & Hold:   {improvement_vs_bh:+.2f}% CAGR")
    
    print("="*100)

def print_regularization_results(results):
    """Print regularization analysis results."""
    
    if not results:
        print("\n⚠️  Aucune analyse de régularisation trouvée")
        return
    
    print("\n" + "="*100)
    print("🔬 ANALYSE DE RÉGULARISATION (Lasso)")
    print("="*100)
    print(f"\n{'Ticker':<10} {'Alpha optimal':<15} {'Test R²':<12} {'Train R²':<12} {'Features':<12}")
    print("-"*100)
    
    for r in results:
        alpha_str = f"{r['alpha']:.2e}"
        test_r2_str = f"{r['test_r2']:.6f}"
        train_r2_str = f"{r['train_r2']:.6f}"
        features_str = f"{int(r['n_features'])}/21"
        
        print(f"{r['ticker']:<10} {alpha_str:<15} {test_r2_str:<12} {train_r2_str:<12} {features_str:<12}")

def print_files_location():
    """Print where to find detailed results."""
    
    print("\n" + "="*100)
    print("📁 FICHIERS DE RÉSULTATS DÉTAILLÉS")
    print("="*100)
    print("\n📊 Stratégie Traditionnelle:")
    print("   • Backtests:      data/SRC/results/backtest/")
    print("   • Walk-forward:   data/SRC/results/variations/")
    print("   • Graphiques:     data/SRC/results/variations/*_equity_curves.png")
    
    print("\n🤖 Machine Learning:")
    print("   • Datasets ML:    data/ML/TICKER_ml_data.csv")
    print("   • Modèles:        ML/models/TICKER_regression_*.pkl")
    print("   • Backtests ML:   data/ML/backtest_results/")
    print("   • Régularisation: data/ML/regularization_analysis/")
    print("   • Graphiques:     data/ML/regularization_analysis/*_analysis.png")

def main():
    """Main function."""
    
    print_header()
    
    # Load all results
    combined_results = []
    reg_results = []
    
    for ticker in TICKERS:
        # Traditional (both biased and walk-forward)
        trad = load_traditional_results(ticker)
        
        # ML
        ml = load_ml_results(ticker)
        
        # Regularization
        reg = load_regularization_results(ticker)
        if reg:
            reg_results.append(reg)
        
        # Combine if both exist
        if trad and ml:
            combined_results.append({
                'ticker': ticker,
                'buy_hold': trad['buy_hold'],
                'best_biased': trad['best_biased'],
                'walk_forward': trad['walk_forward'],
                'ml': ml
            })
    
    # Print main comparison
    print_main_comparison(combined_results)
    
    # Print regularization results
    print_regularization_results(reg_results)
    
    # Print summary
    print_summary(combined_results)
    
    # Print files location
    print_files_location()
    
    print("\n" + "="*100)
    print("✅ Résumé complet affiché!")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
