"""
ML Feature Engineering
======================

Creates clean ML-ready feature datasets from raw OHLCV data.
Calculates momentum, trend, volatility, volume, and market indicators.

Output: Clean CSV files with only calculated features (no raw OHLCV data)
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent pour importer la config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import (
    TICKERS, START_DATE, END_DATE, 
    DATA_RAW_DIR, DATA_FEATURES_DIR,
    get_data_file_path
)

class MLFeatureEngineer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.spy_data = None
        self.features = pd.DataFrame()
    
    def load_stock_data(self):
        """Charge les données brutes du stock."""
        file_path = get_data_file_path(self.ticker)
        print(f"📊 Chargement {self.ticker}: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        self.data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        print(f"   ✓ {len(self.data)} lignes chargées")
        return self.data
    
    def load_spy_data(self):
        """Charge les données SPY pour les features de marché."""
        spy_file = get_data_file_path('SPY')
        print(f"📈 Chargement SPY: {spy_file}")
        
        if not os.path.exists(spy_file):
            raise FileNotFoundError(f"Fichier SPY non trouvé: {spy_file}")
        
        self.spy_data = pd.read_csv(spy_file, index_col='Date', parse_dates=True)
        print(f"   ✓ {len(self.spy_data)} lignes SPY chargées")
        return self.spy_data
    
    def calculate_momentum_features(self):
        """🚀 Calcule les features de momentum."""
        print("   🚀 Momentum features...")
        
        close = self.data['Close']
        
        # Rendements sur différentes périodes
        self.features['ret_1d'] = close.pct_change()
        self.features['ret_5d'] = close.pct_change(5)
        self.features['ret_20d'] = close.pct_change(20)
        self.features['ret_60d'] = close.pct_change(60)
        
        # Momentum à plus long terme
        self.features['momentum_1m'] = close / close.shift(21) - 1  # 1 mois
        self.features['momentum_3m'] = close / close.shift(63) - 1  # 3 mois
    
    def calculate_trend_features(self):
        """📈 Calcule les features de trend et moyennes mobiles."""
        print("   📈 Trend features...")
        
        close = self.data['Close']
        
        # Calcul des moyennes mobiles
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        
        # Prix par rapport à MA200
        self.features['price_over_ma200'] = close / ma_200
        
        # Ratios de moyennes mobiles
        self.features['ma_ratio_5_20'] = ma_5 / ma_20
        self.features['ma_ratio_10_50'] = ma_10 / ma_50
        self.features['ma_ratio_20_200'] = ma_20 / ma_200
    
    def calculate_volatility_features(self):
        """📉 Calcule les features de volatilité."""
        print("   📉 Volatility features...")
        
        # Volatilité sur 20 jours (écart-type des rendements)
        returns_1d = self.data['Close'].pct_change()
        self.features['vol_20d'] = returns_1d.rolling(20).std()
    
    def calculate_volume_features(self):
        """📊 Calcule les features de volume."""
        print("   📊 Volume features...")
        
        if 'Volume' in self.data.columns:
            volume = self.data['Volume']
            
            # Volume moyen sur 20 jours
            self.features['volume_20d_avg'] = volume.rolling(20).mean()
            
            # Ratio volume actuel / moyenne
            self.features['volume_ratio'] = volume / self.features['volume_20d_avg']
        else:
            print("      ⚠️ Pas de données de volume disponibles")
            self.features['volume_20d_avg'] = np.nan
            self.features['volume_ratio'] = np.nan
    
    def calculate_market_features(self):
        """🌍 Calcule les features de marché basées sur SPY."""
        print("   🌍 Market features (SPY)...")
        
        spy_close = self.spy_data['Close']
        
        # Aligner les dates avec le stock principal
        spy_aligned = spy_close.reindex(self.data.index, method='ffill')
        
        # Market momentum
        self.features['spy_ret_5d'] = spy_aligned.pct_change(5)
        self.features['spy_ret_20d'] = spy_aligned.pct_change(20)
        
        # Market volatility
        spy_returns = spy_aligned.pct_change()
        self.features['spy_vol_20d'] = spy_returns.rolling(20).std()
        
        # Market trend (MA ratio)
        spy_ma_20 = spy_aligned.rolling(20).mean()
        spy_ma_50 = spy_aligned.rolling(50).mean()
        self.features['spy_ma_ratio_20_50'] = spy_ma_20 / spy_ma_50
        
        # Market autocorrelation (lag-1)
        self.features['spy_autocorr_1d'] = spy_returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan
        )
    
    def engineer_all_features(self):
        """Lance le calcul de toutes les features ML."""
        print(f"\n=== 🤖 ML Feature Engineering: {self.ticker} ===")
        
        # Initialiser avec l'index des dates
        self.features = pd.DataFrame(index=self.data.index)
        
        # Calculer toutes les features
        self.calculate_momentum_features()
        self.calculate_trend_features() 
        self.calculate_volatility_features()
        self.calculate_volume_features()
        self.calculate_market_features()
        
        # Supprimer les lignes avec trop de NaN (début de série)
        # Garder seulement les lignes avec au moins 80% de données
        threshold = len(self.features.columns) * 0.8
        self.features = self.features.dropna(thresh=threshold)
        
        print(f"   ✅ Features calculées: {self.features.shape}")
        return self.features
    
    def save_features(self):
        """Sauvegarde les features ML."""
        # Créer le répertoire s'il n'existe pas
        os.makedirs(DATA_FEATURES_DIR, exist_ok=True)
        
        # Nom du fichier avec ML pour identifier
        filename = f"{self.ticker}_ML_features.csv"
        file_path = os.path.join(DATA_FEATURES_DIR, filename)
        
        # Sauvegarder
        self.features.to_csv(file_path)
        
        print(f"   💾 Features sauvegardées: {filename}")
        print(f"      📊 Shape: {self.features.shape}")
        print(f"      📅 Période: {self.features.index[0].date()} → {self.features.index[-1].date()}")
        
        return file_path


def process_ticker(ticker):
    """Traite un ticker individuel."""
    try:
        engineer = MLFeatureEngineer(ticker)
        engineer.load_stock_data()
        engineer.load_spy_data()
        engineer.engineer_all_features()
        engineer.save_features()
        return True
    
    except Exception as e:
        print(f"   ❌ Erreur pour {ticker}: {e}")
        return False


def main():
    """Fonction principale - traite tous les tickers."""
    print("=" * 80)
    print("🤖 ML FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Tickers: {TICKERS}")
    print(f"Période: {START_DATE} → {END_DATE}")
    print(f"Output: {DATA_FEATURES_DIR}")
    print()
    
    success_count = 0
    failed_tickers = []
    
    for ticker in TICKERS:
        if process_ticker(ticker):
            success_count += 1
        else:
            failed_tickers.append(ticker)
        print()
    
    # Résumé final
    print("=" * 80)
    print(f"✅ TERMINÉ: {success_count}/{len(TICKERS)} tickers traités avec succès")
    
    if failed_tickers:
        print(f"❌ Échecs: {failed_tickers}")
    
    print(f"📁 Features sauvegardées dans: {DATA_FEATURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()