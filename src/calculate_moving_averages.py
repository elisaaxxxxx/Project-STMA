"""
Calculate Moving Averages for Stock Data
==========================================

Ce script utilise la configuration centralisée pour :
1. Lire les fichiers CSV configurés
2. Calculer les moyennes mobiles selon les périodes configurées
3. Sauvegarder dans le dossier configuré

Les paramètres sont définis dans project_config.py
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Importer la configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import MA_PERIODS, DATA_PROCESSED_DIR, get_ma_file_path


def calculate_moving_averages(input_file, output_folder=None):
    """
    Calcule les moyennes mobiles selon la configuration.
    
    Args:
        input_file: Chemin vers le fichier CSV avec les données
        output_folder: Dossier de sortie (utilise STRATEGY_DIR si None)
    """
    if output_folder is None:
        output_folder = DATA_PROCESSED_DIR
    
    # Read the CSV file
    print(f"Reading {input_file.name}...")
    df = pd.read_csv(input_file)
    
    # Utilise les périodes de MA configurées
    windows = MA_PERIODS
    
    # Calculate moving average for each window
    for window in windows:
        column_name = f"MA_{window}"
        # Calculate: for each day, take that day's Close + previous (window-1) days
        # min_periods=window means we only show the average when we have enough data
        df[column_name] = df['Close'].rolling(window=window, min_periods=window).mean()
        print(f"  ✓ Calculated {column_name}")
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the result
    output_file = output_path / f"{input_file.stem}_with_MAs.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}\n")
    
    return output_file


def main():
    """Traite tous les fichiers CSV selon la configuration."""
    from project_config import DATA_RAW_DIR, print_config, validate_config
    
    # Validation de la configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Erreurs de configuration :")
        for error in errors:
            print(f"  - {error}")
        return
    
    print_config()
    
    # Input folder containing the stock data
    input_folder = Path(DATA_RAW_DIR)
    
    # Find all CSV files
    csv_files = list(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    print("=" * 60)
    
    # Process each file
    for csv_file in csv_files:
        calculate_moving_averages(csv_file)
    
    print("=" * 60)
    print("✅ All done! Check the MA_strategy/ folder for results.")


if __name__ == "__main__":
    main()
