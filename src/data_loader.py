import yfinance as yf
import pandas as pd
from pathlib import Path
import sys
import os

# Importer la configuration (maintenant dans le dossier parent)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import ALL_TICKERS, START_DATE, END_DATE, get_data_file_path, print_config, validate_config

class DataLoader:
    def __init__(self, ticker='SPY', start=None, end=None):
        self.ticker = ticker
        self.start = start if start else START_DATE
        self.end = end if end else END_DATE
        self.data = None
    
    def download(self):
        """Télécharge les données depuis Yahoo Finance"""
        print(f"Downloading {self.ticker} from {self.start} to {self.end}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end)
        return self.data
    
    def validate(self):
        """Vérifie l'intégrité des données"""
        assert self.data is not None, "No data loaded"
        assert not self.data.empty, "Empty dataframe"
        
        # Vérifier les valeurs manquantes
        missing = self.data.isnull().sum()
        if missing.any():
            print(f"Warning: Missing values detected:\n{missing[missing > 0]}")
        
        # Vérifier les prix négatifs
        assert (self.data[['Open', 'High', 'Low', 'Close']] > 0).all().all(), \
               "Negative prices detected"
        
        print(f"✓ Data validated: {len(self.data)} rows from {self.data.index[0]} to {self.data.index[-1]}")
        return True
    
    def save(self, path=None):
        """Sauvegarde les données dans le répertoire configuré.

        Nettoie les en-têtes multi-index produits par yfinance et écrit un
        CSV simple avec une seule ligne d'en-tête : Date,Open,High,Low,Close,Volume.
        """
        if path is None:
            from project_config import DATA_RAW_DIR
            path = DATA_RAW_DIR
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        filename = f"{self.ticker}_{self.start}_{self.end}.csv"
        filepath = target / filename

        # Préparer une copie nettoyée du dataframe :
        df_clean = self.data.copy()
        # Remettre l'index en colonne 'Date'
        try:
            df_clean = df_clean.reset_index()
            # Formater la colonne Date pour n'avoir que la date (sans l'heure)
            if 'Date' in df_clean.columns:
                df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.strftime('%Y-%m-%d')
        except Exception:
            # si reset_index échoue, on continue avec l'original
            pass

        # Aplatir les colonnes MultiIndex si présent
        if hasattr(df_clean.columns, 'levels') and getattr(df_clean.columns, 'nlevels', 1) > 1:
            new_cols = {}
            for col in df_clean.columns:
                if isinstance(col, tuple):
                    # Cherche un label utile (Open/High/Low/Close/Adj Close/Volume)
                    chosen = None
                    for part in col:
                        if isinstance(part, str) and part.strip() in ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'):
                            chosen = part.strip()
                            break
                    if chosen is None:
                        # fallback : joindre les parties non vides
                        chosen = '_'.join([str(p) for p in col if p])
                    new_cols[col] = chosen
                else:
                    new_cols[col] = col
            df_clean = df_clean.rename(columns=new_cols)

        # Si 'Adj Close' existe mais pas 'Close', on renomme
        if 'Adj Close' in df_clean.columns and 'Close' not in df_clean.columns:
            df_clean = df_clean.rename(columns={'Adj Close': 'Close'})

        # Sélectionner colonnes dans l'ordre souhaité si elles existent
        desired = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        existing = [c for c in desired if c in df_clean.columns]
        # S'assurer que 'Date' est la première colonne
        if 'Date' in existing:
            df_to_save = df_clean[['Date'] + [c for c in existing if c != 'Date']]
        else:
            # fallback : sauvegarder tout avec index False
            df_to_save = df_clean

        # Sauvegarder sans index ni multi-en-têtes
        # Écrire un CSV simple avec un seul en-tête : Date,Open,High,Low,Close,Volume
        # S'assurer que les colonnes existent et dans le bon ordre
        desired = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        existing = [c for c in desired if c in df_clean.columns]
        if 'Date' in df_clean.columns and 'Date' not in existing:
            existing.insert(0, 'Date')

        if existing:
            cols_to_write = existing
        else:
            cols_to_write = list(df_clean.columns)

        # Construire un DataFrame propre avec exactement les colonnes désirées
        # en cherchant les meilleures correspondances dans df_clean
        out_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        rows = []
        for col in out_cols:
            # trouver une colonne candidate dans df_clean
            candidates = [c for c in df_clean.columns if str(c).lower() == col.lower()]
            if not candidates:
                # heuristique : match partiel
                candidates = [c for c in df_clean.columns if col.lower() in str(c).lower()]
            if candidates:
                # prendre la première correspondance
                rows.append(candidates[0])
            else:
                rows.append(None)

        # Construire DataFrame final colonne par colonne
        import csv
        with open(filepath, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            # écrire l'en-tête fixe
            writer.writerow(out_cols)
            # écrire les lignes de données
            # déterminer nombre de lignes
            if 'Date' in df_clean.columns:
                nrows = len(df_clean)
            else:
                # si Date était l'index initial
                nrows = len(df_clean)

            for i in range(nrows):
                row = []
                for src in rows:
                    if src is None:
                        row.append('')
                    else:
                        try:
                            row.append(df_clean.iloc[i][src])
                        except Exception:
                            row.append('')
                writer.writerow(row)

        print(f"✓ Data saved to {filepath}")


def _parse_args():
    """Parse CLI arguments for quick use from the command line."""
    import argparse
    
    # Récupérer les valeurs actuelles de la configuration (après rechargement éventuel)
    current_tickers = ','.join(ALL_TICKERS)  # Inclut TICKERS + BENCHMARK
    current_start = START_DATE
    current_end = END_DATE

    parser = argparse.ArgumentParser(description='Download and save OHLCV data using yfinance')
    # Accept either a single ticker or a comma-separated list.
    # For backward compatibility we accept --ticker and --tickers (same dest).
    parser.add_argument('--ticker', '--tickers', '-t', dest='tickers',
                        default=current_tickers,
                        help=f'Ticker or comma-separated tickers to download (config: {current_tickers})')
    parser.add_argument('--start', '-s', default=current_start, help=f'Start date YYYY-MM-DD (config: {current_start})')
    parser.add_argument('--end', '-e', default=current_end, help=f'End date YYYY-MM-DD (config: {current_end})')
    parser.add_argument('--path', '-p', default=None, help='Directory to save CSV (default: configured data/raw/)')
    return parser.parse_args()


if __name__ == '__main__':
    # Force reload de la configuration pour éviter les problèmes de cache
    import importlib
    if 'project_config' in sys.modules:
        importlib.reload(sys.modules['project_config'])
        # Re-importer les variables après rechargement
        from project_config import ALL_TICKERS, START_DATE, END_DATE, get_data_file_path, print_config, validate_config
    
    # Valider la configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Erreurs de configuration :")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    
    print_config()
    
    # When executed as a script, download the requested data and save it.
    args = _parse_args()
    # Support comma-separated tickers (e.g. "SPY,AAPL")
    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    for tk in tickers:
        print(f"\n=== Processing {tk} ===")
        loader = DataLoader(ticker=tk, start=args.start, end=args.end)
        try:
            loader.download()
        except Exception as exc:
            print(f"Error while downloading {tk}: {exc}")
            # continue with next ticker
            continue

        try:
            loader.validate()
        except AssertionError as ae:
            print(f"Validation failed for {tk}: {ae}")
            # still attempt to save whatever we have for inspection

        loader.save(path=args.path)
        print(f"Saved {tk} to {os.path.dirname(get_data_file_path(tk))}")