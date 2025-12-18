import yfinance as yf
import pandas as pd
from pathlib import Path
import sys
import os

# Import configuration (now in parent folder)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import ALL_TICKERS, START_DATE, END_DATE, get_data_file_path, print_config, validate_config

class DataLoader:
    def __init__(self, ticker='SPY', start=None, end=None):
        self.ticker = ticker
        self.start = start if start else START_DATE
        self.end = end if end else END_DATE
        self.data = None
    
    def download(self):
        """Downloads data from Yahoo Finance"""
        print(f"Downloading {self.ticker} from {self.start} to {self.end}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end)
        return self.data
    
    def validate(self):
        """Validates data integrity"""
        assert self.data is not None, "No data loaded"
        assert not self.data.empty, "Empty dataframe"
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            print(f"Warning: Missing values detected:\n{missing[missing > 0]}")
        
        # Check for negative prices
        assert (self.data[['Open', 'High', 'Low', 'Close']] > 0).all().all(), \
               "Negative prices detected"
        
        print(f"✓ Data validated: {len(self.data)} rows from {self.data.index[0]} to {self.data.index[-1]}")
        return True
    
    def save(self, path=None):
        """Saves data to the configured directory.

        Cleans multi-index headers produced by yfinance and writes a
        simple CSV with a single header line: Date,Open,High,Low,Close,Volume.
        """
        if path is None:
            from project_config import DATA_RAW_DIR
            path = DATA_RAW_DIR
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        filename = f"{self.ticker}_{self.start}_{self.end}.csv"
        filepath = target / filename

        # Prepare a cleaned copy of the dataframe:
        df_clean = self.data.copy()
        # Reset index to 'Date' column
        try:
            df_clean = df_clean.reset_index()
            # Format Date column to have only the date (without time)
            if 'Date' in df_clean.columns:
                df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.strftime('%Y-%m-%d')
        except Exception:
            # if reset_index fails, continue with the original
            pass

        # Flatten MultiIndex columns if present
        if hasattr(df_clean.columns, 'levels') and getattr(df_clean.columns, 'nlevels', 1) > 1:
            new_cols = {}
            for col in df_clean.columns:
                if isinstance(col, tuple):
                    # Look for a useful label (Open/High/Low/Close/Adj Close/Volume)
                    chosen = None
                    for part in col:
                        if isinstance(part, str) and part.strip() in ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'):
                            chosen = part.strip()
                            break
                    if chosen is None:
                        # fallback: join non-empty parts
                        chosen = '_'.join([str(p) for p in col if p])
                    new_cols[col] = chosen
                else:
                    new_cols[col] = col
            df_clean = df_clean.rename(columns=new_cols)

        # If 'Adj Close' exists but not 'Close', rename it
        if 'Adj Close' in df_clean.columns and 'Close' not in df_clean.columns:
            df_clean = df_clean.rename(columns={'Adj Close': 'Close'})

        # Select columns in desired order if they exist
        desired = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        existing = [c for c in desired if c in df_clean.columns]
        # Ensure 'Date' is the first column
        if 'Date' in existing:
            df_to_save = df_clean[['Date'] + [c for c in existing if c != 'Date']]
        else:
            # fallback: save everything with index False
            df_to_save = df_clean

        # Save without index or multi-headers
        # Write a simple CSV with a single header: Date,Open,High,Low,Close,Volume
        # Ensure columns exist and are in the correct order
        desired = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        existing = [c for c in desired if c in df_clean.columns]
        if 'Date' in df_clean.columns and 'Date' not in existing:
            existing.insert(0, 'Date')

        if existing:
            cols_to_write = existing
        else:
            cols_to_write = list(df_clean.columns)

        # Build a clean DataFrame with exactly the desired columns
        # by finding the best matches in df_clean
        out_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        rows = []
        for col in out_cols:
            # find a candidate column in df_clean
            candidates = [c for c in df_clean.columns if str(c).lower() == col.lower()]
            if not candidates:
                # heuristic: partial match
                candidates = [c for c in df_clean.columns if col.lower() in str(c).lower()]
            if candidates:
                # take the first match
                rows.append(candidates[0])
            else:
                rows.append(None)

        # Build final DataFrame column by column
        import csv
        with open(filepath, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            # write fixed header
            writer.writerow(out_cols)
            # write data rows
            # determine number of rows
            if 'Date' in df_clean.columns:
                nrows = len(df_clean)
            else:
                # if Date was the initial index
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
    
    # Get current configuration values (after potential reload)
    current_tickers = ','.join(ALL_TICKERS)  # Includes TICKERS + BENCHMARK
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
    # Force reload configuration to avoid cache issues
    import importlib
    if 'project_config' in sys.modules:
        importlib.reload(sys.modules['project_config'])
        # Re-import variables after reload
        from project_config import ALL_TICKERS, START_DATE, END_DATE, get_data_file_path, print_config, validate_config
    
    # Validate configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("❌ Configuration errors:")
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