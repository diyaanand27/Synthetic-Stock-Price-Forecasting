import os
import yfinance as yf
import argparse
from sklearn.preprocessing import MinMaxScaler


# Define helper functions
def symbol_to_path(symbol, base_dir=os.path.join(".", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, f"{symbol}.csv")


def get_data_yf_symbol(symbol, start_date, end_date):
    """Fetch stock data (adjusted close) for the given symbol from Yahoo Finance."""
    return yf.download(
        symbol,
        interval="1d",
        start=start_date,
        end=end_date
    )


def add_technical_indicators(df):
    """Add technical indicators to the data."""
    df['50_MA'] = df['Adj Close'].rolling(window=50).mean()  # 50-day Moving Average
    df['200_MA'] = df['Adj Close'].rolling(window=200).mean()  # 200-day Moving Average
    df['Daily Return'] = df['Adj Close'].pct_change()  # Daily Returns
    df['Volatility'] = df['Daily Return'].rolling(window=20).std()  # Volatility (20-day rolling)
    return df


def train_test_split_time_series(df, test_ratio=0.2):
    """Split time-series data into training and testing sets."""
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    return train_df, test_df


def normalize_data(df):
    """Normalize stock data using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    )
    return df


if __name__ == '__main__':
    # Default settings
    default_symbols = [
        'SMCI', #Super Micro Company
        'PFE',  # Pfizer
        'INTC',  # Intel Corporation
        'AAPL',  # Apple
        'IBM',  # IBM
        'MSFT',  # Microsoft
        'PLTR',  # Palantir
        'JANX',  # Janux
        'AMZN',  # Amazon
        'CRDO',  # Credo
        'SPY',  # S&P 500 ETF
        'PCG'  #PG&E
    ]
    date_start = '2020-01-01'
    date_end = '2023-11-30'

    # Command-line argument parser
    parser = argparse.ArgumentParser(
        prog='get_yf_data.py',
        description='Download and preprocess historical stock data from Yahoo Finance'
    )
    parser.add_argument('-x', nargs='+', action="store", dest="symbols",
                        help="List of stock symbols separated by space")
    parser.add_argument('-s', action="store", dest="date_start", default=date_start,
                        help="Start date for data download (format: YYYY-MM-DD)")
    parser.add_argument('-e', action="store", dest="date_end", default=date_end,
                        help="End date for data download (format: YYYY-MM-DD)")
    args = parser.parse_args()

    # Use provided symbols if available, otherwise default to predefined list
    symbols = args.symbols if args.symbols else default_symbols

    for symbol in symbols:
        try:
            print(f"Fetching data for {symbol} from {args.date_start} to {args.date_end}")
            df = get_data_yf_symbol(symbol, args.date_start, args.date_end)

            # Handle missing data
            df = df.dropna()

            # Add technical indicators
            df = add_technical_indicators(df)

            # Normalize data
            df = normalize_data(df)

            # Split into training and testing sets
            train_df, test_df = train_test_split_time_series(df)

            # Save raw data
            raw_datafile = os.path.join("data/raw", f"{symbol}.csv")
            os.makedirs(os.path.dirname(raw_datafile), exist_ok=True)
            df.to_csv(raw_datafile, index=True)

            # Save processed data
            processed_datafile = os.path.join("data/processed", f"{symbol}.csv")
            os.makedirs(os.path.dirname(processed_datafile), exist_ok=True)
            train_df.to_csv(processed_datafile.replace(".csv", "_train.csv"), index=True)
            test_df.to_csv(processed_datafile.replace(".csv", "_test.csv"), index=True)

            print(f"Data for {symbol} saved successfully.")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

