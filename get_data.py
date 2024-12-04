import pandas as pd
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Helper Functions
def fetch_raw_data(symbols, start_date, end_date):
    """
    Fetch raw stock price data for a list of symbols from Yahoo Finance.
    Parameters:
        symbols (list): List of stock symbols to fetch.
        start_date (str): Start date for historical data (format: 'YYYY-MM-DD').
        end_date (str): End date for historical data (format: 'YYYY-MM-DD').
    Returns:
        pd.DataFrame: A combined DataFrame containing data for all symbols.
    """
    data_frames = []
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
        df['Symbol'] = symbol
        data_frames.append(df)
    return pd.concat(data_frames)

def calculate_portfolio_value(raw_data, weights):
    """
    Calculate the portfolio value based on stock adjusted close prices and weights.
    Parameters:
        raw_data (pd.DataFrame): Raw data containing adjusted close prices for all stocks.
        weights (dict): A dictionary of stock symbols and their weights in the portfolio.
    Returns:
        pd.Series: Portfolio value as a weighted sum of adjusted close prices.
    """
    portfolio = pd.DataFrame()
    for symbol, weight in weights.items():
        portfolio[symbol] = raw_data[raw_data['Symbol'] == symbol]['Adj Close'] * weight
    return portfolio.sum(axis=1)

def clean_portfolio_data(portfolio):
    """
    Clean and preprocess the portfolio data.
    Parameters:
        portfolio (pd.Series): Portfolio value series.
    Returns:
        pd.Series: Cleaned portfolio value series.
    """
    # Handle missing values
    print("Cleaning portfolio data...")
    portfolio = portfolio.fillna(method='ffill').fillna(method='bfill')

    # Normalize the portfolio value
    print("Normalizing portfolio data...")
    scaler = MinMaxScaler()
    portfolio_normalized = pd.Series(
        scaler.fit_transform(portfolio.values.reshape(-1, 1)).flatten(),
        index=portfolio.index
    )
    return portfolio_normalized

# Main Execution
if __name__ == "__main__":
    # Configuration
    symbols = ['JANX', 'MSFT', 'GOOG', 'PLTR', 'NVDA', 'INTC', 'PCG', 'SMCI', 'CRM', 'IBM']  # List of stock symbols in the portfolio
    weights = {'JANX': 0.1, 'MSFT': 0.1, 'GOOG': 0.1, 'PLTR': 0.1, 'NVDA': 0.1, 'INTC': 0.1, 'PCG': 0.1, 'SMCI': 0.1, 'CRM': 0.1, 'IBM': 0.1}  # Portfolio weights
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    output_dir = "data/raw"

    # Fetch raw data
    print("Fetching raw data...")
    raw_data = fetch_raw_data(symbols, start_date, end_date)

    # Save raw data for each symbol
    print(f"Saving raw data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    for symbol in symbols:
        symbol_data = raw_data[raw_data['Symbol'] == symbol]
        symbol_data.to_csv(os.path.join(output_dir, f"{symbol}.csv"), index=True)

    # Calculate portfolio value
    print("Calculating portfolio value...")
    portfolio_value = calculate_portfolio_value(raw_data, weights)

    # Clean and preprocess portfolio value
    print("Cleaning and preprocessing portfolio value...")
    portfolio_cleaned = clean_portfolio_data(portfolio_value)

    # Save cleaned portfolio data
    portfolio_cleaned.to_csv(os.path.join(output_dir, "portfolio.csv"), index=True)
    print("Portfolio data saved successfully.")
