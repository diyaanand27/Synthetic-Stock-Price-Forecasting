import pandas as pd
import matplotlib.pyplot as plt


def load_and_plot_portfolio(file_path):
    """Load stock data from market.csv and plot the stock prices."""
    # Load the portfolio CSV data
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Drop rows with missing 'Adj Close' values (if any)
    df = df.dropna(subset=["Adj Close"])

    # Plot the stock prices ('Adj Close')
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Adj Close"], label="Portfolio Stock Price", color="blue")
    plt.title("Portfolio Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Specify the file path to the market.csv file
    file_path = "data/raw-2023/market.csv"  # Adjust the path as necessary

    # Call the function to load and plot the data
    load_and_plot_portfolio(file_path)
