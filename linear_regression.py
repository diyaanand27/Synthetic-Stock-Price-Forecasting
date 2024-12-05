import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(file_path):
    """Load stock data from a CSV file."""
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df = df.dropna(subset=["Adj Close"])  # Drop rows where 'Adj Close' is NaN
    return df["Adj Close"]


def create_lagged_features(series, lag=5):
    """Create lagged features for stock price prediction."""
    df = pd.DataFrame(series)
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df['Adj Close'].shift(i)
    df = df.dropna()
    return df


def linear_regression_model(df):
    """Train a Linear Regression model."""
    X = df.drop(columns='Adj Close')  # Features (lagged values)
    y = df['Adj Close']  # Target (stock price)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression MAE: {mae}")
    print(f"Linear Regression R²: {r2}")

    return model, X_test, y_test, y_pred


def plot_results(y_test, y_pred, stock_name):
    """Plot actual vs predicted stock prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual Prices", color="blue")
    plt.plot(y_test.index, y_pred, label="Predicted Prices", color="red", linestyle="--")
    plt.legend()
    plt.title(f"Linear Regression - Actual vs Predicted Stock Prices ({stock_name})")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()


def process_all_stocks(data_dir="data/raw"):
    """Process all CSV files in the given directory."""
    # List all CSV files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            print(f"\nProcessing stock data for: {filename}")

            # Load data for the stock
            series = load_data(file_path)

            # Create lagged features (using the last 5 days)
            df = create_lagged_features(series, lag=5)

            # Train the Linear Regression model and evaluate it
            model, X_test, y_test, y_pred = linear_regression_model(df)

            # Plot the results
            plot_results(y_test, y_pred, stock_name=filename)


if __name__ == "__main__":
    # Process all stocks in the data directory
    process_all_stocks(data_dir="data/raw")
