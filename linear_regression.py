import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(file_path):
    """Load stock data from a CSV file."""
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df = df.dropna(subset=["Adj Close"])  # Drop rows where 'Adj Close' is NaN
    return df["Adj Close"]


def create_lagged_features(series, lag=45):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

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


def generate_synthetic_data(model, last_known_data, num_days=252, lag=45):
    """Generate synthetic stock prices for 2023 based on the trained model."""
    synthetic_prices = []
    current_data = last_known_data.copy()

    # Get the last date of the known data (for example, 2022-12-31)
    last_date = current_data.index[-1]

    for i in range(num_days):  # Generate data for 30 trading days in 2023
        # Convert to NumPy array and reshape the last `lag` days
        lagged_features = current_data[-lag:].to_numpy().reshape(1, -1)  # The last `lag` days' data

        # Predict the next stock price
        predicted_price = model.predict(lagged_features)[0]
        synthetic_prices.append(predicted_price)

        # Update the current data with the predicted price for the next prediction
        current_data = pd.concat(
            [current_data, pd.Series([predicted_price], index=[last_date + pd.Timedelta(days=i + 1)])],
            ignore_index=False)[-lag:]

    return synthetic_prices, pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days, freq='B')


def plot_results(y_test, y_pred, actual_data, synthetic_data, synthetic_dates, stock_name):
    """Plot actual vs predicted stock prices."""
    plt.figure(figsize=(10, 6))

    # Plot actual data
    plt.plot(actual_data.index, actual_data, label="Actual Prices", color="blue")

    # Plot predicted data
    plt.plot(y_test.index, y_pred, label="Predicted Prices", color="red", linestyle="--")

    # Plot synthetic data
    plt.plot(synthetic_dates, synthetic_data, label="Synthetic Prices", color="green", linestyle="--")

    plt.legend()
    plt.title(f"Linear Regression - Actual vs Predicted vs Synthetic Stock Prices ({stock_name})")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()

def process_portfolio_file(file_path="data/raw-2022/portfolio.csv"):
    """Process a single portfolio CSV file."""
    print(f"\nProcessing stock data for: {file_path}")

    # Load data for the portfolio
    series = load_data(file_path)

    # Create lagged features (using the last 45 days)
    df = create_lagged_features(series, lag=45)

    # Train the Linear Regression model and evaluate it
    model, X_test, y_test, y_pred = linear_regression_model(df)

    # Generate synthetic data for 1 month (about 20 trading days)
    synthetic_data, synthetic_dates = generate_synthetic_data(model, series, num_days=30, lag=45)

    # Plot the results
    plot_results(y_test, y_pred, series, synthetic_data, synthetic_dates, stock_name="Portfolio")

if __name__ == "__main__":
    # Process the specific portfolio file
    process_portfolio_file(file_path="data/raw-2022/portfolio.csv")
