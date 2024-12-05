import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path, column_name):
    """Load portfolio data from a CSV file."""
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    return df[column_name]

def arima_synthetic_sequence(model, steps=30):
    """Generate synthetic data using an ARIMA model."""
    forecast = model.forecast(steps=steps)
    return forecast

def regression_synthetic_sequence(model, initial_data, lagged_features, lag=5, steps=30):
    """Generate synthetic data using a trained regression model."""
    synthetic_data = list(initial_data[-lag:])
    for _ in range(steps):
        # Prepare lagged features from the synthetic data
        lagged_input = pd.DataFrame([synthetic_data[-lag:]], columns=lagged_features)
        # Predict the next value
        next_value = model.predict(lagged_input)[0]
        synthetic_data.append(next_value)
    return synthetic_data[-steps:]

def combine_predictions(arima_pred, reg_pred, arima_weight=0.5, reg_weight=0.5):
    """Combine ARIMA and regression predictions."""
    combined = arima_weight * np.array(arima_pred) + reg_weight * np.array(reg_pred)
    return combined

def plot_synthetic_data(actual_series, arima_pred, reg_pred, combined_pred, future_dates):
    """Plot actual data and synthetic predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual_series.index, actual_series, label="Actual Data", color="blue")
    plt.plot(future_dates, arima_pred, label="ARIMA Prediction", linestyle="--", color="red")
    plt.plot(future_dates, reg_pred, label="Regression Prediction", linestyle="--", color="green")
    plt.plot(future_dates, combined_pred, label="Combined Prediction", linestyle="--", color="purple")
    plt.legend()
    plt.title("Synthetic Stock Price Sequences")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

if __name__ == "__main__":
    # File and column configuration
    file_path = "data/raw/portfolio.csv"
    column_name = "0"  # Replace with your actual column name

    # Load and prepare data
    portfolio_data = load_data(file_path, column_name)
    train_series = portfolio_data[:-30]  # Use all but the last 30 days for training
    future_dates = pd.date_range(start=train_series.index[-1], periods=31, freq="B")[1:]

    # Ensure proper index frequency for ARIMA
    train_series.index = pd.date_range(start=train_series.index[0], periods=len(train_series), freq='B')

    # Step 4.1: ARIMA-based Synthetic Data
    print("Training ARIMA model...")
    arima_model = ARIMA(train_series, order=(2, 1, 2))
    arima_model_fit = arima_model.fit()
    arima_predictions = arima_synthetic_sequence(arima_model_fit, steps=30)

    # Step 4.2: Regression-based Synthetic Data
    print("Loading Linear Regression model...")
    regression_model = joblib.load("linear_regression_model.pkl")
    lagged_features = joblib.load("lagged_feature_columns.pkl")
    lag = len(lagged_features)

    regression_predictions = regression_synthetic_sequence(
        regression_model, train_series.values, lagged_features, lag=lag, steps=30
    )

    # Step 4.3: Combine ARIMA and Regression Predictions
    combined_predictions = combine_predictions(arima_predictions, regression_predictions)

    # Plot results
    print("Plotting synthetic stock price sequences...")
    plot_synthetic_data(
        train_series,
        arima_predictions,
        regression_predictions,
        combined_predictions,
        future_dates,
    )
