"""
    CSCI 4170 Fall 2024
    Final Project: Synthetic stock price using Scikit-learn
    @Author: Diya Anand
    @Author: Facus Dokubo-Wizzdom
    Notes: Dat ahas been divided into raw and processed. Processed data has been minimized using MinMaxScaler.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

def load_data(symbol, data_dir= "data/raw"):
    #Load historical data for symbols
    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {symbol} not found in {data_dir}. Please try again")
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    return df['Adj Close'].dropna()

#Function to implement a time series model to predict stock prices
def test_stationary(series):
    """Using the ADF test to see if a stock is stationary"""
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] > 0.05:
        print("The series is not stationary. Differencing is required.")
        return False
    print("The series is stationary.")
    return True


def difference_series(series, d=1):
    """Difference the series to make it stationary."""
    return series.diff(periods=d).dropna()

def train_model(series, order):
    """Train the ARIMA model."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def evaluate_model(model_fit, test_series):
    """Evaluate the ARIMA model on test data."""
    forecast = model_fit.forecast(steps=len(test_series))
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    print(f"RMSE: {rmse}")
    return forecast, rmse

def plot_forecast(train_series, test_series, forecast):
    """Plot the train, test, and forecast data."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_series, label="Training Data")
    plt.plot(test_series, label="Testing Data")
    plt.plot(test_series.index, forecast, label="Forecast", linestyle="--")
    plt.legend()
    plt.title("ARIMA Model Forecast")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()

if __name__ == "__main__":
    # Default configuration
    default_symbols = ['AAPL', 'MSFT', 'JANX','PCG', 'PLTR']  # List of stock symbols
    default_order = (1, 1, 1)  # ARIMA order
    test_ratio = 0.2

    # Command-line arguments
    parser = argparse.ArgumentParser(description="ARIMA Model for Stock Price Forecasting")
    parser.add_argument('-s', '--symbols', nargs='+', default=default_symbols,
                        help="List of stock symbols (default: ['AAPL', 'JANX','PCG', 'PLTR'])")
    parser.add_argument('-o', '--order', type=int, nargs=3, default=default_order,
                        help="ARIMA order (p, d, q), default: (1, 1, 1)")
    args = parser.parse_args()

    symbols = args.symbols
    arima_order = tuple(args.order)

    # Loop through each symbol
    for symbol in symbols:
        try:
            print(f"\nProcessing symbol: {symbol}")

            # Load data
            series = load_data(symbol)

            # Split into training and testing sets
            split_idx = int(len(series) * (1 - test_ratio))
            train_series = series[:split_idx]
            test_series = series[split_idx:]

            # Check for stationarity and preprocess
            if not test_stationary(train_series):
                train_series = difference_series(train_series)

            # Train ARIMA model
            print(f"Training ARIMA model for {symbol} with order {arima_order}...")
            model_fit = train_model(train_series, arima_order)

            # Evaluate the model
            print(f"Evaluating ARIMA model for {symbol}...")
            forecast, rmse = evaluate_model(model_fit, test_series)

            # Plot the results
            print(f"Plotting forecast for {symbol}...")
            plot_forecast(train_series, test_series, forecast)

            # Print RMSE
            print(f"Symbol: {symbol}, RMSE: {rmse}")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")