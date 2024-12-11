"""
    CSCI 4170 Fall 2024
    Final Project: Synthetic stock price using Scikit-learn
    @Author: Diya Anand
    @Author: Facus Dokubo-Wizzdom
    Notes: Dat ahas been divided into raw and processed. Processed data has been minimized using MinMaxScaler.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Use ADF test to check stationarity and improve data for model through differencing
def make_stationary(series):

    d = 0  # Initial differencing order then continue differencing
    while True:

        result = adfuller(series.dropna())
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        if result[1] < 0.05:  # If p-value < 0.05, data is stationary
            print(f"Data is stationary after {d} differencing(s).")
            break
        else:
            print(f"Data is non-stationary. Now applying differencing (d={d + 1})...")
            series = series.diff().dropna()  # Apply differencing
            d += 1
    return series, d

# Function to train ARIMA model
def train_arima(series, order):
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA: {e}")
        return None

# Main method for ARIMA sequence prediction
def forecast_arima(model_fit, steps):
    try:
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None

# Main function for real vs simulated plot
def plot_real_vs_simulated(train_data, test_data, stocks):
    plt.figure(figsize=(14, 8))
    for stock in stocks:
        try:
            # Remove training and testing data for the stock
            train_series = train_data[stock]
            test_series = test_data[stock]

            # Make training data stationary
            stationary_data, d = make_stationary(train_series)

            # Train ARIMA model 
            model = train_arima(stationary_data, order=(1, d, 1))
            if not model:
                continue

            # Forecast
            forecast_steps = len(test_series)
            forecast = forecast_arima(model, steps=forecast_steps)
            if forecast is None:
                continue

            # Combine real and synthetic data
            simulated_series = pd.Series(forecast, index=test_series.index)

            # Plot real vs synthetic 
            plt.plot(train_series.index, train_series, label=f"Real {stock} (Train)", linewidth=0.8)
            plt.plot(test_series.index, test_series, label=f"Real {stock} (Test)", linewidth=0.8, linestyle="--")
            plt.plot(simulated_series.index, simulated_series, label=f"Simulated {stock}", linestyle="--")

        except Exception as e:
            print(f"Error processing {stock}: {e}")

    plt.title("Portfolio: Real vs Simulated Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="upper left", ncol=2)
    plt.grid()
    plt.show()

def load_multiple_files(folder_path, stocks):
    """
    Load data into a single DF
    """
    combined_data = pd.DataFrame()
    for stock in stocks:
        file_path = os.path.join(folder_path, f"{stock}.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        stock_data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        stock_data = stock_data[["Adj Close"]]  # Use only the 'Adj Close' column
        stock_data.rename(columns={"Adj Close": stock}, inplace=True)  # Rename column to stock name
        if combined_data.empty:
            combined_data = stock_data
        else:
            combined_data = combined_data.join(stock_data, how="outer")  # Merge on the Date index
    return combined_data

if __name__ == "__main__":
    # Folder paths for training and testing data
    train_folder = "./data/raw-2022"  # Update with your training folder path
    test_folder = "./data/raw-2023"   # Update with your testing folder path

    #load stocks
    stocks = ['JANX', 'MSFT', 'GOOG', 'PLTR', 'NVDA', 'INTC', 'PCG', 'SMCI', 'CRM', 'IBM']

    # Load training and testing data
    train_data = load_multiple_files(train_folder, stocks)
    test_data = load_multiple_files(test_folder, stocks)

    # Fill missing values: fillna is deprecated so use obj.fill or ffill
    train_data = train_data.asfreq("B").ffill()
    test_data = test_data.asfreq("B").ffill()

    print("Data loaded succesfully. ARIMA Implementation Complete.")
    # Plot real vs simulated prices
    plot_real_vs_simulated(train_data, test_data, stocks)
