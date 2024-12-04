import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima

# Helper Functions
def load_cleaned_portfolio(file_path):
    """Load the cleaned portfolio data."""
    return pd.read_csv(file_path, index_col="Date", parse_dates=True)

def difference_series(series, d=1):
    """Difference the series to make it stationary."""
    return series.diff(periods=d).dropna()

def optimize_arima(series):
    """Find the best ARIMA order using auto_arima."""
    print("Optimizing ARIMA parameters with auto_arima...")
    stepwise_fit = auto_arima(series,
                              seasonal=False,  # No seasonality in basic ARIMA
                              trace=True,      # Display results of each configuration
                              suppress_warnings=True,
                              error_action='ignore',
                              stepwise=True)  # Use a stepwise search for faster results
    print(f"Optimal ARIMA order: {stepwise_fit.order}")
    return stepwise_fit.order

def train_arima_model(series, order):
    """Train the ARIMA model."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def evaluate_arima_model(model_fit, test_series):
    """Evaluate the ARIMA model on test data."""
    forecast = model_fit.forecast(steps=len(test_series))
    forecast.index = test_series.index  # Align forecast index with test data
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    print(f"RMSE: {rmse}")
    return forecast, rmse

def plot_forecast(train_series, test_series, forecast):
    """Plot the train, test, and forecast data."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_series, label="Training Data")
    plt.plot(test_series, label="Testing Data")
    plt.plot(forecast, label="Forecast", linestyle="--")
    plt.legend()
    plt.title("ARIMA Model Forecast for Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Configuration
    portfolio_file = "data/raw/portfolio.csv"  # Path to the cleaned portfolio data
    test_ratio = 0.2  # 20% of data for testing

    # Load cleaned portfolio data
    print(f"Loading cleaned portfolio data from {portfolio_file}...")
    portfolio_data = load_cleaned_portfolio(portfolio_file)

    # Extract the portfolio value series
    portfolio_value = portfolio_data.squeeze()  # Convert to Series if it's a single column

    # Split into training and testing sets
    split_idx = int(len(portfolio_value) * (1 - test_ratio))
    train_series = portfolio_value[:split_idx]
    test_series = portfolio_value[split_idx:]

    # Ensure proper datetime index
    train_series.index = pd.to_datetime(train_series.index)
    test_series.index = pd.to_datetime(test_series.index)

    # Stationarity Check and Differencing
    print("Differencing the training series...")
    train_series_diff = difference_series(train_series)

    # Find Optimal ARIMA Order
    optimal_order = optimize_arima(train_series_diff)

    # Train ARIMA Model
    print(f"Training ARIMA model with optimal order {optimal_order}...")
    model_fit = train_arima_model(train_series_diff, optimal_order)

    # Evaluate the Model
    print("Evaluating the ARIMA model...")
    forecast, rmse = evaluate_arima_model(model_fit, test_series)

    # Plot the Forecast
    print("Plotting the forecast...")
    plot_forecast(train_series, test_series, forecast)
