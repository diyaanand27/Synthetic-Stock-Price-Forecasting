import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_cleaned_portfolio(file_path):
    """Load the cleaned portfolio data."""
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    # Ensure the index is a DatetimeIndex with a frequency
    if not df.index.freq:
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq='B')  # Assume business days
    return df.squeeze()  # Convert DataFrame to Series if single column

def difference_series(series, d=1):
    """Difference the series to make it stationary."""
    return series.diff(periods=d).dropna()

def evaluate_arima(series, p_values, d_values, q_values):
    """Evaluate ARIMA models with different (p, d, q) parameters."""
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    # Fit ARIMA model
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=10)
                    # Manually align forecast index with the original series
                    forecast.index = series.index[-len(forecast):]
                    rmse = np.sqrt(mean_squared_error(series[-10:], forecast))
                    if rmse < best_score:
                        best_score, best_cfg = rmse, (p, d, q)
                    print(f"ARIMA{(p, d, q)} RMSE: {rmse}")
                except Exception as e:
                    print(f"ARIMA{(p, d, q)} failed: {e}")
    print(f"Best ARIMA{best_cfg} RMSE: {best_score}")
    return best_cfg

def plot_forecast(series, forecast, title="ARIMA Forecast"):
    """Plot actual vs forecasted portfolio values."""
    forecast.index = series.index[-len(forecast):]  # Align forecast index
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series, label="Actual Portfolio Values", color="blue")
    plt.plot(forecast.index, forecast, label="Forecast", color="red", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

if __name__ == "__main__":
    # Path to the cleaned portfolio data
    file_path = "data/raw/portfolio.csv"  # Adjust this to your file path
    portfolio_data = load_cleaned_portfolio(file_path)

    # Extract the portfolio value series
    portfolio_value = portfolio_data

    # Manually differencing the data
    print("Differencing the series...")
    portfolio_diff = difference_series(portfolio_value)

    # Parameter grid for ARIMA (manual tuning)
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    # Find the best ARIMA parameters
    print("Evaluating ARIMA models...")
    best_order = evaluate_arima(portfolio_diff, p_values, d_values, q_values)

    # Train the best ARIMA model
    print(f"Training ARIMA model with best order {best_order}...")
    model = ARIMA(portfolio_diff, order=best_order)
    model_fit = model.fit()

    # Forecast future portfolio values
    forecast = model_fit.forecast(steps=10)
    forecast.index = portfolio_value.index[-len(forecast):]  # Align forecast index

    # Plot the forecast
    print("Plotting the forecast...")
    plot_forecast(portfolio_value, forecast)
