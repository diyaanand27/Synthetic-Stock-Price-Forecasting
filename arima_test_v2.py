import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Ensure data directory exists
output_dir = "./data/raw"
portfolio_file = os.path.join(output_dir, "portfolio.csv")

# Load data (Assume 'portfolio_values' contains the entire series)
portfolio_values = pd.read_csv(portfolio_file, index_col="Date", parse_dates=True).iloc[:, 0]

# Ensure consistent frequency
portfolio_values = portfolio_values.asfreq('B')


# Train-test split
train_data = portfolio_values['2022-01-01':'2022-12-31']  # Training on 2022
test_data = portfolio_values['2023-01-01':'2023-03-31']  # Testing on first 90 days of 2023

# Fit ARIMA model on training data
order = (2, 1, 2)  # Adjust as necessary
arima_model = ARIMA(train_data, order=order)
arima_result = arima_model.fit()

# Forecast for the test period
forecast_steps = len(test_data)  # Number of test points (90 days)
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean

# Align forecast with test data for comparison
forecast_index = test_data.index  # Use test data's index
forecast_values.index = forecast_index  # Align forecast index

# Evaluate model performance
mae = mean_absolute_error(test_data, forecast_values)
rmse = np.sqrt(mean_squared_error(test_data, forecast_values))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train_data, label="Training Data (2022)", color="blue")
plt.plot(test_data, label="Actual Test Data (2023)", color="green")
plt.plot(forecast_values, label="Forecasted Test Data (2023)", linestyle="--", color="red")
plt.legend()
plt.title("ARIMA - Train-Test Split Forecast")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.show()






"""import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import itertools

def check_stationarity(data):
    result = adfuller(data)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is not stationary. Consider differencing.")

def tune_arima_params(train):
    p = range(0, 3)  # Adjusted range for simplicity
    d = range(0, 2)
    q = range(0, 3)
    pdq_combinations = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_params = None

    for params in pdq_combinations:
        try:
            model = SARIMAX(train, order=params, enforce_stationarity=True, enforce_invertibility=True)
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_params = params
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    print(f"Best Parameters: {best_params}, AIC: {best_aic}")
    return best_params

if __name__ == "__main__":
    # Step 1: Load Portfolio Data
    portfolio_data = pd.read_csv("./data/raw/portfolio.csv", index_col=0, parse_dates=True)
    portfolio_data.index.freq = 'B'  # Ensure business day frequency

    # Step 2: Log Transformation
    portfolio_data_log = np.log(portfolio_data)

    # Step 3: Check Stationarity
    check_stationarity(portfolio_data_log)

    # Step 4: Differencing if Non-Stationary
    if adfuller(portfolio_data_log.values.flatten())[1] > 0.05:
        portfolio_data_diff = portfolio_data_log.diff().dropna()
    else:
        portfolio_data_diff = portfolio_data_log

    # Step 5: Split Data into Train and Test Sets
    train_size = int(len(portfolio_data_diff) * 0.8)
    train, test = portfolio_data_diff[:train_size], portfolio_data_diff[train_size:]

    # Step 6: Hyperparameter Tuning
    best_params = tune_arima_params(train)

    # Step 7: Train ARIMA Model
    model = SARIMAX(train, order=best_params, enforce_stationarity=True, enforce_invertibility=True)
    arima_model = model.fit(disp=False)

    # Step 8: Forecasting
    forecast = arima_model.forecast(steps=len(test))
    forecast_cumsum = forecast.cumsum() + portfolio_data_log.iloc[train_size - 1]  # Reverse differencing
    forecast_final = np.exp(forecast_cumsum)  # Reverse log transformation

    # Handle NaN by filling instead of dropping
    forecast_final = forecast_final.fillna(method='ffill').fillna(method='bfill')
    test = test.fillna(method='ffill').fillna(method='bfill')

    # Align indices of test and forecast_final
    test = test.loc[forecast_final.index]
    forecast_final = forecast_final.loc[test.index]

    # Check for empty datasets
    if test.empty or forecast_final.empty:
        print("Error: Test or forecast_final dataset is empty!")
        exit()

    # Step 9: Evaluate the Model
    rmse = np.sqrt(mean_squared_error(np.exp(test.cumsum()), forecast_final))
    mape = mean_absolute_percentage_error(np.exp(test.cumsum()), forecast_final)
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape * 100:.2f}%")

    # Step 10: Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(np.exp(train.cumsum()), label='Train Stock Price', color='green')
    plt.plot(np.exp(test.cumsum()), label='Real Stock Price', color='blue')
    plt.plot(test.index, forecast_final, label='Predicted Stock Price', color='red')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title('ARIMA Model Stock Price Prediction')
    plt.show()"""
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load data
portfolio_file = "./data/portfolio.csv"
portfolio_values = pd.read_csv(portfolio_file, index_col="Date", parse_dates=True)

# Use 'Portfolio Value' column for ARIMA
portfolio_values = portfolio_values['Portfolio Value']

# Ensure consistent frequency
portfolio_values = portfolio_values.asfreq('B')

# Check stationarity
print("Performing ADF Test...")
adf_result = adfuller(portfolio_values.dropna())
print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")

# Apply differencing if necessary
if adf_result[1] > 0.05:
    portfolio_values_diff = portfolio_values.diff().dropna()
else:
    portfolio_values_diff = portfolio_values

# Train-test split
train_data = portfolio_values_diff['2022-01-01':'2022-12-31']
test_data = portfolio_values_diff['2023-01-01':'2023-03-31']

# Fit ARIMA model
order = (2, 1, 2)  # Replace with optimized parameters if known
arima_model = ARIMA(train_data, order=order)
arima_result = arima_model.fit()

# Forecast
forecast_steps = len(test_data)
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean

# Align forecast with test data
forecast_values.index = test_data.index

# Evaluate
mae = mean_absolute_error(test_data, forecast_values)
rmse = np.sqrt(mean_squared_error(test_data, forecast_values))
print(f"MAE: {mae}, RMSE: {rmse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train_data.cumsum(), label="Training Data")
plt.plot(test_data.cumsum(), label="Actual Test Data", color="green")
plt.plot(forecast_values.cumsum(), label="Forecast", linestyle="--", color="red")
plt.legend()
plt.title("ARIMA - Train-Test Split with New Data")
plt.show()"""



