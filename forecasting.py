# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- STEP 1: LOAD THE DATA ----------
# Replace 'sales_data.csv' with your actual dataset file path
data = pd.read_csv('sales_data.csv')

# Check the first few rows
print("Dataset preview:")
print(data.head())

# Ensure the dataset has 'ds' (date) and 'y' (sales) columns
data.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# ---------- STEP 2: VISUALIZE SALES DATA ----------
plt.figure(figsize=(12, 6))
plt.plot(data['ds'], data['y'], label='Sales', color='blue')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend Over Time')
plt.legend()
plt.grid()
plt.show()

# ---------- STEP 3: TRAIN PROPHET MODEL ----------
# Initialize and fit the Prophet model
model = Prophet()
model.fit(data)

# Create future dates for prediction (next 90 days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Visualize the forecast
fig1 = model.plot(forecast)
plt.show()

# Visualize the seasonality components
fig2 = model.plot_components(forecast)
plt.show()

# ---------- STEP 4: MODEL EVALUATION ----------
# Split the data into train and test sets
train = data[:-30]  # Use all but the last 30 days for training
test = data[-30:]   # Use the last 30 days for testing

# Train the model again on the training set
model = Prophet()
model.fit(train)

# Make future predictions for the test set
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Calculate MAE and RMSE
y_true = test['y'].values
y_pred = forecast['yhat'][-30:].values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

# Print the accuracy metrics
print(f'\nModel Evaluation:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# ---------- STEP 5: SAVE THE RESULTS ----------
# Save the forecast to a CSV file
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_results.csv', index=False)
print("\nForecast saved as 'forecast_results.csv'")
