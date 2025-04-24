# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ---------- STEP 1: LOAD THE DATA ----------
file_path = r'C:\Users\disha\Documents\SALES FORECASTING FOR RETAIL BUSINESS\sales_data_sample.csv'

if os.path.exists(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("Column names in the CSV file:")
    print(data.columns)

    print("File loaded successfully!")
else:
    print(f"Error: {file_path} not found!")
    exit()

# Show preview
print("\nDataset preview:")
print(data.head())

# ---------- STEP 2: DETECT AND RENAME COLUMNS FOR PROPHET ----------
# Try to identify the date and sales column
date_col = None
for col in data.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col and 'SALES' in data.columns:
    data.rename(columns={date_col: 'ds', 'SALES': 'y'}, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
else:
    print("Error: Could not find appropriate 'date' and 'sales' columns!")
    exit()

# Show new column names
print("\nColumns after renaming:")
print(data[['ds', 'y']].head())

# ---------- STEP 3: VISUALIZE SALES DATA ----------
plt.figure(figsize=(12, 6))
plt.plot(data['ds'], data['y'], label='Sales', color='blue', marker='o')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---------- STEP 4: TRAIN PROPHET MODEL ----------
model = Prophet()
model.fit(data)

# Create future dates for prediction (next 90 days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# ---------- STEP 5: VISUALIZE FORECAST ----------
model.plot(forecast)
plt.title('Sales Forecast (with 90-day prediction)')
plt.tight_layout()
plt.show()

# Seasonality breakdown
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# ---------- STEP 6: MODEL EVALUATION ----------
train = data[:-30]
test = data[-30:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

y_true = test['y'].values
y_pred = forecast['yhat'][-30:].values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)

print('\nModel Evaluation:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# ---------- STEP 7: ACTUAL vs PREDICTED ----------
plt.figure(figsize=(12,6))
plt.plot(test['ds'], y_true, label='Actual Sales', marker='o')
plt.plot(test['ds'], y_pred, label='Predicted Sales', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales (Last 30 Days)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- STEP 8: SAVE FORECAST ----------
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_results.csv', index=False)
print("\nForecast saved as 'forecast_results.csv'")
