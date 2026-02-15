# ==============================
# 1️⃣ Install Prophet (if needed)
# ==============================
# install this raaa in terminal
# pip install prophet

# ==============================
# 2️⃣ Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 3️⃣ Load and Prepare Data
# ==============================
df = pd.read_csv('/content/traffic.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Use one junction (same as SARIMAX for fair comparison)
df = df[df['Junction'] == 1].copy()

df = df.sort_values('DateTime')

# Prophet requires columns: ds (date), y (target)
prophet_df = df[['DateTime', 'Vehicles']].rename(
    columns={'DateTime': 'ds', 'Vehicles': 'y'}
)

# ==============================
# 4️⃣ Train-Test Split (80/20)
# ==============================

train_ratio = 0.8
train_size = int(len(prophet_df) * train_ratio)

train_df = prophet_df.iloc[:train_size]
test_df = prophet_df.iloc[train_size:]

# ==============================
# 5️⃣ Train Prophet Model
# ==============================

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)

model.fit(train_df)

# ==============================
# 6️⃣ Forecast on Test Set
# ==============================

future = test_df[['ds']]   # only dates
forecast = model.predict(future)

y_test = test_df['y'].values
y_pred = forecast['yhat'].values

# ==============================
# 7️⃣ Evaluation Metrics
# ==============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Prophet Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ==============================
# 8️⃣ Plot (Last 7 Days)
# ==============================

plt.figure(figsize=(16,6))
plt.plot(test_df['ds'].values[-168:], y_test[-168:], label='Actual')
plt.plot(test_df['ds'].values[-168:], y_pred[-168:], label='Prophet')
plt.legend()
plt.title("Prophet vs Actual (Last 7 Days)")
plt.grid(True)
plt.show()

# ==============================
# 9️⃣ Save Results to CSV
# ==============================

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

results_df.to_csv('/content/prophet_results.csv', index=False)

print("\nFile saved as prophet_results.csv")
