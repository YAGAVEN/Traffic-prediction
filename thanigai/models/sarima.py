# ==============================
# 1️⃣ Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 2️⃣ Load and Prepare Data
# ==============================
df = pd.read_csv('/content/traffic.csv')

df['DateTime'] = pd.to_datetime(df['DateTime'])

# Use one junction (fair comparison)
df = df[df['Junction'] == 1].copy()

df.set_index('DateTime', inplace=True)

# Ensure hourly frequency
df = df.asfreq('H')
df = df.ffill()

y = df['Vehicles']

# ==============================
# 3️⃣ Train-Test Split (80/20)
# ==============================
train_ratio = 0.8
train_size = int(len(y) * train_ratio)

y_train = y[:train_size]
y_test = y[train_size:]

# ==============================
# 4️⃣ Train SARIMA Model
# ==============================
model = SARIMAX(
    y_train,
    order=(1,1,1),
    seasonal_order=(1,1,1,24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_model = model.fit(disp=False)

print(sarima_model.summary())

# ==============================
# 5️⃣ Forecast
# ==============================
forecast = sarima_model.forecast(steps=len(y_test))

# ==============================
# 6️⃣ Evaluation Metrics
# ==============================

# MAE
mae = mean_absolute_error(y_test, forecast)

# MSE
mse = mean_squared_error(y_test, forecast)

# RMSE
rmse = np.sqrt(mse)

# R²
r2 = r2_score(y_test, forecast)

print("\n📊 SARIMA Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ==============================
# 7️⃣ Plot (Last 7 Days)
# ==============================
plt.figure(figsize=(16,6))
plt.plot(y_test.index[-168:], y_test.values[-168:], label='Actual')
plt.plot(y_test.index[-168:], forecast.values[-168:], label='SARIMA')
plt.legend()
plt.title("SARIMA vs Actual (Last 7 Days)")
plt.grid(True)
plt.show()

# ==============================
# 8️⃣ Save Results to CSV
# ==============================
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': forecast.values
})

results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

results_df.to_csv('/content/sarima_results.csv', index=False)

print("\nFile saved as sarima_results.csv")
