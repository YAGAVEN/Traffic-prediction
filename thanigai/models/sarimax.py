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
df = df.asfreq('H')
df = df.ffill()

# Target variable
y = df['Vehicles']

# ==============================
# 3️⃣ Create Exogenous Features
# ==============================

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

exog_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
X = df[exog_features]

# ==============================
# 4️⃣ Train-Test Split (80/20)
# ==============================

train_ratio = 0.8
train_size = int(len(y) * train_ratio)

y_train = y[:train_size]
y_test = y[train_size:]

X_train = X[:train_size]
X_test = X[train_size:]

# ==============================
# 5️⃣ Train SARIMAX Model
# ==============================

model = SARIMAX(
    y_train,
    exog=X_train,
    order=(1,1,1),
    seasonal_order=(1,1,1,24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_model = model.fit(disp=False)

print(sarimax_model.summary())

# ==============================
# 6️⃣ Forecast
# ==============================

forecast = sarimax_model.forecast(
    steps=len(y_test),
    exog=X_test
)

# ==============================
# 7️⃣ Evaluation Metrics
# ==============================

mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, forecast)

print("\n📊 SARIMAX Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ==============================
# 8️⃣ Plot (Last 7 Days)
# ==============================

plt.figure(figsize=(16,6))
plt.plot(y_test.index[-168:], y_test.values[-168:], label='Actual')
plt.plot(y_test.index[-168:], forecast.values[-168:], label='SARIMAX')
plt.legend()
plt.title("SARIMAX vs Actual (Last 7 Days)")
plt.grid(True)
plt.show()

# ==============================
# 9️⃣ Save Results to CSV
# ==============================

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': forecast.values
})

results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

results_df.to_csv('/content/sarimax_results.csv', index=False)

print("\nFile saved as sarimax_results.csv")
