"""
ULTIMATE SimpleRNN Traffic Forecasting Model
=============================================
KEY INSIGHT: For time-series, the PREVIOUS VALUE is the strongest predictor.
This model uses aggressive feature engineering to compensate for SimpleRNN limitations.

Strategy:
1. Create DIRECT lag features (t-1, t-2, ... are THE BEST predictors)
2. Use VERY short sequences (SimpleRNN has vanishing gradients)
3. Use WIDE layers to capture complex patterns
4. Focus on MSE loss directly
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "feature_engineered_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rnn_traffic_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "rnn_scaler.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "rnn_predictions_improved.csv")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "rnn_best_metrics.json")

# CRITICAL: Short sequence for SimpleRNN (vanishing gradient problem)
INPUT_SEQUENCE_LENGTH = 12  # Very short - SimpleRNN works best this way
TRAIN_RATIO = 0.8
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path):
    """Load dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Find datetime column
    datetime_col = None
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(100))
                datetime_col = col
                break
            except:
                continue
    
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        if "Junction" in df.columns:
            df = df.sort_values(["Junction", datetime_col]).reset_index(drop=True)
        else:
            df = df.sort_values(datetime_col).reset_index(drop=True)
    
    return df, datetime_col


def detect_target(df):
    """Detect target column."""
    for candidate in ("traffic_volume", "Vehicles"):
        if candidate in df.columns:
            return candidate
    return df.select_dtypes(include=[np.number]).columns[-1]


# ---------------------------------------------------------------------------
# CRITICAL: Feature Engineering for Low MSE
# ---------------------------------------------------------------------------
def create_optimal_features(df, target_col):
    """
    Create features that DIRECTLY predict target.
    KEY: Lag features are THE MOST IMPORTANT for time series.
    """
    print("\n=== Creating Optimal Features ===")
    
    # Start fresh - only keep essential columns
    result = pd.DataFrame()
    
    # 1. IMMEDIATE LAGS - Most important features!
    print("Creating lag features...")
    for i in range(1, 13):  # lag_1 to lag_12
        result[f'lag_{i}'] = df[target_col].shift(i)
    
    # 2. SAME-HOUR PATTERNS (daily seasonality)
    result['lag_24'] = df[target_col].shift(24)   # Same hour yesterday
    result['lag_48'] = df[target_col].shift(48)   # Same hour 2 days ago
    result['lag_168'] = df[target_col].shift(168) # Same hour last week
    
    # 3. ROLLING MEANS - Trend indicators
    for w in [3, 6, 12, 24]:
        result[f'rmean_{w}'] = df[target_col].rolling(w, min_periods=1).mean()
    
    # 4. ROLLING STD - Volatility
    for w in [6, 12, 24]:
        result[f'rstd_{w}'] = df[target_col].rolling(w, min_periods=1).std()
    
    # 5. DIFFERENCES - Momentum
    result['diff_1'] = df[target_col].diff(1)
    result['diff_2'] = df[target_col].diff(2)
    result['diff_24'] = df[target_col].diff(24)
    
    # 6. TEMPORAL FEATURES (from datetime if exists)
    if 'hour' in df.columns:
        result['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if 'day_of_week' in df.columns:
        result['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    if 'is_weekend' in df.columns:
        result['is_weekend'] = df['is_weekend']
    if 'is_rush_hour' in df.columns:
        result['is_rush_hour'] = df['is_rush_hour']
    
    # 7. RATIO FEATURES
    result['ratio_to_mean24'] = df[target_col] / (result['rmean_24'] + 1)
    result['ratio_to_lag1'] = df[target_col] / (result['lag_1'] + 1)
    
    # Fill NaN
    result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)
    result = result.replace([np.inf, -np.inf], 0)
    
    print(f"Created {len(result.columns)} features")
    return result


# ---------------------------------------------------------------------------
# Scaling & Sequences
# ---------------------------------------------------------------------------
def normalize(X, y):
    """MinMaxScaler for both features and target."""
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    X_s = feat_scaler.fit_transform(X)
    
    tgt_scaler = MinMaxScaler(feature_range=(0, 1))
    y_s = tgt_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_s, y_s, feat_scaler, tgt_scaler


def create_sequences(X, y, window):
    """Create sequences for RNN."""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


# ---------------------------------------------------------------------------
# Model - Optimized for Minimum MSE
# ---------------------------------------------------------------------------
def build_model(input_shape, n_features):
    """
    Wide SimpleRNN model optimized for low MSE.
    Uses very wide layers to compensate for SimpleRNN limitations.
    """
    model = Sequential([
        # Layer 1: Very wide to capture all patterns
        SimpleRNN(256, return_sequences=True, input_shape=input_shape,
                  activation='tanh', recurrent_dropout=0.0),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 2: Still wide
        SimpleRNN(128, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Layer 3: Final RNN
        SimpleRNN(64, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        
        # Dense head - very wide for complex mapping
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(actual, predicted):
    """Compute all evaluation metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    denom = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denom == 0, 0.0, 
                             2.0 * np.abs(actual - predicted) / denom)) * 100
    
    within_10 = np.mean(np.abs(actual - predicted) <= 0.10 * np.abs(actual + 1)) * 100
    within_20 = np.mean(np.abs(actual - predicted) <= 0.20 * np.abs(actual + 1)) * 100
    
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "MAPE": mape, "SMAPE": smape,
        "Accuracy_10pct": within_10, "Accuracy_20pct": within_20,
    }


def print_metrics(metrics):
    """Print metrics in clean format."""
    print("\n" + "=" * 60)
    print("  🎯 EVALUATION METRICS")
    print("=" * 60)
    for name, val in metrics.items():
        marker = "✅" if (name == "MSE" and val < 100) or (name == "R2" and val > 0.9) else "  "
        print(f"  {marker} {name:>16s} : {val:.4f}")
    print("=" * 60)


def save_predictions(datetimes, actual, predicted, path):
    """Save predictions CSV."""
    pd.DataFrame({
        "datetime": datetimes,
        "actual": actual,
        "predicted": predicted,
        "residual": actual - predicted,
        "abs_error": np.abs(actual - predicted),
    }).to_csv(path, index=False)
    print(f"Predictions saved → {path}")


def save_metrics_json(metrics, path):
    """Save metrics as JSON."""
    with open(path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics JSON saved → {path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  🚀 ULTIMATE SimpleRNN Model — Traffic Forecasting")
    print("  🎯 Goal: Minimize MSE with optimal feature engineering")
    print("=" * 65)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Load data
    df, datetime_col = load_data(DATA_PATH)
    target_col = detect_target(df)
    print(f"Target: {target_col}")
    
    # Store datetimes for output
    if datetime_col:
        datetimes = df[datetime_col].values
    else:
        datetimes = np.arange(len(df))
    
    # Create optimal features
    features_df = create_optimal_features(df, target_col)
    X = features_df.values
    y = df[target_col].values
    feature_names = features_df.columns.tolist()
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Scale
    X_scaled, y_scaled, feat_scaler, tgt_scaler = normalize(X, y)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, INPUT_SEQUENCE_LENGTH)
    print(f"Sequences: X={X_seq.shape}, y={y_seq.shape}")
    
    # Time-series split (no shuffling!)
    n = len(X_seq)
    train_end = int(n * TRAIN_RATIO)
    val_start = int(train_end * 0.85)
    
    X_train = X_seq[:val_start]
    y_train = y_seq[:val_start]
    X_val = X_seq[val_start:train_end]
    y_val = y_seq[val_start:train_end]
    X_test = X_seq[train_end:]
    y_test = y_seq[train_end:]
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]), len(feature_names))
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=15, min_lr=1e-7, verbose=1
        ),
    ]
    
    # Train
    print("\n🏋️ Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    )
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Clip negative predictions (traffic can't be negative)
    y_pred = np.clip(y_pred, 0, None)
    
    # Metrics
    metrics = compute_metrics(y_actual, y_pred)
    print_metrics(metrics)
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({
            'feature_scaler': feat_scaler,
            'target_scaler': tgt_scaler,
            'feature_names': feature_names,
            'target_col': target_col,
            'sequence_length': INPUT_SEQUENCE_LENGTH,
        }, f)
    print(f"Scaler saved → {SCALER_PATH}")
    
    # Save predictions
    test_datetimes = datetimes[train_end + INPUT_SEQUENCE_LENGTH:
                               train_end + INPUT_SEQUENCE_LENGTH + len(y_actual)]
    save_predictions(test_datetimes, y_actual, y_pred, OUTPUT_PATH)
    save_metrics_json(metrics, METRICS_PATH)
    
    print("\n" + "=" * 65)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Final MSE:  {metrics['MSE']:.2f}")
    print(f"  Final RMSE: {metrics['RMSE']:.2f}")
    print(f"  Final R²:   {metrics['R2']:.4f}")
    print(f"  Final MAE:  {metrics['MAE']:.2f}")
    
    if metrics['MSE'] < 100:
        print("\n  🎉 TARGET ACHIEVED: MSE < 100!")
    elif metrics['MSE'] < 200:
        print("\n  📈 Good progress! MSE < 200")
    else:
        print(f"\n  📊 Current MSE: {metrics['MSE']:.2f} - needs improvement")


if __name__ == "__main__":
    main()
