"""
Inference Script for SimpleRNN Traffic Model
=============================================
Loads saved model + scaler, creates same features as training,
and predicts the next timestep traffic volume.
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rnn_traffic_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "rnn_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "outputs", "feature_engineered_dataset.csv")
DEFAULT_SEQUENCE_LENGTH = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load saved model and scaler artefacts."""
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    return model, scaler_data


def get_sequence_length(scaler_data):
    """Read sequence length from saved scaler."""
    return scaler_data.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)


def detect_target(df):
    """Detect target column."""
    for candidate in ("traffic_volume", "Vehicles"):
        if candidate in df.columns:
            return candidate
    return df.select_dtypes(include=[np.number]).columns[-1]


def create_optimal_features(df, target_col):
    """
    Create SAME features as training script.
    Must match exactly what rnn_train_improved.py creates.
    """
    result = pd.DataFrame()
    
    # 1. IMMEDIATE LAGS
    for i in range(1, 13):
        result[f'lag_{i}'] = df[target_col].shift(i)
    
    # 2. SAME-HOUR PATTERNS
    result['lag_24'] = df[target_col].shift(24)
    result['lag_48'] = df[target_col].shift(48)
    result['lag_168'] = df[target_col].shift(168)
    
    # 3. ROLLING MEANS
    for w in [3, 6, 12, 24]:
        result[f'rmean_{w}'] = df[target_col].rolling(w, min_periods=1).mean()
    
    # 4. ROLLING STD
    for w in [6, 12, 24]:
        result[f'rstd_{w}'] = df[target_col].rolling(w, min_periods=1).std()
    
    # 5. DIFFERENCES
    result['diff_1'] = df[target_col].diff(1)
    result['diff_2'] = df[target_col].diff(2)
    result['diff_24'] = df[target_col].diff(24)
    
    # 6. TEMPORAL FEATURES
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
    
    return result


def prepare_input(df, scaler_data, n_rows):
    """Prepare the last n_rows as a model-ready input tensor."""
    target_col = scaler_data["target_col"]
    feature_scaler = scaler_data["feature_scaler"]
    feature_names = scaler_data.get("feature_names", [])
    
    # Sort by datetime if exists
    datetime_col = None
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(10))
                datetime_col = col
                break
            except:
                continue
    
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)
    
    # Create features same as training
    features_df = create_optimal_features(df, target_col)
    
    # Ensure same feature order as training
    if feature_names:
        # Only use features that exist
        available = [f for f in feature_names if f in features_df.columns]
        features_df = features_df[available]
    
    # Get last n_rows
    X = features_df.tail(n_rows).values
    
    # Scale
    X_scaled = feature_scaler.transform(X)
    return X_scaled.reshape(1, n_rows, -1)


def predict_next(model, X, scaler_data):
    """Run prediction and inverse-transform to original scale."""
    target_scaler = scaler_data["target_scaler"]
    pred_scaled = model.predict(X, verbose=0).flatten()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return max(0, pred[0])  # Traffic can't be negative


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  SimpleRNN — Traffic Volume Inference")
    print("=" * 50)

    model, scaler_data = load_artifacts()
    seq_len = get_sequence_length(scaler_data)
    print(f"Model loaded : {MODEL_PATH}")
    print(f"Scaler loaded: {SCALER_PATH}")
    print(f"Sequence len : {seq_len}")
    print(f"Features     : {len(scaler_data.get('feature_names', []))}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset rows : {len(df)}")

    X = prepare_input(df, scaler_data, seq_len)
    print(f"Input shape  : {X.shape}")

    prediction = predict_next(model, X, scaler_data)
    print(f"\n>>> Predicted next traffic volume: {prediction:.2f}")


if __name__ == "__main__":
    main()
