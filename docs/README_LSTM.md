# Traffic Forecasting with LSTM - Complete Pipeline

A comprehensive PyTorch-based LSTM implementation for traffic volume prediction with extensive feature engineering.

## Features

### Feature Engineering (30+ Features)
- **Temporal Features**: hour, day_of_week, month, day_of_month, week_of_year, quarter, is_weekend
- **Lag Features**: lag_1, lag_2, lag_3 (previous traffic values)
- **Rolling Statistics**: 3h and 6h rolling mean/std
- **Junction-Specific**: junction mean, std, and ratio features
- **Cyclical Encoding**: sin/cos transformations for hour and day
- **Custom Flags**: rush_hour, night, peak_day, holiday
- **Scaling**: StandardScaler or MinMaxScaler support
- **Junction Encoding**: One-hot encoding for multiple junctions

### LSTM Model Architecture
- **Multi-layer LSTM**: 2-3 configurable layers (default: 128, 64 hidden units)
- **Regularization**: Dropout (0.2-0.3) between layers
- **Activation**: ReLU for hidden layers, Linear for output
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam with configurable learning rate
- **Early Stopping**: Patience-based with validation monitoring
- **Checkpointing**: Saves best model during training

### Evaluation & Outputs
- **Metrics**: MSE, RMSE, MAE, R²
- **Visualizations**: 
  - Training/validation loss curves
  - Predicted vs actual scatter plot
  - Time series comparison
  - Residual analysis
- **Saved Artifacts**:
  - Feature-engineered CSV
  - Trained model checkpoints
  - Scalers (feature and target)
  - Metrics JSON
  - Prediction results

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Execution
```bash
python traffic_lstm_pipeline.py
```

### Configuration
Edit the `CONFIG` dictionary in `traffic_lstm_pipeline.py`:

```python
CONFIG = {
    'data_path': './raw-data/traffic.csv',
    'sequence_length': 24,          # Look-back window (hours)
    'prediction_horizon': 1,         # Predict next N hours
    'train_ratio': 0.8,              # 80% train, 20% test
    'hidden_units': [128, 64],       # LSTM layer sizes
    'dropout': 0.2,                  # Dropout rate
    'batch_size': 64,                # Training batch size
    'epochs': 150,                   # Maximum epochs
    'learning_rate': 0.001,          # Adam learning rate
    'patience': 10,                  # Early stopping patience
    'scaler_type': 'standard',       # 'standard' or 'minmax'
}
```

## Dataset Format

Expected CSV format:
```
DateTime,Junction,Vehicles,ID
2015-11-01 00:00:00,1,15,20151101001
2015-11-01 01:00:00,1,13,20151101011
```

**Columns**:
- `DateTime`: Timestamp (any parseable format)
- `Junction`: Junction ID (integer)
- `Vehicles`: Traffic count (integer)
- `ID`: Unique identifier (string)

## Output Structure

```
Traffic-Prediction/
├── raw-data/
│   └── traffic.csv
├── outputs/
│   ├── feature_engineered_dataset.csv
│   ├── evaluation_metrics.json
│   ├── prediction_visualization.png
│   ├── residual_analysis.png
│   └── test_predictions.csv
├── models/
│   ├── best_model.pth
│   ├── lstm_traffic_model_final.pth
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
└── traffic_lstm_pipeline.py
```

## Pipeline Stages

### Stage 1: Feature Engineering
1. Load and parse datetime
2. Extract temporal features
3. Create lag features (per junction)
4. Calculate rolling statistics
5. Compute junction-specific features
6. Apply cyclical encoding
7. Add custom flags
8. Handle missing values (forward fill)
9. Scale features
10. Encode categorical variables

### Stage 2: Model Training
1. Create sequences (samples, sequence_length, features)
2. Time-based train/val/test split (no shuffling)
3. Build multi-layer LSTM
4. Train with early stopping
5. Save best model checkpoint

### Stage 3: Evaluation
1. Load best model
2. Predict on test set
3. Calculate metrics (MSE, RMSE, MAE, R²)
4. Generate visualizations
5. Save results

## Model Inference (Example)

```python
import torch
import pickle
import pandas as pd
import numpy as np

# Load model
checkpoint = torch.load('models/lstm_traffic_model_final.pth')
model = LSTMModel(...)  # Initialize with same architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scalers
with open('models/feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('models/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

# Prepare new data
# ... (apply same feature engineering steps)
X_new_scaled = feature_scaler.transform(X_new)

# Create sequences and predict
sequences = create_sequences(X_new_scaled, ...)
with torch.no_grad():
    predictions_scaled = model(torch.FloatTensor(sequences))
    predictions = target_scaler.inverse_transform(predictions_scaled.numpy())
```

## Performance Notes

- **GPU**: Automatically uses CUDA if available
- **Training Time**: ~5-15 minutes on CPU (depends on dataset size)
- **Memory**: Scales with sequence_length and batch_size
- **Reproducibility**: Fixed random seeds (42) for consistent results

## Customization

### Adding More Features
Add custom features in `load_and_engineer_features()`:
```python
df['my_feature'] = df['Vehicles'].shift(7)  # Weekly lag
```

### Changing Model Architecture
Modify hidden units in CONFIG:
```python
'hidden_units': [256, 128, 64],  # 3-layer LSTM
```

### Hyperparameter Tuning
Consider tuning:
- `sequence_length`: 12, 24, 48, 168 (hours)
- `hidden_units`: [64, 32], [128, 64], [256, 128, 64]
- `dropout`: 0.1, 0.2, 0.3, 0.5
- `learning_rate`: 0.0001, 0.001, 0.01
- `batch_size`: 32, 64, 128

## Troubleshooting

**Out of Memory**: Reduce `batch_size` or `sequence_length`

**Poor Performance**: 
- Increase `hidden_units`
- Add more epochs
- Try different `sequence_length`
- Check feature scaling

**Overfitting**: 
- Increase `dropout`
- Reduce model complexity
- Add more training data

**Underfitting**:
- Increase model complexity
- Reduce `dropout`
- Train longer

## Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

## License
MIT License

## Author
Traffic Forecasting LSTM Pipeline - 2026
