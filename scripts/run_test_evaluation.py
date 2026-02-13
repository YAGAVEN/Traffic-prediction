"""
Run test evaluation on the trained LSTM model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from datetime import datetime
import os

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'data_path': '../outputs/feature_engineered_dataset.csv',
    'model_path': '../models/lstm_traffic_model_final.pth',
    'feature_scaler_path': '../models/feature_scaler.pkl',
    'target_scaler_path': '../models/target_scaler.pkl',
    'output_dir': '../outputs',
    'sequence_length': 24,
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

# Define LSTM Model (same as training)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_units[0], batch_first=True))
        
        # Additional LSTM layers
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(nn.LSTM(hidden_units[i-1], hidden_units[i], batch_first=True))
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units[-1], 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        x = x[:, -1, :]  # Take last time step
        x = self.fc(x)
        return x

# Dataset class
class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Create sequences
def create_sequences(data, target, seq_length=24):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = target[i+seq_length]
        sequences.append(seq)
        targets.append(label)
    
    return np.array(sequences), np.array(targets)

print("\n" + "="*80)
print("LOADING DATA AND MODEL")
print("="*80)

# Load engineered data
print(f"\nLoading engineered data from {CONFIG['data_path']}...")
df = pd.read_csv(CONFIG['data_path'])
print(f"Data shape: {df.shape}")

# Prepare features
feature_cols = [c for c in df.columns if c not in ['DateTime', 'ID', 'Vehicles', 'Junction']]

# One-hot encode Junction
junction_dummies = pd.get_dummies(df['Junction'], prefix='junction')
df = pd.concat([df, junction_dummies], axis=1)
feature_cols.extend(junction_dummies.columns.tolist())

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Load scalers
with open(CONFIG['feature_scaler_path'], 'rb') as f:
    scaler = pickle.load(f)
with open(CONFIG['target_scaler_path'], 'rb') as f:
    target_scaler = pickle.load(f)

print(f"Scalers loaded")

# Prepare data
X = df[feature_cols].values
y = df['Vehicles'].values

# Scale data
X_scaled = scaler.transform(X)
y_scaled = target_scaler.transform(y.reshape(-1, 1)).flatten()

# Create sequences
print(f"\nCreating sequences with length {CONFIG['sequence_length']}...")
sequences, targets = create_sequences(X_scaled, y_scaled, CONFIG['sequence_length'])
print(f"Sequences shape: {sequences.shape}")
print(f"Targets shape: {targets.shape}")

# Split data (exact same logic as training pipeline)
train_size = int(len(sequences) * 0.8)  # 80% of all data
val_size = int(train_size * 0.1)  # 10% of train for validation

# Use only test set (remaining 20% after train)
test_sequences = sequences[train_size:]
test_targets = targets[train_size:]

print(f"\nTest set size: {len(test_sequences)}")

# Create DataLoader
test_dataset = TrafficDataset(test_sequences, test_targets)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Load model
print(f"\nLoading model from {CONFIG['model_path']}...")
checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])

# Get model config from checkpoint
model_config = checkpoint.get('config', {'hidden_units': [128, 64], 'dropout': 0.2})
hidden_units = model_config.get('hidden_units', [128, 64])
dropout = model_config.get('dropout', 0.2)

# Initialize model
input_size = sequences.shape[2]
model = LSTMModel(input_size, hidden_units, dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(CONFIG['device'])
model.eval()

print(f"Model loaded successfully")
print(f"Model architecture: {hidden_units} hidden units, {dropout} dropout")

# Run evaluation
print("\n" + "="*80)
print("RUNNING TEST EVALUATION")
print("="*80)

predictions = []
actuals = []

with torch.no_grad():
    for sequences_batch, targets_batch in test_loader:
        sequences_batch = sequences_batch.to(CONFIG['device'])
        targets_batch = targets_batch.to(CONFIG['device'])
        
        outputs = model(sequences_batch)
        predictions.extend(outputs.cpu().numpy().flatten())
        actuals.extend(targets_batch.cpu().numpy())

# Convert to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# Inverse transform
predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(actuals_original, predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals_original, predictions_original)
r2 = r2_score(actuals_original, predictions_original)

print(f"\nTest Set Metrics:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# Save results with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'test_predictions_{timestamp}.csv'
output_path = os.path.join(CONFIG['output_dir'], output_filename)

results_df = pd.DataFrame({
    'Actual': actuals_original,
    'Predicted': predictions_original,
    'Residual': actuals_original - predictions_original,
    'Absolute_Error': np.abs(actuals_original - predictions_original)
})

results_df.to_csv(output_path, index=False)
print(f"\n✅ Test predictions saved to: {output_path}")

# Save metrics
metrics_filename = f'test_metrics_{timestamp}.json'
metrics_path = os.path.join(CONFIG['output_dir'], metrics_filename)

metrics = {
    'MSE': float(mse),
    'RMSE': float(rmse),
    'MAE': float(mae),
    'R2': float(r2),
    'test_samples': len(actuals_original),
    'model_path': CONFIG['model_path'],
    'timestamp': timestamp
}

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Test metrics saved to: {metrics_path}")

# Print summary statistics
print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"\nActual values:")
print(f"  Min:  {actuals_original.min():.2f}")
print(f"  Max:  {actuals_original.max():.2f}")
print(f"  Mean: {actuals_original.mean():.2f}")
print(f"  Std:  {actuals_original.std():.2f}")

print(f"\nPredicted values:")
print(f"  Min:  {predictions_original.min():.2f}")
print(f"  Max:  {predictions_original.max():.2f}")
print(f"  Mean: {predictions_original.mean():.2f}")
print(f"  Std:  {predictions_original.std():.2f}")

print(f"\nError statistics:")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  Root Mean Squared Error: {rmse:.4f}")
print(f"  Max Error: {np.max(np.abs(results_df['Residual'])):.2f}")
print(f"  Median Absolute Error: {np.median(np.abs(results_df['Residual'])):.2f}")

print("\n" + "="*80)
print("EVALUATION COMPLETED!")
print("="*80)
