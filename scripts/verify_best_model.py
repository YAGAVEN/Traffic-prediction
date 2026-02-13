"""
Verify which saved model produces the best test predictions
"""

import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

print("="*80)
print("VERIFYING WHICH MODEL IS BEST")
print("="*80)

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_units[0], batch_first=True))
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(nn.LSTM(hidden_units[i-1], hidden_units[i], batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units[-1], 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class TrafficDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(data, target, seq_length=24):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(target[i+seq_length])
    return np.array(sequences), np.array(targets)

def evaluate_model(model_path, model_name):
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Load data
        df = pd.read_csv('../outputs/feature_engineered_dataset.csv')
        feature_cols = [c for c in df.columns if c not in ['DateTime', 'ID', 'Vehicles', 'Junction']]
        junction_dummies = pd.get_dummies(df['Junction'], prefix='junction')
        df = pd.concat([df, junction_dummies], axis=1)
        feature_cols.extend(junction_dummies.columns.tolist())
        
        # Load scalers
        with open('../models/feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../models/target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Prepare data
        X = df[feature_cols].values
        y = df['Vehicles'].values
        X_scaled = scaler.transform(X)
        y_scaled = target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences, targets = create_sequences(X_scaled, y_scaled, 24)
        
        # Test split
        train_size = int(len(sequences) * 0.8)
        test_sequences = sequences[train_size:]
        test_targets = targets[train_size:]
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('config', {'hidden_units': [128, 64], 'dropout': 0.2})
        
        model = LSTMModel(sequences.shape[2], 
                         model_config.get('hidden_units', [128, 64]),
                         model_config.get('dropout', 0.2))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        test_dataset = TrafficDataset(test_sequences, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        predictions, actuals = [], []
        with torch.no_grad():
            for seq_batch, tgt_batch in test_loader:
                outputs = model(seq_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(tgt_batch.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform
        predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_orig = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Metrics
        mse = mean_squared_error(actuals_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_orig, predictions_orig)
        r2 = r2_score(actuals_orig, predictions_orig)
        
        negative_preds = np.sum(predictions_orig < 0)
        
        print(f"  Test Samples:        {len(actuals_orig)}")
        print(f"  R² Score:            {r2:.4f}")
        print(f"  RMSE:                {rmse:.4f}")
        print(f"  MAE:                 {mae:.4f}")
        print(f"  MSE:                 {mse:.4f}")
        print(f"  Negative Predictions: {negative_preds}")
        print(f"  Median Abs Error:    {np.median(np.abs(actuals_orig - predictions_orig)):.2f}")
        
        return {
            'model': model_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'negative_preds': negative_preds,
            'samples': len(actuals_orig)
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# Test both models
results = []

models_to_test = [
    ('../models/lstm_traffic_model_final.pth', 'lstm_traffic_model_final.pth'),
    ('../models/best_model.pth', 'best_model.pth')
]

for model_path, model_name in models_to_test:
    result = evaluate_model(model_path, model_name)
    if result:
        results.append(result)

# Compare
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

for result in results:
    print(f"\n{result['model']}:")
    print(f"  R²:   {result['r2']:.4f}")
    print(f"  RMSE: {result['rmse']:.4f}")
    print(f"  MAE:  {result['mae']:.4f}")
    print(f"  Negative Predictions: {result['negative_preds']}")

if len(results) == 2:
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    # Compare with original best predictions
    original_best = {
        'model': 'test_predictions.csv (original)',
        'r2': 0.5861,
        'rmse': 6.4126,
        'mae': 4.4273
    }
    
    print(f"\nOriginal Best (test_predictions.csv):")
    print(f"  R²:   {original_best['r2']:.4f}")
    print(f"  RMSE: {original_best['rmse']:.4f}")
    print(f"  MAE:  {original_best['mae']:.4f}")
    
    # Find which current model matches best
    for result in results:
        if abs(result['r2'] - original_best['r2']) < 0.01 and abs(result['mae'] - original_best['mae']) < 0.1:
            print(f"\n✅ MATCH FOUND: {result['model']}")
            print(f"   This is the model that generated test_predictions.csv")
            print(f"   STATUS: ✓ Best model is STORED and AVAILABLE")
            break
    else:
        print(f"\n⚠️  No exact match found.")
        best_current = max(results, key=lambda x: x['r2'])
        print(f"\nBest current model: {best_current['model']}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
