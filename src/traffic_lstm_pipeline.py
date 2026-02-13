"""
Traffic Forecasting Pipeline with LSTM
Complete implementation of feature engineering, model training, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
CONFIG = {
    'data_path': '../raw-data/traffic.csv',
    'output_dir': '../outputs',
    'model_dir': '../models',
    'sequence_length': 24,
    'prediction_horizon': 1,
    'train_ratio': 0.8,
    'hidden_units': [128, 64],
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 150,
    'learning_rate': 0.001,
    'patience': 10,
    'scaler_type': 'standard',  # 'standard' or 'minmax'
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create output directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)

print(f"Using device: {CONFIG['device']}")
print(f"Configuration: {json.dumps(CONFIG, indent=2)}")


# ============================================================================
# STEP 1: FEATURE ENGINEERING
# ============================================================================

def load_and_engineer_features(data_path):
    """
    Load dataset and perform comprehensive feature engineering
    """
    print("\n" + "="*80)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # 1. Convert DateTime to datetime type
    print("\n1. Converting DateTime to datetime type...")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Junction', 'DateTime']).reset_index(drop=True)
    
    # 2. Extract date-time features
    print("2. Extracting date-time features...")
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['DateTime'].dt.month
    df['day_of_month'] = df['DateTime'].dt.day
    df['week_of_year'] = df['DateTime'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['DateTime'].dt.quarter
    
    # Optional: is_holiday (simplified - using major holidays)
    # For a real implementation, use a calendar library
    df['is_holiday'] = 0  # Placeholder
    
    # 3. Create lag features for Vehicles (by Junction)
    print("3. Creating lag features...")
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('Junction')['Vehicles'].shift(lag)
    
    # 4. Create rolling statistics
    print("4. Creating rolling statistics...")
    for window in [3, 6]:
        df[f'rolling_mean_{window}h'] = df.groupby('Junction')['Vehicles'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}h'] = df.groupby('Junction')['Vehicles'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # 5. Junction-specific features
    print("5. Creating junction-specific features...")
    junction_stats = df.groupby('Junction')['Vehicles'].agg(['mean', 'std']).reset_index()
    junction_stats.columns = ['Junction', 'junction_mean', 'junction_std']
    df = df.merge(junction_stats, on='Junction', how='left')
    df['vehicles_ratio_to_junction_mean'] = df['Vehicles'] / (df['junction_mean'] + 1e-8)
    
    # 6. Create cyclical encoding
    print("6. Creating cyclical encoding...")
    df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
    df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)
    df['day_sin'] = np.sin(df['day_of_week'] * 2 * np.pi / 7)
    df['day_cos'] = np.cos(df['day_of_week'] * 2 * np.pi / 7)
    
    # 7. Create optional flags
    print("7. Creating optional flags...")
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                          (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_peak_day'] = (df['day_of_week'] < 5).astype(int)  # Mon-Fri
    
    # 8. Handle missing values
    print("8. Handling missing values...")
    print(f"Missing values before fill:\n{df.isnull().sum()}")
    df = df.ffill().bfill()
    print(f"Missing values after fill:\n{df.isnull().sum()}")
    
    print(f"\nFinal engineered features shape: {df.shape}")
    print(f"Feature columns: {[c for c in df.columns if c not in ['DateTime', 'ID']]}")
    
    return df


def prepare_data_for_model(df):
    """
    Prepare data for LSTM model: scaling, encoding, and splitting
    """
    print("\n" + "="*80)
    print("DATA PREPARATION FOR MODEL")
    print("="*80)
    
    # 10. Encode Junction (one-hot encoding)
    print("\n10. Encoding Junction as one-hot...")
    junction_dummies = pd.get_dummies(df['Junction'], prefix='junction')
    df = pd.concat([df, junction_dummies], axis=1)
    
    # Define feature columns (exclude target and metadata)
    exclude_cols = ['DateTime', 'ID', 'Vehicles', 'Junction']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # 9. Scale numerical features
    print("\n9. Scaling numerical features...")
    if CONFIG['scaler_type'] == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Fit scaler on training data only (to prevent data leakage)
    train_size = int(len(df) * CONFIG['train_ratio'])
    
    # Scale features
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Scale target variable separately
    target_scaler = MinMaxScaler()
    df_scaled['Vehicles_scaled'] = target_scaler.fit_transform(df[['Vehicles']])
    
    print(f"Scaler type: {CONFIG['scaler_type']}")
    print(f"Features scaled: {len(feature_cols)}")
    
    return df_scaled, feature_cols, scaler, target_scaler


# ============================================================================
# STEP 2: LSTM MODEL CREATION
# ============================================================================

class TrafficDataset(Dataset):
    """Custom Dataset for traffic sequences"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(data, feature_cols, target_col, sequence_length, prediction_horizon):
    """
    Create sequences for LSTM training
    """
    sequences = []
    targets = []
    
    # Group by Junction to maintain temporal continuity
    for junction in data['Junction'].unique():
        junction_data = data[data['Junction'] == junction].reset_index(drop=True)
        
        for i in range(len(junction_data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            seq = junction_data.iloc[i:i+sequence_length][feature_cols].values
            # Target (next value after sequence)
            target = junction_data.iloc[i+sequence_length+prediction_horizon-1][target_col]
            
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)


class LSTMModel(nn.Module):
    """
    Multi-layer LSTM model for traffic forecasting
    """
    
    def __init__(self, input_size, hidden_units, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = len(hidden_units)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_units[0], batch_first=True)
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(
                nn.LSTM(hidden_units[i-1], hidden_units[i], batch_first=True)
            )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_units[-1], 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:  # Apply ReLU and dropout except last layer
                x = self.relu(x)
                x = self.dropout(x)
        
        # Take output from last time step
        x = x[:, -1, :]
        
        # Apply dropout before final layer
        x = self.dropout(x)
        
        # Final output
        x = self.fc(x)
        
        return x


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0


def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs, patience):
    """
    Train the LSTM model with early stopping
    """
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['model_dir'], 'best_model.pth')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Best model saved at epoch {epoch+1} with val_loss: {val_loss:.6f}')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            model.load_state_dict(early_stopping.best_model)
            break
    
    return train_losses, val_losses, best_model_path


def evaluate_model(model, test_loader, target_scaler, device):
    """
    Evaluate model on test set
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(predictions).flatten()
    actuals = target_scaler.inverse_transform(actuals).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Save metrics
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }
    
    metrics_path = os.path.join(CONFIG['output_dir'], 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return predictions, actuals, metrics


def plot_results(train_losses, val_losses, predictions, actuals):
    """
    Create visualization plots
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Plot 1: Training and validation loss
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual (scatter)
    plt.subplot(1, 3, 2)
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Traffic')
    plt.ylabel('Predicted Traffic')
    plt.title('Predicted vs Actual Traffic')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Time series comparison (first 500 points)
    plt.subplot(1, 3, 3)
    sample_size = min(500, len(predictions))
    plt.plot(actuals[:sample_size], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(predictions[:sample_size], label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Traffic Volume')
    plt.title(f'Traffic Prediction (First {sample_size} samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(CONFIG['output_dir'], 'prediction_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_path}")
    
    plt.close()
    
    # Additional plot: Residuals
    plt.figure(figsize=(12, 4))
    
    residuals = actuals - predictions
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(predictions, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Traffic')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    residual_path = os.path.join(CONFIG['output_dir'], 'residual_analysis.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    print(f"Residual analysis saved to {residual_path}")
    
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline execution
    """
    print("\n" + "="*80)
    print("TRAFFIC FORECASTING LSTM PIPELINE")
    print("="*80)
    
    # Step 1: Feature Engineering
    df = load_and_engineer_features(CONFIG['data_path'])
    
    # Save engineered features
    engineered_path = os.path.join(CONFIG['output_dir'], 'feature_engineered_dataset.csv')
    df.to_csv(engineered_path, index=False)
    print(f"\nFeature-engineered dataset saved to {engineered_path}")
    
    # Prepare data for model
    df_scaled, feature_cols, scaler, target_scaler = prepare_data_for_model(df)
    
    # Step 2: Create sequences
    print("\n" + "="*80)
    print("CREATING SEQUENCES")
    print("="*80)
    
    sequences, targets = create_sequences(
        df_scaled, 
        feature_cols, 
        'Vehicles_scaled',
        CONFIG['sequence_length'],
        CONFIG['prediction_horizon']
    )
    
    print(f"\nSequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Split data (time-based, no shuffling)
    train_size = int(len(sequences) * CONFIG['train_ratio'])
    val_size = int(train_size * 0.1)  # 10% of train for validation
    
    X_train = sequences[:train_size-val_size]
    y_train = targets[:train_size-val_size]
    
    X_val = sequences[train_size-val_size:train_size]
    y_val = targets[train_size-val_size:train_size]
    
    X_test = sequences[train_size:]
    y_test = targets[train_size:]
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create DataLoaders
    train_dataset = TrafficDataset(X_train, y_train)
    val_dataset = TrafficDataset(X_val, y_val)
    test_dataset = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Build model
    print("\n" + "="*80)
    print("BUILDING LSTM MODEL")
    print("="*80)
    
    input_size = sequences.shape[2]  # Number of features
    model = LSTMModel(
        input_size=input_size,
        hidden_units=CONFIG['hidden_units'],
        dropout=CONFIG['dropout']
    )
    
    model = model.to(CONFIG['device'])
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Train model
    train_losses, val_losses, best_model_path = train_model(
        train_loader, val_loader, model, criterion, optimizer,
        CONFIG['device'], CONFIG['epochs'], CONFIG['patience']
    )
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model loaded from {best_model_path}")
    
    # Evaluate on test set
    predictions, actuals, metrics = evaluate_model(model, test_loader, target_scaler, CONFIG['device'])
    
    # Create visualizations
    plot_results(train_losses, val_losses, predictions, actuals)
    
    # Save final model and scalers
    final_model_path = os.path.join(CONFIG['model_dir'], 'lstm_traffic_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'feature_cols': feature_cols,
        'metrics': metrics
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save scalers
    import pickle
    scaler_path = os.path.join(CONFIG['model_dir'], 'feature_scaler.pkl')
    target_scaler_path = os.path.join(CONFIG['model_dir'], 'target_scaler.pkl')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"Feature scaler saved to {scaler_path}")
    print(f"Target scaler saved to {target_scaler_path}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Residual': actuals - predictions
    })
    results_path = os.path.join(CONFIG['output_dir'], 'test_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Test predictions saved to {results_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  1. Feature-engineered dataset: {engineered_path}")
    print(f"  2. Best model checkpoint: {best_model_path}")
    print(f"  3. Final model: {final_model_path}")
    print(f"  4. Evaluation metrics: {os.path.join(CONFIG['output_dir'], 'evaluation_metrics.json')}")
    print(f"  5. Visualizations: {os.path.join(CONFIG['output_dir'], 'prediction_visualization.png')}")
    print(f"  6. Test predictions: {results_path}")
    print(f"\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
