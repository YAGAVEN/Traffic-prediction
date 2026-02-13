"""
Inference Script for Traffic Forecasting LSTM Model
Use this script to make predictions with a trained model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from datetime import datetime, timedelta

class LSTMModel(nn.Module):
    """LSTM model architecture (must match training)"""
    
    def __init__(self, input_size, hidden_units, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = len(hidden_units)
        
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_units[0], batch_first=True))
        
        for i in range(1, len(hidden_units)):
            self.lstm_layers.append(nn.LSTM(hidden_units[i-1], hidden_units[i], batch_first=True))
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units[-1], 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def engineer_features_for_inference(df):
    """
    Apply the same feature engineering as training
    """
    # Sort by Junction and DateTime
    df = df.sort_values(['Junction', 'DateTime']).reset_index(drop=True)
    
    # Extract date-time features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['DateTime'].dt.month
    df['day_of_month'] = df['DateTime'].dt.day
    df['week_of_year'] = df['DateTime'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['DateTime'].dt.quarter
    df['is_holiday'] = 0
    
    # Lag features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('Junction')['Vehicles'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6]:
        df[f'rolling_mean_{window}h'] = df.groupby('Junction')['Vehicles'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}h'] = df.groupby('Junction')['Vehicles'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Junction-specific features
    junction_stats = df.groupby('Junction')['Vehicles'].agg(['mean', 'std']).reset_index()
    junction_stats.columns = ['Junction', 'junction_mean', 'junction_std']
    df = df.merge(junction_stats, on='Junction', how='left')
    df['vehicles_ratio_to_junction_mean'] = df['Vehicles'] / (df['junction_mean'] + 1e-8)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
    df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)
    df['day_sin'] = np.sin(df['day_of_week'] * 2 * np.pi / 7)
    df['day_cos'] = np.cos(df['day_of_week'] * 2 * np.pi / 7)
    
    # Optional flags
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                          (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_peak_day'] = (df['day_of_week'] < 5).astype(int)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # One-hot encode Junction
    junction_dummies = pd.get_dummies(df['Junction'], prefix='junction')
    df = pd.concat([df, junction_dummies], axis=1)
    
    return df


class TrafficPredictor:
    """Class for making traffic predictions"""
    
    def __init__(self, model_dir='../models'):
        self.model_dir = model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model checkpoint
        model_path = os.path.join(model_dir, 'lstm_traffic_model_final.pth')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.feature_cols = checkpoint['feature_cols']
        
        # Initialize model
        input_size = len(self.feature_cols)
        self.model = LSTMModel(
            input_size=input_size,
            hidden_units=self.config['hidden_units'],
            dropout=self.config['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scalers
        with open(os.path.join(model_dir, 'feature_scaler.pkl'), 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        with open(os.path.join(model_dir, 'target_scaler.pkl'), 'rb') as f:
            self.target_scaler = pickle.load(f)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {self.device}")
        print(f"Sequence length: {self.config['sequence_length']}")
    
    def prepare_sequence(self, df, junction_id):
        """Prepare input sequence for prediction"""
        
        # Filter by junction
        junction_data = df[df['Junction'] == junction_id].copy()
        
        if len(junction_data) < self.config['sequence_length']:
            raise ValueError(f"Need at least {self.config['sequence_length']} hours of data")
        
        # Get last sequence_length rows
        recent_data = junction_data.tail(self.config['sequence_length'])
        
        # Scale features
        sequence = recent_data[self.feature_cols].values
        sequence_scaled = self.feature_scaler.transform(sequence)
        
        return sequence_scaled
    
    def predict(self, df, junction_id):
        """
        Make prediction for next time step
        
        Args:
            df: DataFrame with historical data (already engineered)
            junction_id: Junction ID to predict for
            
        Returns:
            Predicted traffic volume
        """
        # Prepare sequence
        sequence = self.prepare_sequence(df, junction_id)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction_scaled = self.model(sequence_tensor)
            prediction = self.target_scaler.inverse_transform(
                prediction_scaled.cpu().numpy()
            )
        
        return float(prediction[0, 0])
    
    def predict_multiple_steps(self, df, junction_id, n_steps=24):
        """
        Make multi-step predictions (iterative)
        
        Args:
            df: DataFrame with historical data
            junction_id: Junction ID to predict for
            n_steps: Number of future steps to predict
            
        Returns:
            List of predictions
        """
        predictions = []
        current_df = df.copy()
        
        for step in range(n_steps):
            # Make prediction
            pred = self.predict(current_df, junction_id)
            predictions.append(pred)
            
            # Create next timestamp
            last_time = current_df[current_df['Junction'] == junction_id]['DateTime'].iloc[-1]
            next_time = last_time + timedelta(hours=1)
            
            # Create new row with prediction
            new_row = current_df[current_df['Junction'] == junction_id].iloc[-1:].copy()
            new_row['DateTime'] = next_time
            new_row['Vehicles'] = pred
            
            # Append and re-engineer features
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            current_df = engineer_features_for_inference(current_df)
        
        return predictions


def main():
    """Example usage"""
    
    print("="*80)
    print("TRAFFIC FORECASTING - INFERENCE")
    print("="*80)
    
    # Load historical data
    print("\nLoading historical data...")
    df = pd.read_csv('../raw-data/traffic.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Engineer features
    print("Engineering features...")
    df = engineer_features_for_inference(df)
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = TrafficPredictor(model_dir='../models')
    
    # Example 1: Single prediction
    print("\n" + "-"*80)
    print("Example 1: Single Step Prediction")
    print("-"*80)
    
    junction_id = 1
    prediction = predictor.predict(df, junction_id)
    
    last_time = df[df['Junction'] == junction_id]['DateTime'].iloc[-1]
    next_time = last_time + timedelta(hours=1)
    
    print(f"\nJunction: {junction_id}")
    print(f"Last timestamp: {last_time}")
    print(f"Predicted for: {next_time}")
    print(f"Predicted traffic: {prediction:.2f} vehicles")
    
    # Example 2: Multi-step prediction
    print("\n" + "-"*80)
    print("Example 2: Multi-Step Prediction (24 hours)")
    print("-"*80)
    
    n_steps = 24
    predictions = predictor.predict_multiple_steps(df, junction_id, n_steps)
    
    print(f"\nPredictions for next {n_steps} hours:")
    for i, pred in enumerate(predictions[:10], 1):  # Show first 10
        future_time = last_time + timedelta(hours=i)
        print(f"  {future_time}: {pred:.2f} vehicles")
    
    if n_steps > 10:
        print(f"  ... ({n_steps - 10} more predictions)")
    
    # Save predictions
    results = []
    for i, pred in enumerate(predictions, 1):
        future_time = last_time + timedelta(hours=i)
        results.append({
            'DateTime': future_time,
            'Junction': junction_id,
            'Predicted_Vehicles': pred
        })
    
    results_df = pd.DataFrame(results)
    output_path = '../outputs/future_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Example 3: Predict for all junctions
    print("\n" + "-"*80)
    print("Example 3: Predictions for All Junctions")
    print("-"*80)
    
    all_junctions = df['Junction'].unique()
    print(f"\nPredicting for {len(all_junctions)} junctions...")
    
    all_predictions = {}
    for junc in all_junctions:
        try:
            pred = predictor.predict(df, junc)
            all_predictions[junc] = pred
            print(f"  Junction {junc}: {pred:.2f} vehicles")
        except Exception as e:
            print(f"  Junction {junc}: Error - {str(e)}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
