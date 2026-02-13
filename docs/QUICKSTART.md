# Traffic Forecasting LSTM - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python traffic_lstm_pipeline.py
```

This will:
- ✅ Load data from `raw-data/traffic.csv`
- ✅ Engineer 30+ features
- ✅ Train LSTM model with early stopping
- ✅ Evaluate on test set
- ✅ Generate visualizations
- ✅ Save model and results

**Expected runtime**: 5-15 minutes (CPU), 2-5 minutes (GPU)

### 3. Make Predictions
```bash
python inference.py
```

This will:
- ✅ Load trained model
- ✅ Predict next hour traffic
- ✅ Generate 24-hour forecast
- ✅ Predict for all junctions

---

## 📁 Project Structure

```
Traffic-Prediction/
├── raw-data/
│   └── traffic.csv                          # Input dataset
│
├── outputs/                                  # Generated outputs
│   ├── feature_engineered_dataset.csv       # Engineered features
│   ├── evaluation_metrics.json              # MSE, RMSE, MAE, R²
│   ├── prediction_visualization.png         # Training curves + predictions
│   ├── residual_analysis.png                # Error analysis
│   ├── test_predictions.csv                 # Test set results
│   └── future_predictions.csv               # Future forecasts
│
├── models/                                   # Saved models
│   ├── best_model.pth                       # Best checkpoint
│   ├── lstm_traffic_model_final.pth         # Final model
│   ├── feature_scaler.pkl                   # Feature scaler
│   └── target_scaler.pkl                    # Target scaler
│
├── traffic_lstm_pipeline.py                 # Main training script
├── inference.py                              # Prediction script
├── requirements.txt                          # Dependencies
└── README_LSTM.md                            # Full documentation
```

---

## 🎯 Features Implemented

### ✅ Feature Engineering (30+ features)
- Temporal: hour, day_of_week, month, day_of_month, week_of_year, quarter
- Lag features: lag_1, lag_2, lag_3
- Rolling statistics: 3h/6h mean and std
- Junction statistics: mean, std, ratio
- Cyclical encoding: hour_sin/cos, day_sin/cos
- Flags: is_weekend, is_rush_hour, is_night, is_peak_day, is_holiday
- One-hot: Junction encoding

### ✅ LSTM Model
- Multi-layer LSTM (2-3 layers, configurable)
- Hidden units: [128, 64] (default)
- Dropout: 0.2 (configurable)
- Early stopping (patience=10)
- Model checkpointing
- GPU support

### ✅ Training & Evaluation
- Time-based train/test split (80/20)
- Batch training (size=64)
- Adam optimizer
- MSE loss
- Metrics: MSE, RMSE, MAE, R²
- Comprehensive visualizations

### ✅ Outputs
- Feature-engineered CSV
- Trained model checkpoints
- Scalers for inference
- Evaluation metrics JSON
- Prediction plots
- Test results CSV

---

## 📊 Expected Results

Based on typical traffic datasets:
- **RMSE**: 2-5 vehicles
- **MAE**: 1.5-4 vehicles
- **R²**: 0.75-0.95

Results vary based on:
- Data quality and size
- Junction complexity
- Seasonal patterns
- Model hyperparameters

---

## 🔧 Configuration

Edit `CONFIG` in `traffic_lstm_pipeline.py`:

```python
CONFIG = {
    'sequence_length': 24,      # Hours of history to use
    'hidden_units': [128, 64],  # LSTM layer sizes
    'dropout': 0.2,             # Dropout rate
    'batch_size': 64,           # Training batch size
    'epochs': 150,              # Max epochs
    'learning_rate': 0.001,     # Learning rate
    'patience': 10,             # Early stopping patience
}
```

---

## 💡 Tips for Best Results

1. **More Training Data**: Use at least 3-6 months of hourly data
2. **Sequence Length**: Try 24 (1 day), 168 (1 week), or 720 (1 month)
3. **Model Size**: Increase hidden_units for complex patterns
4. **Regularization**: Adjust dropout if overfitting/underfitting
5. **Feature Selection**: Remove low-importance features if needed

---

## 🐛 Troubleshooting

### Out of Memory
```python
CONFIG['batch_size'] = 32      # Reduce batch size
CONFIG['sequence_length'] = 12  # Reduce sequence length
```

### Poor Predictions
- Check data quality (missing values, outliers)
- Increase model complexity: `hidden_units: [256, 128, 64]`
- Try longer sequences: `sequence_length: 48`
- Adjust learning rate: `learning_rate: 0.0001`

### Overfitting
- Increase dropout: `dropout: 0.3`
- Reduce model size: `hidden_units: [64, 32]`
- Add more training data

### Slow Training
- Use GPU (CUDA)
- Reduce batch_size or epochs
- Simplify model architecture

---

## 📈 Sample Output

```
STEP 1: FEATURE ENGINEERING
================================================================================
Loading data from ./raw-data/traffic.csv...
Initial shape: (48120, 4)
Features engineered: 35

MODEL TRAINING
================================================================================
Epoch [10/150] - Train Loss: 0.012345, Val Loss: 0.013456
Best model saved at epoch 47 with val_loss: 0.009876
Early stopping triggered at epoch 57

MODEL EVALUATION
================================================================================
Test Set Metrics:
  MSE:  3.2145
  RMSE: 1.7929
  MAE:  1.3421
  R²:   0.8742

PIPELINE COMPLETED SUCCESSFULLY!
```

---

## 🔮 Making Predictions

### Single Prediction
```python
from inference import TrafficPredictor

predictor = TrafficPredictor(model_dir='./models')
prediction = predictor.predict(df, junction_id=1)
print(f"Next hour traffic: {prediction:.2f}")
```

### Multi-Step Forecast
```python
predictions = predictor.predict_multiple_steps(df, junction_id=1, n_steps=24)
# Returns 24 hourly predictions
```

---

## 📚 Full Documentation

See `README_LSTM.md` for:
- Detailed feature descriptions
- Model architecture explanation
- Advanced configuration
- Custom feature engineering
- Hyperparameter tuning guide

---

## ✨ Key Highlights

✅ **Complete Pipeline**: Raw data → Trained model → Predictions
✅ **Production-Ready**: Scalers, checkpoints, inference script
✅ **GPU Accelerated**: Automatic CUDA detection
✅ **Reproducible**: Fixed random seeds
✅ **Extensible**: Easy to add features or modify architecture
✅ **Well-Documented**: Comments, README, examples

---

## 📞 Support

For issues or questions:
1. Check `README_LSTM.md` for detailed documentation
2. Review configuration options
3. Verify data format matches expected schema
4. Check console output for error messages

---

**Ready to forecast traffic? Run the pipeline and let the LSTM learn! 🚗📊**
