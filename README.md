# Traffic Prediction Project - Updated Structure

## Project Structure

```
Traffic-Prediction/
├── src/                          # Core source code
│   ├── traffic_lstm_pipeline.py  # Main training pipeline
│   └── inference.py              # Inference script for predictions
├── scripts/                      # Utility and validation scripts
│   ├── compare_predictions.py    # Compare prediction results
│   ├── run_test_evaluation.py    # Run model evaluation on test set
│   ├── verify_best_model.py      # Verify which model performs best
│   └── verify_implementation.py  # Verify PRD compliance
├── docs/                         # Documentation
│   ├── PRD.txt                   # Product Requirements Document
│   ├── PRD-FE.txt                # Feature Engineering PRD
│   ├── PROJECT_OVERVIEW.txt      # Project overview
│   ├── IMPLEMENTATION_SUMMARY.txt
│   ├── PRD_COMPLIANCE_VERIFICATION.txt
│   ├── QUICKSTART.md             # Quick start guide
│   └── README_LSTM.md            # LSTM implementation details
├── models/                       # Saved models and scalers
├── outputs/                      # Output files and predictions
├── raw-data/                     # Raw dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How to Run

### Training the Model
From the project root directory:
```bash
cd src
python traffic_lstm_pipeline.py
```

### Making Predictions
From the project root directory:
```bash
cd src
python inference.py
```

### Running Validation Scripts
From the project root directory:
```bash
cd scripts
python verify_implementation.py    # Verify PRD compliance
python run_test_evaluation.py      # Evaluate on test set
python compare_predictions.py      # Compare prediction files
python verify_best_model.py        # Find best model
```

## Important Notes

1. **Working Directory**: All scripts now expect to be run from their respective directories (src/ or scripts/)
2. **Path Updates**: All file paths have been updated to use relative paths (`../` notation)
3. **No Code Changes**: Only file locations and import paths were changed - the functionality remains identical

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   cd src
   python traffic_lstm_pipeline.py
   ```

3. Verify implementation:
   ```bash
   cd scripts
   python verify_implementation.py
   ```

For more details, see `docs/QUICKSTART.md`
