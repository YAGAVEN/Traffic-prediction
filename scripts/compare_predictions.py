"""
Compare all test prediction files and determine the best model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import glob

print("="*80)
print("COMPARING ALL TEST PREDICTION FILES")
print("="*80)

# Find all test prediction CSV files
prediction_files = glob.glob('../outputs/test_predictions*.csv')
prediction_files.sort()

print(f"\nFound {len(prediction_files)} prediction files:")
for f in prediction_files:
    print(f"  - {os.path.basename(f)}")

# Store results
results = []

for pred_file in prediction_files:
    filename = os.path.basename(pred_file)
    print(f"\n{'-'*80}")
    print(f"Analyzing: {filename}")
    print(f"{'-'*80}")
    
    try:
        # Load predictions
        df = pd.read_csv(pred_file)
        
        # Check required columns
        if 'Actual' not in df.columns or 'Predicted' not in df.columns:
            print(f"  ⚠️  Missing required columns. Skipping.")
            continue
        
        # Calculate metrics
        actual = df['Actual'].values
        predicted = df['Predicted'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            print(f"  ⚠️  No valid data. Skipping.")
            continue
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Additional metrics
        residuals = actual - predicted
        max_error = np.max(np.abs(residuals))
        median_ae = np.median(np.abs(residuals))
        
        # Count negative predictions (unrealistic for vehicle counts)
        negative_predictions = np.sum(predicted < 0)
        
        # Count extreme predictions (more than 200 vehicles - seems unrealistic based on data)
        extreme_predictions = np.sum(predicted > 200)
        
        print(f"  Samples: {len(actual)}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Max Error: {max_error:.2f}")
        print(f"  Median Absolute Error: {median_ae:.2f}")
        print(f"  Negative Predictions: {negative_predictions}")
        print(f"  Extreme Predictions (>200): {extreme_predictions}")
        
        # Store results
        results.append({
            'filename': filename,
            'samples': len(actual),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'max_error': max_error,
            'median_ae': median_ae,
            'negative_preds': negative_predictions,
            'extreme_preds': extreme_predictions,
            'actual_mean': actual.mean(),
            'actual_std': actual.std(),
            'pred_mean': predicted.mean(),
            'pred_std': predicted.std()
        })
        
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        continue

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

if len(results) == 0:
    print("No valid prediction files found!")
else:
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Sort by R² (descending) and MAE (ascending)
    comparison_df['score'] = comparison_df['r2'] - (comparison_df['mae'] / 100)
    comparison_df = comparison_df.sort_values('score', ascending=False)
    
    # Display comparison table
    print("\nRanked by Performance (R² score - normalized MAE):\n")
    
    for idx, row in comparison_df.iterrows():
        rank = list(comparison_df.index).index(idx) + 1
        print(f"{'='*80}")
        print(f"Rank {rank}: {row['filename']}")
        print(f"{'='*80}")
        print(f"  Samples:           {int(row['samples'])}")
        print(f"  R² Score:          {row['r2']:.4f} {'✓ BEST' if rank == 1 else ''}")
        print(f"  RMSE:              {row['rmse']:.4f}")
        print(f"  MAE:               {row['mae']:.4f}")
        print(f"  Max Error:         {row['max_error']:.2f}")
        print(f"  Median Abs Error:  {row['median_ae']:.2f}")
        print(f"  Negative Preds:    {int(row['negative_preds'])} {'⚠️' if row['negative_preds'] > 0 else '✓'}")
        print(f"  Extreme Preds:     {int(row['extreme_preds'])} {'⚠️' if row['extreme_preds'] > 100 else '✓'}")
        print(f"  Actual Mean/Std:   {row['actual_mean']:.2f} ± {row['actual_std']:.2f}")
        print(f"  Pred Mean/Std:     {row['pred_mean']:.2f} ± {row['pred_std']:.2f}")
        print()
    
    # Recommendations
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    best_model = comparison_df.iloc[0]
    
    print(f"\n🏆 BEST MODEL: {best_model['filename']}")
    print(f"\nKey Metrics:")
    print(f"  • R² Score:    {best_model['r2']:.4f} (explains {best_model['r2']*100:.1f}% of variance)")
    print(f"  • RMSE:        {best_model['rmse']:.4f} vehicles")
    print(f"  • MAE:         {best_model['mae']:.4f} vehicles (avg error)")
    print(f"  • Test Samples: {int(best_model['samples'])}")
    
    # Quality indicators
    print(f"\nQuality Indicators:")
    
    if best_model['r2'] > 0.5:
        print(f"  ✓ Good R² score (>0.5)")
    elif best_model['r2'] > 0.3:
        print(f"  ⚠️  Moderate R² score (0.3-0.5)")
    else:
        print(f"  ❌ Poor R² score (<0.3)")
    
    if best_model['mae'] < 5:
        print(f"  ✓ Excellent MAE (<5 vehicles)")
    elif best_model['mae'] < 10:
        print(f"  ✓ Good MAE (<10 vehicles)")
    else:
        print(f"  ⚠️  High MAE (>{10} vehicles)")
    
    if best_model['negative_preds'] == 0:
        print(f"  ✓ No negative predictions (physically realistic)")
    else:
        print(f"  ⚠️  {int(best_model['negative_preds'])} negative predictions (unrealistic)")
    
    # Save comparison results
    output_file = '../outputs/model_comparison_results.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\n📊 Comparison results saved to: {output_file}")
    
    # Save best model info
    best_model_info = {
        'best_model_file': best_model['filename'],
        'metrics': {
            'r2': float(best_model['r2']),
            'rmse': float(best_model['rmse']),
            'mae': float(best_model['mae']),
            'mse': float(best_model['mse'])
        },
        'samples': int(best_model['samples']),
        'quality_flags': {
            'has_negative_predictions': int(best_model['negative_preds']) > 0,
            'has_extreme_predictions': int(best_model['extreme_preds']) > 100
        }
    }
    
    with open('../outputs/best_model_selection.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"📋 Best model info saved to: ../outputs/best_model_selection.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
