"""
Script to reorganize project structure
"""
import os
import shutil

# Create folder structure
folders = ['src', 'scripts', 'docs']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Define file movements
moves = {
    'docs': [
        'IMPLEMENTATION_SUMMARY.txt',
        'PRD-FE.txt',
        'PRD.txt',
        'PRD_COMPLIANCE_VERIFICATION.txt',
        'PROJECT_OVERVIEW.txt',
        'QUICKSTART.md',
        'README_LSTM.md'
    ],
    'src': [
        'traffic_lstm_pipeline.py',
        'inference.py'
    ],
    'scripts': [
        'compare_predictions.py',
        'run_test_evaluation.py',
        'verify_best_model.py',
        'verify_implementation.py'
    ]
}

# Move files
for dest_folder, files in moves.items():
    for file in files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(dest_folder, file))
            print(f"Moved {file} -> {dest_folder}/")
        else:
            print(f"File not found: {file}")

print("\nReorganization complete!")
