"""
Model Evaluation Script for SageMaker Processing Job (v2)
Evaluates trained XGBoost model on test set and outputs metrics
Uses centralized configuration from settings_v2
"""
import json
import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse
import subprocess
import sys

# Ensure pyarrow is installed for Parquet support
try:
    import pyarrow
except ImportError:
    print("Installing pyarrow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

# ===== Container-local constants (do NOT import project config inside the Processing container) =====
# These mirror the values in settings_v2 but must be self-contained inside
# the Processing job because only the evaluation script is uploaded to S3.
PROCESSING_MODEL_PATH = "/opt/ml/processing/model"
PROCESSING_TEST_PATH = "/opt/ml/processing/test"
PROCESSING_EVALUATION_PATH = "/opt/ml/processing/evaluation"
XGBOOST_MODEL_FILENAME = "xgboost-model"
PREDICTION_THRESHOLD = 0.5

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=PROCESSING_MODEL_PATH)
    parser.add_argument('--test-path', type=str, default=PROCESSING_TEST_PATH)
    parser.add_argument('--output-path', type=str, default=PROCESSING_EVALUATION_PATH)
    parser.add_argument('--input-content-type', type=str, default='application/x-parquet')
    return parser.parse_args()

def load_model(model_path):
    """Extract and load XGBoost model from tar.gz"""
    model_tar = os.path.join(model_path, 'model.tar.gz')
    
    # Extract model
    with tarfile.open(model_tar, 'r:gz') as tar:
        tar.extractall(path='/tmp/model')
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(f'/tmp/model/{XGBOOST_MODEL_FILENAME}')
    
    return model

def load_test_data(test_path, content_type='application/x-parquet'):
    """Load test data from Parquet or CSV files"""
    print(f"Loading data with content type: {content_type}")
    files = os.listdir(test_path)
    
    dfs = []
    
    if content_type == 'application/x-parquet':
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if not parquet_files:
             # Fallback if no specific extension but content type claims parquet
             # (SageMaker processing inputs might not preserve extension if not specified)
             parquet_files = [f for f in files if not f.startswith('.')]
        
        print(f"Found {len(parquet_files)} Parquet files")
        for f in parquet_files:
            path = os.path.join(test_path, f)
            dfs.append(pd.read_parquet(path))
            
    elif content_type == 'text/csv':
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            csv_files = [f for f in files if not f.startswith('.')]

        print(f"Found {len(csv_files)} CSV files")
        for f in csv_files:
            path = os.path.join(test_path, f)
            # Benchmark model CSVs have target first, no header
            dfs.append(pd.read_csv(path, header=None))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    if not dfs:
        raise ValueError(f"No valid data files found in {test_path}")

    test_df = pd.concat(dfs)
    
    # First column is target, rest are features
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and calculate metrics"""
    # Create DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Get predictions
    predictions_proba = model.predict(dtest)
    predictions = (predictions_proba > PREDICTION_THRESHOLD).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    return {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist()
    }

def save_evaluation_report(metrics, output_path):
    """Save evaluation metrics in SageMaker format"""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Format for SageMaker Pipeline condition check
    evaluation_output = {
        'metrics': {
            'f1_score': {
                'value': metrics['f1_score']
            },
            'precision': {
                'value': metrics['precision']
            },
            'recall': {
                'value': metrics['recall']
            }
        }
    }
    
    # Save evaluation.json for pipeline
    with open(os.path.join(output_path, 'evaluation.json'), 'w') as f:
        json.dump(evaluation_output, f, indent=2)
    
    # Save detailed metrics for human review
    with open(os.path.join(output_path, 'metrics_detailed.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def print_evaluation_results(metrics):
    """Print formatted evaluation results"""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"F1 Score:   {metrics['f1_score']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"{metrics['confusion_matrix']}")
    print("="*70 + "\n")

if __name__ == '__main__':
    args = parse_args()
    
    print("Starting model evaluation...")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Load test data
    print(f"Loading test data from {args.test_path}")
    X_test, y_test = load_test_data(args.test_path, args.input_content_type)
    print(f"Test set size: {len(y_test):,} samples")
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save results
    print(f"Saving evaluation results to {args.output_path}")
    save_evaluation_report(metrics, args.output_path)
    
    print("✅ Evaluation complete!")
