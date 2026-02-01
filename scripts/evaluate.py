"""
Model Evaluation Script for SageMaker Processing Job
Evaluates trained XGBoost model on test set and outputs metrics
"""
import json
import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-path', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    return parser.parse_args()

def load_model(model_path):
    """Extract and load XGBoost model from tar.gz"""
    model_tar = os.path.join(model_path, 'model.tar.gz')
    
    # Extract model
    with tarfile.open(model_tar, 'r:gz') as tar:
        tar.extractall(path='/tmp/model')
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model('/tmp/model/xgboost-model')
    
    return model

def load_test_data(test_path):
    """Load test data from CSV files"""
    test_files = [
        os.path.join(test_path, f) 
        for f in os.listdir(test_path) 
        if f.endswith('.csv')
    ]
    
    test_df = pd.concat([pd.read_csv(f, header=None) for f in test_files])
    
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
    predictions = (predictions_proba > 0.5).astype(int)
    
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
    X_test, y_test = load_test_data(args.test_path)
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