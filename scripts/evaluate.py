"""
Model Evaluation Script for SageMaker Processing
Evaluates trained model on test set and generates metrics
Handles both CSV and Parquet input formats
"""
import argparse
import json
import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-path', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    
    # Input format
    parser.add_argument('--input-content-type', type=str, default='text/csv',
                       help='Content type of input data (text/csv or application/x-parquet)')
    
    return parser.parse_args()

def load_model(model_path):
    """Load XGBoost model from tar.gz file"""
    print(f"\n--- Loading Model ---")
    print(f"Model path: {model_path}")
    
    # Find model.tar.gz
    model_tar = os.path.join(model_path, 'model.tar.gz')
    
    if not os.path.exists(model_tar):
        raise FileNotFoundError(f"Model file not found: {model_tar}")
    
    # Extract model
    with tarfile.open(model_tar, 'r:gz') as tar:
        tar.extractall(path=model_path)
    
    # Load XGBoost model
    model_file = os.path.join(model_path, 'xgboost-model')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"XGBoost model file not found: {model_file}")
    
    model = xgb.Booster()
    model.load_model(model_file)
    
    print(f"✓ Model loaded successfully")
    return model

def load_test_data(test_path, content_type):
    """Load test data from CSV or Parquet"""
    print(f"\n--- Loading Test Data ---")
    print(f"Test path: {test_path}")
    print(f"Content type: {content_type}")
    
    # List files in test directory
    test_files = []
    for root, dirs, files in os.walk(test_path):
        for file in files:
            if content_type == 'text/csv' and file.endswith('.csv'):
                test_files.append(os.path.join(root, file))
            elif content_type == 'application/x-parquet' and file.endswith('.parquet'):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test file(s)")
    
    if not test_files:
        raise FileNotFoundError(f"No test files found in {test_path}")
    
    # Load data based on content type
    dfs = []
    for path in test_files:
        print(f"  Loading: {path}")
        if content_type == 'text/csv':
            # CSV format: no headers, target in first column
            df = pd.read_csv(path, header=None)
        else:
            # Parquet format
            df = pd.read_parquet(path)
        
        dfs.append(df)
    
    # Combine all files
    data = pd.concat(dfs, ignore_index=True)
    
    print(f"✓ Test data loaded: {data.shape}")
    
    # Split into features and target
    # First column is target (DELAYED), rest are features
    y_test = data.iloc[:, 0].values
    X_test = data.iloc[:, 1:].values
    
    print(f"  Features: {X_test.shape}")
    print(f"  Target: {y_test.shape}")
    print(f"  Target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute metrics"""
    print(f"\n--- Evaluating Model ---")
    
    # Create DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Get predictions (probabilities)
    y_pred_proba = model.predict(dtest)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                0        1")
    print(f"Actual  0    {tn:>6,}  {fp:>6,}")
    print(f"        1    {fn:>6,}  {tp:>6,}")
    
    # Prepare metrics dictionary
    metrics = {
        'metrics': {
            'accuracy': {'value': float(accuracy)},
            'precision': {'value': float(precision)},
            'recall': {'value': float(recall)},
            'f1_score': {'value': float(f1)},
            'auc': {'value': float(auc)}
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }
    
    return metrics

def save_evaluation_report(metrics, output_path):
    """Save evaluation metrics as JSON"""
    print(f"\n--- Saving Evaluation Report ---")
    
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'evaluation.json')
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Evaluation report saved: {output_file}")
    
    # Print summary
    print(f"\n📄 Evaluation Summary:")
    print(f"  F1 Score:  {metrics['metrics']['f1_score']['value']:.4f}")
    print(f"  Precision: {metrics['metrics']['precision']['value']:.4f}")
    print(f"  Recall:    {metrics['metrics']['recall']['value']:.4f}")
    print(f"  AUC:       {metrics['metrics']['auc']['value']:.4f}")

def main():
    """Main execution"""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    args = parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load test data
    X_test, y_test = load_test_data(args.test_path, args.input_content_type)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_evaluation_report(metrics, args.output_path)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()