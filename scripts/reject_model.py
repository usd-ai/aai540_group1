"""
Model Rejection Script
Runs when model fails to meet performance threshold
Logs rejection reason and saves report
"""
import argparse
import json
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--actual-f1', type=float, required=True)
    parser.add_argument('--precision', type=float, default=0.0)
    parser.add_argument('--recall', type=float, default=0.0)
    parser.add_argument('--auc', type=float, default=0.0)
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/rejection')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("MODEL REJECTION")
    print("=" * 70)
    
    # Display actual metrics
    print(f"\n📊 Model Performance:")
    print(f"   F1 Score:  {args.actual_f1:.4f}")
    print(f"   Precision: {args.precision:.4f}")
    print(f"   Recall:    {args.recall:.4f}")
    print(f"   AUC:       {args.auc:.4f}")
    
    print(f"\n❌ REJECTION REASON:")
    print(f"   F1 Score ({args.actual_f1:.4f}) < Threshold ({args.threshold:.2f})")
    print(f"   Performance gap: {args.threshold - args.actual_f1:.4f}")
    print(f"   Model does not meet minimum quality standards")
    
    # Diagnose the issue
    print(f"\n🔍 Diagnosis:")
    if args.actual_f1 < 0.10:
        print("   • Extremely poor performance - model may be predicting all one class")
        print("   • Check class balancing (scale_pos_weight parameter)")
    elif args.precision < 0.15:
        print("   • Low precision - too many false positives")
        print("   • Consider increasing decision threshold or improving features")
    elif args.recall < 0.20:
        print("   • Low recall - missing too many delays")
        print("   • Model is too conservative, consider class balancing")
    else:
        print("   • Model performance is close but not quite meeting threshold")
        print("   • Fine-tuning hyperparameters may help")
    
    # Create rejection report
    rejection_report = {
        'rejection_timestamp': datetime.utcnow().isoformat(),
        'rejection_reason': 'Model performance below threshold',
        'threshold': {
            'metric': 'f1_score',
            'required': args.threshold,
            'actual': args.actual_f1,
            'gap': args.threshold - args.actual_f1
        },
        'model_metrics': {
            'f1_score': args.actual_f1,
            'precision': args.precision,
            'recall': args.recall,
            'auc': args.auc
        },
        'status': 'REJECTED',
        'recommendations': []
    }
    
    # Add specific recommendations based on metrics
    if args.actual_f1 < 0.10:
        rejection_report['recommendations'].extend([
            'Add class balancing: Set scale_pos_weight to handle class imbalance',
            'Verify target variable is correct',
            'Check if model is predicting all one class'
        ])
    
    if args.precision < 0.15 or args.recall < 0.20:
        rejection_report['recommendations'].extend([
            'Review feature engineering - add domain-specific features',
            'Add target-encoded features (historical delay rates)',
            'Add volume-based features (flight frequency metrics)'
        ])
    
    rejection_report['recommendations'].extend([
        'Increase model complexity (deeper trees, more rounds)',
        'Tune learning rate for better convergence',
        'Collect more training data if available'
    ])
    
    # Save rejection report
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, 'rejection_report.json')
    
    with open(output_file, 'w') as f:
        json.dump(rejection_report, f, indent=2)
    
    print(f"\n📄 Rejection report saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("MODEL REJECTED - NOT REGISTERED")
    print("=" * 70)
    
    print("\n💡 Recommendations:")
    for rec in rejection_report['recommendations']:
        print(f"   • {rec}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
