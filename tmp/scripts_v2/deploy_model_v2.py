"""
Deploy Registered Model to Batch Transform (v2)
Runs nightly predictions on next-day flights
Uses centralized configuration from settings_v2
"""
import boto3
import sagemaker

from sagemaker.model import ModelPackage
from datetime import datetime

import argparse
import sys
import settings_v2 as cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Model to Batch Transform")
    parser.add_argument("--input-data", type=str, 
                        default=f's3://{cfg.BUCKET}/{cfg.PREFIX}/data/inference/test_inference_small.csv',
                        help="S3 URI for input data")
    parser.add_argument("--content-type", type=str, default="text/csv",
                        help="MIME type of input data (e.g., text/csv, application/x-parquet)")
    parser.add_argument("--instance-type", type=str, default=cfg.TRANSFORM_INSTANCE_TYPE,
                        help="Instance type for transformer")
    parser.add_argument("--instance-count", type=int, default=cfg.TRANSFORM_INSTANCE_COUNT,
                        help="Instance count")
    return parser.parse_args()

# ===========================
# GET LATEST APPROVED MODEL
# ===========================
if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*70)
    print("DEPLOYING MODEL TO BATCH TRANSFORM")
    print("="*70)

    sm_client = boto3.client('sagemaker', region_name=cfg.REGION)

    print("\n🔍 Finding latest approved model...")

    # List models in registry
    response = sm_client.list_model_packages(
        ModelPackageGroupName=cfg.MODEL_PACKAGE_GROUP,
        ModelApprovalStatus='Approved',  # Only approved models
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )

    if not response['ModelPackageSummaryList']:
        print("\n❌ No approved models found!")
        print("   Please approve a model first:")
        print("   1. Go to SageMaker Studio → Model Registry")
        print(f"   2. Find '{cfg.MODEL_PACKAGE_GROUP}'")
        print("   3. Select a model version")
        print("   4. Click 'Update status' → 'Approve'")
        exit(1)

    latest_model = response['ModelPackageSummaryList'][0]
    model_package_arn = latest_model['ModelPackageArn']

    print(f"✅ Found approved model:")
    print(f"   ARN: {model_package_arn}")
    print(f"   Status: {latest_model['ModelApprovalStatus']}")
    print(f"   Created: {latest_model['CreationTime']}")

    # ===========================
    # CREATE MODEL PACKAGE
    # ===========================
    print("\n📦 Creating SageMaker Model from registered package...")

    role = cfg.ROLE
    session = cfg.sagemaker_session

    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=session
    )

    print("✅ Model package created")

    # ===========================
    # CREATE BATCH TRANSFORMER
    # ===========================
    print("\n🔧 Creating Batch Transformer...")

    transformer = model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=cfg.get_s3_path('predictions'),
        assemble_with='Line',
        accept='text/csv',
        strategy='SingleRecord'  # Process one record at a time
    )

    print("✅ Transformer created")
    print(f"   Instance: {cfg.TRANSFORM_INSTANCE_TYPE}")
    print(f"   Output: {cfg.get_s3_path('predictions')}")

    # ===========================
    # RUN BATCH PREDICTION (DEMO)
    # ===========================
    print("\n" + "="*70)
    print("RUNNING BATCH PREDICTION (DEMO)")
    print("="*70)

    # Run transform
    job_name = f'flight-delay-predictions-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    transformer.transform(
        data=args.input_data,
        content_type=args.content_type,
        split_type='Line',
        job_name=job_name,
        wait=True,  # Wait for completion
        logs=True   # Show progress
    )

    print("\n✅ Batch transform complete!")

    # ===========================
    # CHECK OUTPUT
    # ===========================
    print("\n📊 Checking predictions...")

    output_path = transformer.output_path

    print(f"\n✅ Predictions saved to:")
    print(f"   {output_path}")

    # List output files
    s3 = boto3.client('s3', region_name=cfg.REGION)
    response = s3.list_objects_v2(
        Bucket=cfg.BUCKET,
        Prefix=f'{cfg.PREFIX}/predictions/{job_name}/'
    )

    if 'Contents' in response:
        print(f"\n📄 Output files:")
        for obj in response['Contents']:
            size_mb = obj['Size'] / (1024 * 1024)
            print(f"   {obj['Key']} ({size_mb:.2f} MB)")

    # ===========================
    # DOWNLOAD SAMPLE PREDICTIONS
    # ===========================
    print("\n📥 Downloading sample predictions...")

    # Find output file
    output_file = None
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.out'):
            output_file = obj['Key']
            break

    if output_file:
        local_file = 'sample_predictions.csv'
        s3.download_file(cfg.BUCKET, output_file, local_file)
        
        # Show first 10 predictions
        import pandas as pd
        predictions = pd.read_csv(local_file, header=None, nrows=10)
        
        print(f"\n✅ Sample predictions (first 10):")
        print(predictions)
        print(f"\nPredictions: 0 = on-time, 1 = delayed")
        
        # Summary
        all_preds = pd.read_csv(local_file, header=None)
        print(f"\n📊 Prediction Summary:")
        print(f"   Total predictions: {len(all_preds):,}")
        print(f"   Predicted on-time (0): {(all_preds[0] < 0.5).sum():,} ({(all_preds[0] < 0.5).sum()/len(all_preds)*100:.1f}%)")
        print(f"   Predicted delayed (1): {(all_preds[0] >= 0.5).sum():,} ({(all_preds[0] >= 0.5).sum()/len(all_preds)*100:.1f}%)")

    print("\n" + "="*70)
    print("🎉 DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"\n✅ Model deployed and tested")
    print(f"✅ Batch predictions working")
