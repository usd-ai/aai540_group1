"""
Deploy Registered Model to Batch Transform
Runs nightly predictions on next-day flights
"""
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import ModelPackage
from datetime import datetime

# ===========================
# CONFIGURATION
# ===========================
BUCKET = 'sagemaker-us-east-1-425709451100'
PREFIX = 'aai540-group1'
MODEL_PACKAGE_GROUP = 'flight-delay-models'

# ===========================
# GET LATEST APPROVED MODEL
# ===========================
print("\n" + "="*70)
print("DEPLOYING MODEL TO BATCH TRANSFORM")
print("="*70)

sm_client = boto3.client('sagemaker')

print("\n🔍 Finding latest approved model...")

# List models in registry
response = sm_client.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus='Approved',  # Only approved models
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

if not response['ModelPackageSummaryList']:
    print("\n❌ No approved models found!")
    print("   Please approve a model first:")
    print("   1. Go to SageMaker Studio → Model Registry")
    print("   2. Find 'flight-delay-models'")
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

role = get_execution_role()
session = sagemaker.Session()

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
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{BUCKET}/{PREFIX}/predictions/',
    assemble_with='Line',
    accept='text/csv',
    strategy='SingleRecord'  # Process one record at a time
)

print("✅ Transformer created")
print(f"   Instance: ml.m5.xlarge")
print(f"   Output: s3://{BUCKET}/{PREFIX}/predictions/")

# ===========================
# RUN BATCH PREDICTION (DEMO)
# ===========================
print("\n" + "="*70)
print("RUNNING BATCH PREDICTION (DEMO)")
print("="*70)

# For demo, use test set as input
input_data = f's3://{BUCKET}/{PREFIX}/data/inference/test_inference_small.csv'

print(f"\n📂 Input data: {input_data}")
print("⏱️  Starting batch transform job...")

job_name = f'flight-delay-predictions-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

transformer.transform(
    data=input_data,
    content_type='text/csv',
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
s3 = boto3.client('s3')
response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix=f'{PREFIX}/predictions/{job_name}/'
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
    s3.download_file(BUCKET, output_file, local_file)
    
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