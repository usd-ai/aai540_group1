"""
Prepare raw data splits from original CSV files for SageMaker Pipeline
Matches the exact preprocessing from the notebook
Uses centralized configuration from settings.py
"""
import pandas as pd
import numpy as np
import boto3
import settings as cfg

# ============================================================================
# Configuration
# ============================================================================
BASE_URL = f"https://{cfg.BUCKET}.s3.{cfg.REGION}.amazonaws.com/{cfg.PREFIX}/data/raw/"
OUTPUT_PREFIX = f'{cfg.PREFIX}/pipeline-data/raw/'
DELAY_THRESHOLD = 15  # Industry standard

print("=" * 70)
print("PREPARING DATA FOR SAGEMAKER PIPELINE")
print("=" * 70)
print(f"Region: {cfg.REGION}")
print(f"Bucket: {cfg.BUCKET}")
print(f"Source: {cfg.PREFIX}/data/raw/")
print(f"Output: {OUTPUT_PREFIX}")

# ============================================================================
# 1. Load Raw Data
# ============================================================================
print("\n" + "=" * 70)
print("LOADING DATA FROM S3")
print("=" * 70)

print("Loading flights.csv (this may take 30-60 seconds)...")
df_flights = pd.read_csv(BASE_URL + "flights.csv", low_memory=False)
print(f"✓ Flights loaded: {len(df_flights):,} rows × {df_flights.shape[1]} columns")

print("\nLoading airlines.csv...")
df_airlines = pd.read_csv(BASE_URL + "airlines.csv")
print(f"✓ Airlines loaded: {len(df_airlines):,} rows")

print("\nLoading airports.csv...")
df_airports = pd.read_csv(BASE_URL + "airports.csv")
print(f"✓ Airports loaded: {len(df_airports):,} rows")

# ============================================================================
# 2. Data Cleaning (EXACTLY as in notebook)
# ============================================================================
print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

print(f"Original: {len(df_flights):,} rows")

# Remove cancelled and diverted flights
df = df_flights[
    (df_flights['CANCELLED'] == 0) & (df_flights['DIVERTED'] == 0)
].copy()
print(f"After removing cancelled/diverted: {len(df):,} rows")

# Select columns for modeling
cols = [
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
    'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE', 'ARRIVAL_DELAY'
]
df = df[cols].copy()

# Remove rows with missing ARRIVAL_DELAY
before = len(df)
df = df[df['ARRIVAL_DELAY'].notna()].copy()
removed = before - len(df)
print(f"✓ Removed {removed:,} rows with missing ARRIVAL_DELAY")

# Remove rows with missing SCHEDULED_TIME or DISTANCE
before = len(df)
df = df[
    df['SCHEDULED_TIME'].notna() & 
    df['DISTANCE'].notna()
].copy()
removed = before - len(df)
if removed > 0:
    print(f"✓ Removed {removed:,} rows with missing SCHEDULED_TIME/DISTANCE")

# ============================================================================
# 3. Feature Selection
# ============================================================================
print("\n--- Selecting Relevant Features ---")

COLS_TO_KEEP = [
    # Temporal features
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
    
    # Route and flight characteristics
    'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
    'DISTANCE', 'SCHEDULED_TIME',
    
    # Target variable
    'ARRIVAL_DELAY'
]

df = df[COLS_TO_KEEP].copy()

# ============================================================================
# 4. Create Target Variable 
# ============================================================================
print("\n--- Creating Target Variable ---")

# Create binary target: DELAYED = 1 if ARRIVAL_DELAY > 15 min
df['DELAYED'] = (df['ARRIVAL_DELAY'] > DELAY_THRESHOLD).astype(int)

delayed_count = (df['DELAYED'] == 1).sum()
ontime_count = (df['DELAYED'] == 0).sum()
delay_rate = delayed_count / len(df) * 100

print(f"\nTarget variable: DELAYED (1 if ARRIVAL_DELAY > {DELAY_THRESHOLD} min)")
print(f"  Delayed (1):  {delayed_count:>10,} ({delay_rate:>5.2f}%)")
print(f"  On-time (0):  {ontime_count:>10,} ({100-delay_rate:>5.2f}%)")
print(f"  Class ratio:  {ontime_count/delayed_count:.2f}:1 (on-time:delayed)")

# Drop ARRIVAL_DELAY (no longer needed)
df.drop('ARRIVAL_DELAY', axis=1, inplace=True)

print(f"\nFinal dataset: {len(df):,} rows x {df.shape[1]} columns")

# ============================================================================
# 5. Temporal Split 
# ============================================================================
print("\n" + "=" * 70)
print("TEMPORAL DATA SPLIT")
print("=" * 70)

print("\n--- Splitting Data by Month ---")

train_df = df[df['MONTH'] <= 9].copy()   # Jan-Sep
val_df = df[df['MONTH'] == 10].copy()    # Oct
test_df = df[df['MONTH'] == 11].copy()   # Nov
prod_df = df[df['MONTH'] == 12].copy()   # Dec

total_rows = len(df)

datasets = [
    ('Training', train_df, 'Jan-Sep (1-9)'),
    ('Validation', val_df, 'Oct (10)'),
    ('Test', test_df, 'Nov (11)'),
    ('Production', prod_df, 'Dec (12)')
]

print(f"{'Dataset':<12} {'Rows':>10}  {'Percent':>7}  {'Delay Rate':>11}  {'Months'}")
print("-" * 70)

for name, data, month_range in datasets:
    n_rows = len(data)
    pct = n_rows / total_rows * 100
    delay_rate = data['DELAYED'].mean() * 100
    print(f"{name:<12} {n_rows:>10,}  {pct:>6.1f}%    {delay_rate:>6.2f}%      {month_range}")

# ============================================================================
# 6. Detailed Statistics
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

for name, data, _ in datasets:
    print(f"\n{name} Dataset:")
    print(f"  Rows: {len(data):,}")
    print(f"  Delayed: {(data['DELAYED'] == 1).sum():,} ({data['DELAYED'].mean()*100:.2f}%)")
    print(f"  Airlines: {data['AIRLINE'].nunique()}")
    print(f"  Airports: {data['ORIGIN_AIRPORT'].nunique()} origin, {data['DESTINATION_AIRPORT'].nunique()} dest")
    print(f"  Avg distance: {data['DISTANCE'].mean():.1f} miles")
    print(f"  Avg scheduled time: {data['SCHEDULED_TIME'].mean():.1f} min")

# ============================================================================
# 7. Save as CSV
# ============================================================================
print("\n" + "=" * 70)
print("SAVING CSV FILES")
print("=" * 70)

# Final columns (no ARRIVAL_DELAY, includes DELAYED)
RAW_COLUMNS = [
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
    'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
    'DISTANCE', 'SCHEDULED_TIME', 'DELAYED'
]

train_df[RAW_COLUMNS].to_csv('train_raw.csv', index=False)
val_df[RAW_COLUMNS].to_csv('val_raw.csv', index=False)
test_df[RAW_COLUMNS].to_csv('test_raw.csv', index=False)

print(f"✓ train_raw.csv ({len(train_df):,} rows)")
print(f"✓ val_raw.csv   ({len(val_df):,} rows)")
print(f"✓ test_raw.csv  ({len(test_df):,} rows)")

# ============================================================================
# 8. Upload to S3
# ============================================================================
print("\n" + "=" * 70)
print("UPLOADING TO S3")
print("=" * 70)

s3 = boto3.client('s3', region_name=cfg.REGION)

files_to_upload = {
    'train_raw.csv': f'{OUTPUT_PREFIX}train.csv',
    'val_raw.csv': f'{OUTPUT_PREFIX}val.csv',
    'test_raw.csv': f'{OUTPUT_PREFIX}test.csv'
}

for local_file, s3_key in files_to_upload.items():
    s3.upload_file(local_file, cfg.BUCKET, s3_key)
    size_mb = pd.read_csv(local_file).memory_usage(deep=True).sum() / 1024 / 1024
    print(f"✓ s3://{cfg.BUCKET}/{s3_key} ({size_mb:.1f} MB)")

# ============================================================================
# 9. Verify Upload
# ============================================================================
print("\n--- Verifying S3 uploads ---")

response = s3.list_objects_v2(Bucket=cfg.BUCKET, Prefix=OUTPUT_PREFIX)

if 'Contents' in response:
    print(f"\nFiles in s3://{cfg.BUCKET}/{OUTPUT_PREFIX}:")
    for obj in response['Contents']:
        file_size_mb = obj['Size'] / 1024 / 1024
        print(f"  ✓ {obj['Key'].split('/')[-1]} ({file_size_mb:.1f} MB)")

# ============================================================================
# 10. Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ DATA PREPARATION COMPLETE")
print("=" * 70)

print(f"\n📊 Summary:")
print(f"   Total rows processed: {len(df):,}")
print(f"   Train/Val/Test/Prod: {len(train_df):,} / {len(val_df):,} / {len(test_df):,} / {len(prod_df):,}")
print(f"   Overall delay rate: {df['DELAYED'].mean()*100:.2f}%")
print(f"   Output: s3://{cfg.BUCKET}/{OUTPUT_PREFIX}")

print(f"\n📋 Raw Columns (for Feature Engineering):")
for i, col in enumerate(RAW_COLUMNS, 1):
    print(f"   {i:2d}. {col}")

print(f"\n🔧 Next Steps:")
print(f"   1. Feature engineering will create 20 features from these 10 raw columns")
print(f"   2. Run: python pipeline_definition.py")
print(f"   3. Run: python run_experiment.py --experiment baseline")

print("=" * 70)