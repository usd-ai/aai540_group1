"""
Feature Engineering Script for SageMaker Processing
Reads CSV files, creates features, outputs CSV for training
Matches the exact notebook feature engineering logic
"""
import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--use-advanced-features', type=str, default='true')
    return parser.parse_args()

def add_temporal_features(data):
    """Add time-based features (from notebook)"""
    # Extract hour from SCHEDULED_DEPARTURE (HHMM format)
    data['DEP_HOUR'] = data['SCHEDULED_DEPARTURE'] // 100
    
    # Cyclical encoding (captures 23:00 is close to 01:00)
    data['HOUR_SIN'] = np.sin(2 * np.pi * data['DEP_HOUR'] / 24)
    data['HOUR_COS'] = np.cos(2 * np.pi * data['DEP_HOUR'] / 24)
    
    # Peak hours: morning (6-9) or evening (16-20)
    is_morning = (data['DEP_HOUR'] >= 6) & (data['DEP_HOUR'] <= 9)
    is_evening = (data['DEP_HOUR'] >= 16) & (data['DEP_HOUR'] <= 20)
    data['IS_PEAK_HOUR'] = (is_morning | is_evening).astype(int)
    
    # Weekend indicator
    data['IS_WEEKEND'] = (data['DAY_OF_WEEK'] >= 6).astype(int)
    
    return data

def add_distance_features(data):
    """Add distance-based features (from notebook)"""
    # Long-haul flights (> 1500 miles)
    data['IS_LONG_HAUL'] = (data['DISTANCE'] > 1500).astype(int)
    
    # Distance buckets: Short (<500), Medium (500-1500), Long (>1500)
    data['DISTANCE_BUCKET'] = pd.cut(
        data['DISTANCE'],
        bins=[0, 500, 1500, 5000],
        labels=[0, 1, 2]
    ).astype(int)
    
    return data

def main():
    args = parse_args()
    
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    # ============================================================================
    # Load CSV Data
    # ============================================================================
    print("\nLoading raw CSV data...")
    print(f"Input directory: {args.input_data}")
    
    # List files
    if os.path.exists(args.input_data):
        files = os.listdir(args.input_data)
        print(f"Files found: {files}")
    else:
        print(f"ERROR: Input directory does not exist")
        return
    
    # Read CSV files
    train_df = pd.read_csv(os.path.join(args.input_data, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.input_data, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.input_data, 'test.csv'))

    # Optionally load production data (December) if present
    prod_path = os.path.join(args.input_data, 'prod.csv')
    prod_df = None
    if os.path.exists(prod_path):
        prod_df = pd.read_csv(prod_path)
        print(f"Prod:  {prod_df.shape}")

    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")
    
    # ============================================================================
    # Convert categorical columns to strings
    # ============================================================================
    print("\n--- Converting categorical columns to strings ---")
    
    all_dfs = [train_df, val_df, test_df] + ([prod_df] if prod_df is not None else [])
    for df in all_dfs:
        df['AIRLINE'] = df['AIRLINE'].astype(str)
        df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
        df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].astype(str)
    
    print("✓ AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT converted to strings")
    print(f"Sample values:")
    print(f"  AIRLINE: {train_df['AIRLINE'].iloc[0]}")
    print(f"  ORIGIN_AIRPORT: {train_df['ORIGIN_AIRPORT'].iloc[0]}")
    print(f"  DESTINATION_AIRPORT: {train_df['DESTINATION_AIRPORT'].iloc[0]}")
    
    # ============================================================================
    # Temporal Features
    # ============================================================================
    print("\n--- Creating Temporal Features ---")
    
    for data in all_dfs:
        add_temporal_features(data)

    print("✓ DEP_HOUR, HOUR_SIN, HOUR_COS, IS_PEAK_HOUR, IS_WEEKEND")

    # ============================================================================
    # Distance Features
    # ============================================================================
    print("\n--- Creating Distance Features ---")

    for data in all_dfs:
        add_distance_features(data)
    
    print("✓ IS_LONG_HAUL, DISTANCE_BUCKET")
    
    # ============================================================================
    # Target Encoding (Computed from TRAIN only!)
    # ============================================================================
    print("\n--- Creating Target-Encoded Features ---")
    
    # Global delay rate (fallback for unseen categories)
    global_delay_rate = train_df['DELAYED'].mean()
    
    # Compute delay rates from TRAINING data only
    airline_delay = train_df.groupby('AIRLINE')['DELAYED'].mean().to_dict()
    origin_delay = train_df.groupby('ORIGIN_AIRPORT')['DELAYED'].mean().to_dict()
    dest_delay = train_df.groupby('DESTINATION_AIRPORT')['DELAYED'].mean().to_dict()
    
    # Create route ID (NOW SAFE - strings!)
    for data in all_dfs:
        data['ROUTE'] = data['ORIGIN_AIRPORT'] + '_' + data['DESTINATION_AIRPORT']

    route_delay = train_df.groupby('ROUTE')['DELAYED'].mean().to_dict()

    # Apply to all datasets (unseen categories get global rate)
    for data in all_dfs:
        data['AIRLINE_DELAY_RATE'] = data['AIRLINE'].map(airline_delay).fillna(global_delay_rate)
        data['ORIGIN_DELAY_RATE'] = data['ORIGIN_AIRPORT'].map(origin_delay).fillna(global_delay_rate)
        data['DEST_DELAY_RATE'] = data['DESTINATION_AIRPORT'].map(dest_delay).fillna(global_delay_rate)
        data['ROUTE_DELAY_RATE'] = data['ROUTE'].map(route_delay).fillna(global_delay_rate)
    
    print("✓ AIRLINE_DELAY_RATE, ORIGIN_DELAY_RATE, DEST_DELAY_RATE, ROUTE_DELAY_RATE")
    print(f"  Global delay rate (fallback): {global_delay_rate:.4f}")

    if args.use_advanced_features.lower() == 'true':  
        # IMPROVED MODEL: All features including target encoding and volume
        print("Building IMPROVED feature set (advanced features enabled)...")
        
        FEATURE_COLS = [
            # Temporal
            'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEP_HOUR', 'SCHEDULED_DEPARTURE',
            'HOUR_SIN', 'HOUR_COS', 'IS_PEAK_HOUR', 'IS_WEEKEND',
            
            # Distance
            'DISTANCE', 'SCHEDULED_TIME', 'IS_LONG_HAUL', 'DISTANCE_BUCKET',
            
            # Target-encoded (advanced features)
            'AIRLINE_DELAY_RATE', 'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE', 'ROUTE_DELAY_RATE'
        ]
        
        # Create volume features (advanced features)
        print("--- Creating Advanced Features (target encoding + volume) ---")
        origin_counts = train_df['ORIGIN_AIRPORT'].value_counts().to_dict()
        dest_counts = train_df['DESTINATION_AIRPORT'].value_counts().to_dict()
        route_counts = train_df['ROUTE'].value_counts().to_dict()
        
        # Apply log-scaled counts (log1p handles zeros gracefully)
        for data in all_dfs:
            data['ORIGIN_FLIGHTS'] = np.log1p(data['ORIGIN_AIRPORT'].map(origin_counts).fillna(0))
            data['DEST_FLIGHTS'] = np.log1p(data['DESTINATION_AIRPORT'].map(dest_counts).fillna(0))
            data['ROUTE_FLIGHTS'] = np.log1p(data['ROUTE'].map(route_counts).fillna(0))
        
        print("✓ Target encoding: AIRLINE_DELAY_RATE, ORIGIN_DELAY_RATE, DEST_DELAY_RATE, ROUTE_DELAY_RATE")
        print("✓ Volume features: ORIGIN_FLIGHTS, DEST_FLIGHTS, ROUTE_FLIGHTS")
        
        FEATURE_COLS.extend(['ORIGIN_FLIGHTS', 'DEST_FLIGHTS', 'ROUTE_FLIGHTS'])
        
        print(f"✓ IMPROVED feature set: {len(FEATURE_COLS)} features")
        
    else:
        # BASELINE MODEL: Basic features ONLY
        print("Building BASELINE feature set (basic features only)...")
        
        FEATURE_COLS = [
            # Temporal (basic)
            'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEP_HOUR',
            'HOUR_SIN', 'HOUR_COS', 'IS_PEAK_HOUR', 'IS_WEEKEND',
            
            # Distance (basic)
            'DISTANCE', 'SCHEDULED_TIME', 'IS_LONG_HAUL', 'DISTANCE_BUCKET'
        ]
        
        print(f"✓ BASELINE feature set: {len(FEATURE_COLS)} features")
        print("  Basic features only (temporal + distance)")
        print("  Advanced features excluded (target encoding + volume)")

    # ============================================================================
    # Prepare Final Feature Matrix
    # ============================================================================
    print("\n--- Preparing Final Features ---")
    
    TARGET_COL = 'DELAYED'
    
    # Create CSV with target first, then features (XGBoost format)
    final_columns = [TARGET_COL] + FEATURE_COLS
    
    train_final = train_df[final_columns]
    val_final = val_df[final_columns]
    test_final = test_df[final_columns]
    prod_final = prod_df[final_columns] if prod_df is not None else None

    print(f"\n✓ Feature count: {len(FEATURE_COLS)}")
    print(f"✓ Output columns: {len(final_columns)} (1 target + {len(FEATURE_COLS)} features)")
    print(f"✓ Output shapes:")
    print(f"  Train: {train_final.shape}")
    print(f"  Val:   {val_final.shape}")
    print(f"  Test:  {test_final.shape}")
    if prod_final is not None:
        print(f"  Prod:  {prod_final.shape}")
    
    # ============================================================================
    # Save as CSV (no headers for XGBoost)
    # ============================================================================
    print("\n--- Saving Features as CSV ---")
    
    os.makedirs(os.path.join(args.output_data, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_data, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_data, 'test'), exist_ok=True)

    train_final.to_csv(os.path.join(args.output_data, 'train', 'train.csv'),
                       header=False, index=False)
    val_final.to_csv(os.path.join(args.output_data, 'validation', 'val.csv'),
                     header=False, index=False)
    test_final.to_csv(os.path.join(args.output_data, 'test', 'test.csv'),
                      header=False, index=False)

    print("✓ train.csv (no headers)")
    print("✓ val.csv (no headers)")
    print("✓ test.csv (no headers)")

    if prod_final is not None:
        os.makedirs(os.path.join(args.output_data, 'production'), exist_ok=True)
        prod_final.to_csv(os.path.join(args.output_data, 'production', 'prod.csv'),
                          header=False, index=False)
        print("✓ prod.csv (no headers)")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Features ({len(FEATURE_COLS)}):")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"   {i:2d}. {col}")
    
    print("=" * 70)

if __name__ == '__main__':
    main()