"""
Configuration settings for AAI-540 Group 1 Flight Delay Prediction project.

This module centralizes AWS configuration, S3 paths, and project constants.
"""

import boto3
import sagemaker

# AWS Configuration
REGION = boto3.Session().region_name or 'us-east-1'

# SageMaker session and default bucket
sagemaker_session = sagemaker.Session()
DEFAULT_BUCKET = sagemaker_session.default_bucket()

# Project Configuration
PROJECT_NAME = 'aai540-group1'
PROJECT_PREFIX = f'{PROJECT_NAME}/'

# S3 Path Structure
S3_BASE_URI = f's3://{DEFAULT_BUCKET}/{PROJECT_PREFIX}'

S3_PATHS = {
    # Raw data storage
    'raw_data': f'{S3_BASE_URI}data/raw/',
    'processed_data': f'{S3_BASE_URI}data/processed/',
    'parquet_data': f'{S3_BASE_URI}data/parquet/',
    
    # Feature store
    'features': f'{S3_BASE_URI}features/',
    
    # Athena
    'athena_staging': f'{S3_BASE_URI}athena/staging/',
    
    # Training
    'training_input': f'{S3_BASE_URI}training/input/',
    'training_output': f'{S3_BASE_URI}training/output/',
    
    # Evaluation
    'evaluation': f'{S3_BASE_URI}evaluation/',
    
    # Inference
    'batch_inference': f'{S3_BASE_URI}inference/batch/',
    
    # Monitoring
    'monitoring': f'{S3_BASE_URI}monitoring/',
}

# Athena Configuration
ATHENA_DATABASE = 'aai540_group1_db'
ATHENA_WORKGROUP = 'primary'

# Dataset Information
DATASET_NAME = 'flight-delays'
DATASET_KAGGLE_ID = 'usdot/flight-delays'

# Expected data files
DATA_FILES = {
    'flights': 'flights.csv',
    'airlines': 'airlines.csv',
    'airports': 'airports.csv'
}

def get_s3_path(key):
    """Get S3 path by key."""
    return S3_PATHS.get(key, S3_BASE_URI)

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("AAI-540 Group 1 Project Configuration")
    print("=" * 60)
    print(f"Region: {REGION}")
    print(f"S3 Bucket: {DEFAULT_BUCKET}")
    print(f"Project Prefix: {PROJECT_PREFIX}")
    print(f"Athena Database: {ATHENA_DATABASE}")
    print(f"\nS3 Base URI: {S3_BASE_URI}")
    print("=" * 60)
