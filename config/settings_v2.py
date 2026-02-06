"""
Centralized configuration (v2) copied into the `config` package for
consistent imports across scripts.

This file mirrors the implementation in `scripts/settings_v2.py` so that
`from config import settings_v2` and `importlib.reload(config.settings_v2)`
work as expected in notebooks and scripts.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import boto3
import sagemaker
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

REGION: str = os.environ.get("AWS_REGION", boto3.Session().region_name or "us-east-1")

sagemaker_session: sagemaker.Session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

ROLE: str = os.environ.get("SAGEMAKER_ROLE", sagemaker.get_execution_role())

# Use shared public bucket by default, but allow override via env var
PUBLIC_BUCKET: str = "sagemaker-us-east-1-425709451100"
BUCKET: str = os.environ.get("SAGEMAKER_BUCKET") or PUBLIC_BUCKET

PREFIX: str = os.environ.get("SAGEMAKER_PREFIX", "aai540-group1")
PROJECT_NAME: str = PREFIX
MODEL_PACKAGE_GROUP: str = "flight-delay-models"
PIPELINE_NAME: str = "FlightDelayTrainingPipeline"

def s3_uri(*parts: str) -> str:
    path = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{BUCKET}/{PREFIX}/{path}/"

S3_PATHS: Dict[str, str] = {
    "raw_data": s3_uri("data", "raw"),
    "processed_data": s3_uri("data", "processed"),
    "parquet_data": s3_uri("data", "parquet"),
    "features": s3_uri("features"),
    "athena_staging": s3_uri("athena", "staging"),
    "train": s3_uri("data", "train"),
    "validation": s3_uri("data", "validation"),
    "test": s3_uri("data", "test"),
    "models": s3_uri("models"),
    "evaluation": s3_uri("evaluation"),
    "predictions": s3_uri("predictions"),
    "inference_input": s3_uri("data", "inference"),
    "scripts": s3_uri("scripts"),
    "monitoring": s3_uri("monitoring"),
}

def get_s3_path(key: str) -> str:
    return S3_PATHS.get(key, s3_uri(""))

TRAINING_INSTANCE_TYPE: str = "ml.m5.xlarge"
TRAINING_INSTANCE_COUNT: int = 1

PROCESSING_INSTANCE_TYPE: str = "ml.m5.xlarge"
PROCESSING_INSTANCE_COUNT: int = 1

TRANSFORM_INSTANCE_TYPE: str = "ml.m5.xlarge"
TRANSFORM_INSTANCE_COUNT: int = 1

XGBOOST_FRAMEWORK_VERSION: str = "1.5-1"

def xgboost_image_uri() -> str:
    return sagemaker.image_uris.retrieve(framework="xgboost", region=REGION, version=XGBOOST_FRAMEWORK_VERSION)

DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.1,
    "num_round": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5.5,
    "min_child_weight": 1,
}

F1_THRESHOLD: float = float(os.environ.get("F1_THRESHOLD", "0.50"))
PREDICTION_THRESHOLD: float = 0.5
MODEL_APPROVAL_STATUS: str = "PendingManualApproval"

PROCESSING_MODEL_PATH: str = "/opt/ml/processing/model"
PROCESSING_TEST_PATH: str = "/opt/ml/processing/test"
PROCESSING_EVALUATION_PATH: str = "/opt/ml/processing/evaluation"

XGBOOST_MODEL_FILENAME: str = "xgboost-model"

ATHENA_DATABASE: str = "aai540_group1_db"
ATHENA_WORKGROUP: str = "primary"

DATASET_NAME: str = "flight-delays"
DATASET_KAGGLE_ID: str = "usdot/flight-delays"

DATA_FILES: Dict[str, str] = {
    "flights": "flights.csv",
    "airlines": "airlines.csv",
    "airports": "airports.csv",
}

def ensure_model_package_group_exists(group_name: str = MODEL_PACKAGE_GROUP, description: str = "Flight delay prediction model versions") -> str:
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        resp = sm.describe_model_package_group(ModelPackageGroupName=group_name)
        logger.info("Model package group '%s' already exists.", group_name)
        return resp["ModelPackageGroupArn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFound"):
            raise
    resp = sm.create_model_package_group(ModelPackageGroupName=group_name, ModelPackageGroupDescription=description)
    logger.info("Created model package group '%s'.", group_name)
    return resp["ModelPackageGroupArn"]

def upload_to_s3(local_path: str, s3_key: str, *, bucket: str = BUCKET, overwrite: bool = True) -> str:
    s3 = boto3.client("s3", region_name=REGION)
    s3_uri_result = f"s3://{bucket}/{s3_key}"
    if not overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            logger.info("S3 object already exists (skipping): %s", s3_uri_result)
            return s3_uri_result
        except ClientError:
            pass
    s3.upload_file(local_path, bucket, s3_key)
    logger.info("Uploaded %s → %s", local_path, s3_uri_result)
    return s3_uri_result

def ensure_pipeline_exists(pipeline_name: str = PIPELINE_NAME) -> bool:
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.describe_pipeline(PipelineName=pipeline_name)
        logger.info("Pipeline '%s' exists.", pipeline_name)
        return True
    except ClientError:
        logger.info("Pipeline '%s' does not exist yet.", pipeline_name)
        return False

def print_config() -> None:
    print("=" * 70)
    print("AAI-540 Group 1 — Active Configuration")
    print("=" * 70)
    print(f"  Region:               {REGION}")
    print(f"  S3 Bucket:            {BUCKET}")
    print(f"  Project Prefix:       {PREFIX}")
    print(f"  Pipeline Name:        {PIPELINE_NAME}")
    print(f"  Model Package Group:  {MODEL_PACKAGE_GROUP}")
    print(f"  XGBoost Container:    {XGBOOST_FRAMEWORK_VERSION}")
    print(f"  F1 Threshold:         {F1_THRESHOLD}")
    print(f"  Training Instance:    {TRAINING_INSTANCE_TYPE}")
    print(f"  Athena Database:      {ATHENA_DATABASE}")
    print()
    print("  Hyperparameter Defaults:")
    for k, v in DEFAULT_HYPERPARAMETERS.items():
        print(f"    {k:20s} = {v}")
    print()
    print("  Key S3 Paths:")
    for k, v in S3_PATHS.items():
        print(f"    {k:20s} → {v}")
    print("=" * 70)
