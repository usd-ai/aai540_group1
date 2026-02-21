"""
Centralized configuration for AAI-540 Group 1 — Flight Delay Prediction.

This module is the **single source of truth** for every tunable value used
across the SageMaker pipeline scripts (pipeline_definition, run_experiment,
deploy_model, evaluate).

Design principles
─────────────────
1. All magic numbers / strings that were previously hardcoded in individual
   scripts live here.
2. Environment-variable overrides are supported so each teammate can point
   to their own bucket or region without editing code.
3. Idempotent helper functions are provided for common AWS resource-creation
   tasks (model-package group, S3 upload, etc.).
4. Values that are truly constant (container paths inside a SageMaker
   Processing job, XGBoost model filename) are still defined here so they
   can be referenced rather than scattered.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import sys

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.model import ModelPackage

# Optional: load environment variables from a .env file when present
# (useful for local development/test). This requires `python-dotenv`.
try:
    from dotenv import load_dotenv

    # Find and load a .env file if present in the repo root or current dir
    # Do not override already set environment variables by default.
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, skip silently — the module will
    # still work by reading from the real environment variables.
    pass

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────
# AWS / SageMaker Basics
# ──────────────────────────────────────────────
REGION: str = os.environ.get(
    "AWS_REGION",
    boto3.Session().region_name or "us-east-1",
)

sagemaker_session: sagemaker.Session = sagemaker.Session(
    boto_session=boto3.Session(region_name=REGION)
)

ROLE: str = os.environ.get("SAGEMAKER_ROLE")
if not ROLE:
    try:
        ROLE = sagemaker.get_execution_role()
    except ValueError:
        # Fallback for local testing without IAM role
        print("⚠️ Could not get execution role. Using dummy role for local testing.")
        ROLE = "arn:aws:iam::000000000000:role/dummy-role"

# Bucket — env-var takes precedence so each teammate can override.
# Fallback: the hardcoded shared bucket the team has been using.
# Prefer an explicit env var, otherwise use the SageMaker session default bucket
# (works in Studio / job containers). Keep a final hardcoded fallback.
BUCKET: str = os.environ.get("SAGEMAKER_BUCKET", "sagemaker-us-east-1-425709451100")

# ──────────────────────────────────────────────
# Project Identifiers
# ──────────────────────────────────────────────
PREFIX: str = os.environ.get("SAGEMAKER_PREFIX", "aai540-group1")
PROJECT_NAME: str = PREFIX                           # alias for readability
MODEL_PACKAGE_GROUP: str = "flight-delay-models"
PIPELINE_NAME: str = "FlightDelayTrainingPipeline"

# ──────────────────────────────────────────────
# S3 Path Helpers
# ──────────────────────────────────────────────
def s3_uri(*parts: str) -> str:
    """Build an S3 URI under the project prefix.

    >>> s3_uri("data", "train")
    's3://sagemaker-us-east-1-…/aai540-group1/data/train/'

    Trailing slashes are added automatically so the result is always
    a "folder" URI that SageMaker expects.
    """
    path = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{BUCKET}/{PREFIX}/{path}/"


# Canonical S3 paths consumed by the pipeline scripts.
S3_PATHS: Dict[str, str] = {
    # ── Raw / processed data ──
    "raw_data":          s3_uri("data", "raw"),
    "processed_data":    s3_uri("data", "processed"),
    "parquet_data":      s3_uri("data", "parquet"),
    # ── Feature store ──
    "features":          s3_uri("features"),
    # ── Athena ──
    "athena_staging":    s3_uri("athena", "staging"),
    # ── Training data splits ──
    "train":             s3_uri("data", "train"),
    "validation":        s3_uri("data", "validation"),
    "test":              s3_uri("data", "test"),
    # ── Model artefacts ──
    "models":            s3_uri("models"),
    # ── Evaluation output ──
    "evaluation":        s3_uri("evaluation"),
    # ── Inference ──
    "predictions":       s3_uri("predictions"),
    "inference_input":   s3_uri("data", "inference"),
    # ── Scripts (uploaded to S3 for Processing jobs) ──
    "scripts":           s3_uri("scripts"),
    # ── Processed data (feature-engineered pipeline outputs) ──
    "processed_train":       s3_uri("processed-data", "train"),
    "processed_validation":  s3_uri("processed-data", "validation"),
    "processed_test":        s3_uri("processed-data", "test"),
    "processed_production":  s3_uri("processed-data", "production"),
    # ── Monitoring ──
    "monitoring":        s3_uri("monitoring"),
    "data_capture":      s3_uri("monitoring", "datacapture"),
    "baseline_data":     s3_uri("monitoring", "baselining", "data"),
    "baseline_results":  s3_uri("monitoring", "baselining", "results"),
    "monitor_reports":   s3_uri("monitoring", "reports"),
    "ground_truth":      s3_uri("monitoring", "ground_truth"),
    # ── Pipeline data (cleaned splits) ──
    "pipeline_raw":      s3_uri("pipeline-data", "raw"),
}


def get_s3_path(key: str) -> str:
    """Return a canonical S3 path by key, or the base URI if the key is unknown."""
    return S3_PATHS.get(key, s3_uri(""))

# ──────────────────────────────────────────────
# Instance Types
# ──────────────────────────────────────────────
TRAINING_INSTANCE_TYPE: str = "ml.m5.xlarge"
TRAINING_INSTANCE_COUNT: int = 1

PROCESSING_INSTANCE_TYPE: str = "ml.m5.xlarge"
PROCESSING_INSTANCE_COUNT: int = 1

TRANSFORM_INSTANCE_TYPE: str = "ml.m5.xlarge"
TRANSFORM_INSTANCE_COUNT: int = 1

INFERENCE_INSTANCE_TYPES: list[str] = ["ml.m5.xlarge"]
TRANSFORM_INSTANCE_TYPES: list[str] = ["ml.m5.xlarge"]

MONITORING_INSTANCE_TYPE: str = "ml.m5.xlarge"
MONITORING_INSTANCE_COUNT: int = 1

# ──────────────────────────────────────────────
# XGBoost Container
# ──────────────────────────────────────────────
XGBOOST_FRAMEWORK_VERSION: str = "1.5-1"    # SageMaker built-in container tag


def xgboost_image_uri() -> str:
    """Return the ECR URI for the SageMaker XGBoost built-in container."""
    return sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version=XGBOOST_FRAMEWORK_VERSION,
    )

# ──────────────────────────────────────────────
# Hyperparameter Defaults  (single canonical set)
# ──────────────────────────────────────────────
DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "max_depth":         6,
    "eta":               0.1,
    "num_round":         100,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  5.5,
    "min_child_weight":  1,
}

# ──────────────────────────────────────────────
# Pipeline Thresholds
# ──────────────────────────────────────────────
# Model-quality gate: the minimum F1 score required for automatic
# registration into the Model Registry.
F1_THRESHOLD: float = float(os.environ.get("F1_THRESHOLD", "0.20"))

# Prediction threshold used in evaluate.py to turn probabilities into classes
PREDICTION_THRESHOLD: float = 0.5

# Approval status assigned to newly registered model packages
MODEL_APPROVAL_STATUS: str = "PendingManualApproval"

# ──────────────────────────────────────────────
# SageMaker Processing-Job Paths
#   (these are container-internal and rarely change)
# ──────────────────────────────────────────────
PROCESSING_MODEL_PATH: str = "/opt/ml/processing/model"
PROCESSING_TEST_PATH: str = "/opt/ml/processing/test"
PROCESSING_EVALUATION_PATH: str = "/opt/ml/processing/evaluation"

# XGBoost model filename produced by the SageMaker built-in container
XGBOOST_MODEL_FILENAME: str = "xgboost-model"

# ──────────────────────────────────────────────
# Athena / Data-set Metadata  (carried over from v1)
# ──────────────────────────────────────────────
ATHENA_DATABASE: str = "aai540_group1_db"
ATHENA_WORKGROUP: str = "primary"

DATASET_NAME: str = "flight-delays"
DATASET_KAGGLE_ID: str = "usdot/flight-delays"

DATA_FILES: Dict[str, str] = {
    "flights":  "flights.csv",
    "airlines": "airlines.csv",
    "airports": "airports.csv",
}

# ──────────────────────────────────────────────
# Idempotent Helper Functions
# ──────────────────────────────────────────────

def ensure_model_package_group_exists(
    group_name: str = MODEL_PACKAGE_GROUP,
    description: str = "Flight delay prediction model versions",
) -> str:
    """Create the Model Package Group if it doesn't already exist.

    Returns the group ARN in both cases (created or already existed).
    """
    sm = boto3.client("sagemaker", region_name=REGION)

    try:
        resp = sm.describe_model_package_group(
            ModelPackageGroupName=group_name
        )
        logger.info("Model package group '%s' already exists.", group_name)
        return resp["ModelPackageGroupArn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in (
            "ValidationException",
            "ResourceNotFound",
        ):
            raise

    # Group does not exist — create it.
    resp = sm.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=description,
    )
    logger.info("Created model package group '%s'.", group_name)
    return resp["ModelPackageGroupArn"]


def upload_to_s3(
    local_path: str,
    s3_key: str,
    *,
    bucket: str = BUCKET,
    overwrite: bool = True,
) -> str:
    """Upload a local file to S3. Idempotent: skips if the object already
    exists and *overwrite* is ``False``.

    Returns the full ``s3://`` URI.
    """
    s3 = boto3.client("s3", region_name=REGION)
    s3_uri_result = f"s3://{bucket}/{s3_key}"

    if not overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            logger.info("S3 object already exists (skipping): %s", s3_uri_result)
            return s3_uri_result
        except ClientError:
            pass  # Object does not exist — proceed with upload

    s3.upload_file(local_path, bucket, s3_key)
    logger.info("Uploaded %s → %s", local_path, s3_uri_result)
    return s3_uri_result


def ensure_pipeline_exists(pipeline_name: str = PIPELINE_NAME) -> bool:
    """Return True if the SageMaker Pipeline already exists."""
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        sm.describe_pipeline(PipelineName=pipeline_name)
        logger.info("Pipeline '%s' exists.", pipeline_name)
        return True
    except ClientError:
        logger.info("Pipeline '%s' does not exist yet.", pipeline_name)
        return False


# ──────────────────────────────────────────────
# Model Monitoring
# ──────────────────────────────────────────────
ENDPOINT_NAME_PREFIX: str = "flight-delay-endpoint"
MONITOR_SCHEDULE_PREFIX: str = "flight-delay-monitor"
DATA_CAPTURE_SAMPLING_PERCENTAGE: int = 100
MONITORING_MAX_RUNTIME_SECONDS: int = 1800
MONITORING_VOLUME_SIZE_GB: int = 20
MONITORING_PROBLEM_TYPE: str = "BinaryClassification"

# Monitoring schedule frequency: "hourly" or "daily"
# "hourly" = cron(0 * ? * * *)  — good for demos/testing
# "daily"  = cron(0 0 ? * * *)  — typical for production (runs at midnight UTC)
MONITORING_SCHEDULE_FREQUENCY: str = os.environ.get("MONITORING_FREQUENCY", "daily")

# Number of test-set records to use for baselining (need >= 200 for std-dev)
BASELINE_SAMPLE_SIZE: int = 500

# Number of December (production) records to stream through the endpoint
# December data is uploaded to s3://.../pipeline-data/raw/prod.csv
PRODUCTION_SAMPLE_SIZE: int = 500

# CloudWatch alarm settings
CW_ALARM_METRIC: str = "binary_classification_f1"
CW_ALARM_THRESHOLD: float = F1_THRESHOLD
CW_ALARM_PERIOD_SECONDS: int = 600
CW_ALARM_EVALUATION_PERIODS: int = 1

def find_approved_model() -> dict:
    """Return the latest approved model package from the registry.

    Exits with code 1 if no approved model is found.
    """
    sm = boto3.client("sagemaker", region_name=REGION)
    resp = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not resp["ModelPackageSummaryList"]:
        logger.error(
            "No approved models found in '%s'. "
            "Approve a model in SageMaker Studio first.",
            MODEL_PACKAGE_GROUP,
        )
        sys.exit(1)
    return resp["ModelPackageSummaryList"][0]


def create_model_from_registry(model_package_arn: str) -> ModelPackage:
    """Create a SageMaker ModelPackage object from a registry ARN."""
    return ModelPackage(
        role=ROLE,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )


def monitor_image_uri() -> str:
    """Return the ECR URI for the SageMaker Model Monitor container."""
    return sagemaker.image_uris.retrieve(
        framework="model-monitor",
        region=REGION,
    )

# ──────────────────────────────────────────────
# Pretty-print Config
# ──────────────────────────────────────────────

def print_config() -> None:
    """Print the active configuration to stdout."""
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
