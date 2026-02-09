"""
Model Quality Monitoring for Flight Delay Prediction (Step 5)

Sets up SageMaker Model Quality Monitor on an already-deployed endpoint:
  1. Validates that the endpoint exists
  2. Generates baseline from test data (November) predictions
  3. Streams production data (December) through the endpoint
  4. Uploads ground truth for production data
  5. Creates a monitoring schedule to detect model quality drift
  6. Sets up a CloudWatch alarm on F1 score

Execution order:
  Step 0-3:  Pipeline (FE -> Train -> Eval -> Register)  [includes production split]
  Step 4:    deploy_model.py realtime   (endpoint with data capture)
  Step 5:    monitor_model.py           (this script)

Usage:
  python monitor_model.py --endpoint-name <name>                # full setup

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from time import sleep

import boto3
import pandas as pd
from sagemaker.model_monitor import (
    EndpointInput,
    ModelQualityMonitor,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.predictor import Predictor
from sagemaker.s3 import S3Downloader, S3Uploader
from sagemaker.serializers import CSVSerializer

import settings as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")


def _download_processed_data(s3_prefix: str, local_path: str) -> str:
    """Download processed CSV files from an S3 prefix. Returns path to first CSV found."""
    files = S3Downloader.list(s3_prefix)
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        logger.error("No CSV files found at %s", s3_prefix)
        sys.exit(1)
    S3Downloader.download(csv_files[0], local_path)
    filename = csv_files[0].split("/")[-1]
    return f"{local_path}/{filename}"


def _validate_endpoint(endpoint_name: str) -> None:
    """Verify the endpoint exists and is InService."""
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    try:
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        if status != "InService":
            logger.error("Endpoint '%s' exists but status is '%s' (expected InService)", endpoint_name, status)
            sys.exit(1)
        print(f"  Endpoint '{endpoint_name}' is InService")
    except sm.exceptions.ClientError:
        logger.error(
            "Endpoint '%s' not found. Deploy it first:\n"
            "  python deploy_model.py realtime --endpoint-name %s",
            endpoint_name, endpoint_name,
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Step 1 — Generate baseline from test data (November)
# ─────────────────────────────────────────────────────────────

def generate_baseline(predictor: Predictor, endpoint_name: str) -> ModelQualityMonitor:
    """Send test data through the endpoint and create model quality baseline."""
    print(f"\n{'='*70}")
    print("STEP 1: GENERATING MODEL QUALITY BASELINE (Test Data - November)")
    print(f"{'='*70}")

    # Download processed test data (headerless CSV: col0=DELAYED, cols1+=features)
    test_s3_path = cfg.get_s3_path("processed_test")
    local_test = _download_processed_data(test_s3_path, "/tmp/monitor_test")
    test_data = pd.read_csv(local_test, header=None)

    print(f"  Test data loaded: {test_data.shape}")
    print(f"  Sampling {cfg.BASELINE_SAMPLE_SIZE} records for baseline...")

    sample = test_data.head(cfg.BASELINE_SAMPLE_SIZE)
    labels = sample.iloc[:, 0].values           # ground truth (DELAYED)
    features = sample.iloc[:, 1:]               # feature columns

    # Send each record through the endpoint and collect predictions
    baseline_file = "/tmp/monitor_baseline/baseline_predictions.csv"
    os.makedirs("/tmp/monitor_baseline", exist_ok=True)

    with open(baseline_file, "w") as f:
        f.write("probability,prediction,label\n")
        for i in range(len(features)):
            row = features.iloc[i].values
            csv_row = ",".join(str(v) for v in row)
            probability = float(predictor.predict(csv_row))
            prediction = 1 if probability >= cfg.PREDICTION_THRESHOLD else 0
            f.write(f"{probability},{prediction},{int(labels[i])}\n")
            if (i + 1) % 50 == 0:
                print(f"    Baseline predictions: {i+1}/{cfg.BASELINE_SAMPLE_SIZE}")
            sleep(0.2)

    print(f"  Baseline predictions complete: {cfg.BASELINE_SAMPLE_SIZE} records")

    # Upload baseline predictions to S3
    baseline_data_uri = cfg.get_s3_path("baseline_data")
    baseline_dataset_uri = S3Uploader.upload(baseline_file, baseline_data_uri)
    print(f"  Uploaded baseline to: {baseline_dataset_uri}")

    # Create model quality monitor and run baseline job
    session = cfg.sagemaker_session
    monitor = ModelQualityMonitor(
        role=cfg.ROLE,
        instance_count=cfg.MONITORING_INSTANCE_COUNT,
        instance_type=cfg.MONITORING_INSTANCE_TYPE,
        volume_size_in_gb=cfg.MONITORING_VOLUME_SIZE_GB,
        max_runtime_in_seconds=cfg.MONITORING_MAX_RUNTIME_SECONDS,
        sagemaker_session=session,
    )

    baseline_job_name = f"flight-delay-baseline-{_timestamp()}"
    baseline_results_uri = cfg.get_s3_path("baseline_results")

    print(f"\n  Running baseline job: {baseline_job_name}")
    job = monitor.suggest_baseline(
        job_name=baseline_job_name,
        baseline_dataset=baseline_dataset_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        problem_type=cfg.MONITORING_PROBLEM_TYPE,
        inference_attribute="prediction",
        probability_attribute="probability",
        ground_truth_attribute="label",
    )
    job.wait(logs=False)

    # Display baseline metrics
    baseline_job = monitor.latest_baselining_job
    stats = baseline_job.baseline_statistics().body_dict
    metrics = stats.get("binary_classification_metrics", {})
    print(f"\n  Baseline Metrics:")
    for key in ["f1", "precision", "recall", "accuracy", "auc"]:
        val = metrics.get(key, {}).get("value", "N/A")
        print(f"    {key:12s}: {val}")

    constraints = baseline_job.suggested_constraints().body_dict
    print(f"\n  Suggested Constraints:")
    for name, info in constraints.get("binary_classification_constraints", {}).items():
        print(f"    {name:24s}: {info['threshold']:.4f} ({info['comparison_operator']})")

    return monitor


# ─────────────────────────────────────────────────────────────
# Step 2 — Stream production data (December) through endpoint
# ─────────────────────────────────────────────────────────────

def stream_production_data(predictor: Predictor, endpoint_name: str) -> str:
    """Stream December (production) data through the endpoint and upload ground truth."""
    print(f"\n{'='*70}")
    print("STEP 2: STREAMING PRODUCTION DATA (December)")
    print(f"{'='*70}")

    # Download processed production data
    prod_s3_path = cfg.get_s3_path("processed_production")
    local_prod = _download_processed_data(prod_s3_path, "/tmp/monitor_prod")
    prod_data = pd.read_csv(local_prod, header=None)

    print(f"  Production data loaded: {prod_data.shape}")
    print(f"  Streaming {cfg.PRODUCTION_SAMPLE_SIZE} records...")

    sample = prod_data.head(cfg.PRODUCTION_SAMPLE_SIZE)
    labels = sample.iloc[:, 0].values
    features = sample.iloc[:, 1:]

    # Stream records through the endpoint with inference IDs
    for i in range(len(features)):
        row = features.iloc[i].values
        csv_row = ",".join(str(v) for v in row)
        cfg.sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=csv_row,
            InferenceId=str(i),
        )
        if (i + 1) % 100 == 0:
            print(f"    Streamed: {i+1}/{cfg.PRODUCTION_SAMPLE_SIZE}")
        sleep(0.5)

    print(f"  Streaming complete: {cfg.PRODUCTION_SAMPLE_SIZE} records")

    # Upload ground truth labels (matching by inference ID)
    gt_records = []
    for i in range(len(labels)):
        gt_records.append(json.dumps({
            "groundTruthData": {
                "data": str(int(labels[i])),
                "encoding": "CSV",
            },
            "eventMetadata": {
                "eventId": str(i),
            },
            "eventVersion": "0",
        }))

    ts = datetime.now(timezone.utc)
    ground_truth_s3 = cfg.get_s3_path("ground_truth")
    gt_uri = f"{ground_truth_s3}{ts:%Y/%m/%d/%H/%M%S}.jsonl"
    gt_data = "\n".join(gt_records)
    S3Uploader.upload_string_as_file_body(gt_data, gt_uri)

    print(f"  Ground truth uploaded: {gt_uri}")
    print(f"    Records: {len(gt_records)}")
    print(f"    Delay rate: {sum(labels)/len(labels)*100:.1f}%")

    return ground_truth_s3


# ─────────────────────────────────────────────────────────────
# Step 3 — Create monitoring schedule
# ─────────────────────────────────────────────────────────────

def create_monitoring_schedule(
    monitor: ModelQualityMonitor,
    endpoint_name: str,
    ground_truth_s3: str,
) -> str:
    """Set up hourly model quality monitoring schedule."""
    print(f"\n{'='*70}")
    print("STEP 3: CREATING MONITORING SCHEDULE")
    print(f"{'='*70}")

    schedule_name = f"{cfg.MONITOR_SCHEDULE_PREFIX}-{_timestamp()}"

    endpoint_input = EndpointInput(
        endpoint_name=endpoint_name,
        probability_attribute="0",
        probability_threshold_attribute=cfg.PREDICTION_THRESHOLD,
        destination="/opt/ml/processing/input_data",
    )

    baseline_job = monitor.latest_baselining_job

    # Set schedule frequency from config
    freq = cfg.MONITORING_SCHEDULE_FREQUENCY
    if freq == "hourly":
        cron_expr = CronExpressionGenerator.hourly()
    else:
        cron_expr = CronExpressionGenerator.daily()

    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_input,
        output_s3_uri=cfg.get_s3_path("monitor_reports"),
        problem_type=cfg.MONITORING_PROBLEM_TYPE,
        ground_truth_input=ground_truth_s3,
        constraints=baseline_job.suggested_constraints(),
        schedule_cron_expression=cron_expr,
        enable_cloudwatch_metrics=True,
    )

    desc = monitor.describe_schedule()
    print(f"  Schedule:  {schedule_name}")
    print(f"  Status:    {desc['MonitoringScheduleStatus']}")
    print(f"  Frequency: {freq} ({cron_expr})")

    return schedule_name


# ─────────────────────────────────────────────────────────────
# Step 4 — CloudWatch alarm
# ─────────────────────────────────────────────────────────────

def create_cloudwatch_alarm(endpoint_name: str, schedule_name: str) -> None:
    """Create a CloudWatch alarm that fires when F1 drops below threshold."""
    print(f"\n{'='*70}")
    print("STEP 4: CREATING CLOUDWATCH ALARM")
    print(f"{'='*70}")

    cw = boto3.client("cloudwatch", region_name=cfg.REGION)
    alarm_name = f"flight-delay-model-quality-{cfg.CW_ALARM_METRIC}"

    cw.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription=(
            f"Triggers when {cfg.CW_ALARM_METRIC} score drops below "
            f"{cfg.CW_ALARM_THRESHOLD} for the flight delay model"
        ),
        ActionsEnabled=True,
        MetricName=cfg.CW_ALARM_METRIC,
        Namespace="aws/sagemaker/Endpoints/model-metrics",
        Statistic="Average",
        Dimensions=[
            {"Name": "Endpoint", "Value": endpoint_name},
            {"Name": "MonitoringSchedule", "Value": schedule_name},
        ],
        Period=cfg.CW_ALARM_PERIOD_SECONDS,
        EvaluationPeriods=cfg.CW_ALARM_EVALUATION_PERIODS,
        DatapointsToAlarm=1,
        Threshold=cfg.CW_ALARM_THRESHOLD,
        ComparisonOperator="LessThanOrEqualToThreshold",
        TreatMissingData="breaching",
    )

    print(f"  Alarm:     {alarm_name}")
    print(f"  Metric:    {cfg.CW_ALARM_METRIC}")
    print(f"  Threshold: <= {cfg.CW_ALARM_THRESHOLD}")
    print(f"  Period:    {cfg.CW_ALARM_PERIOD_SECONDS}s")


# ─────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────

def cleanup(endpoint_name: str | None = None) -> None:
    """Delete monitoring schedule, endpoint, and CloudWatch alarm."""
    print(f"\n{'='*70}")
    print("CLEANUP")
    print(f"{'='*70}")

    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    cw = boto3.client("cloudwatch", region_name=cfg.REGION)

    # Find and delete monitoring schedules
    schedules = sm.list_monitoring_schedules(MaxResults=100)
    for sched in schedules.get("MonitoringScheduleSummaries", []):
        name = sched["MonitoringScheduleName"]
        if name.startswith(cfg.MONITOR_SCHEDULE_PREFIX):
            print(f"  Deleting schedule: {name}")
            sm.delete_monitoring_schedule(MonitoringScheduleName=name)

    # Find and delete endpoints
    endpoints = sm.list_endpoints(MaxResults=100)
    for ep in endpoints.get("Endpoints", []):
        name = ep["EndpointName"]
        if name.startswith(cfg.ENDPOINT_NAME_PREFIX) or name == endpoint_name:
            print(f"  Deleting endpoint: {name}")
            sm.delete_endpoint(EndpointName=name)

    # Delete CloudWatch alarm
    alarm_name = f"flight-delay-model-quality-{cfg.CW_ALARM_METRIC}"
    try:
        cw.delete_alarms(AlarmNames=[alarm_name])
        print(f"  Deleted alarm: {alarm_name}")
    except Exception:
        pass

    print("  Cleanup complete")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Set up Model Quality Monitoring for flight delay prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_model.py --endpoint-name flight-delay-endpoint-2025-01-15-1200
  python monitor_model.py --endpoint-name <name> --skip-baseline
  python monitor_model.py --cleanup
        """,
    )
    p.add_argument("--endpoint-name", type=str, default=None,
                    help="Name of the deployed real-time endpoint (required unless --cleanup)")
    p.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline generation (reuse existing)")
    p.add_argument("--cleanup", action="store_true",
                    help="Delete endpoint, schedule, and alarm")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("FLIGHT DELAY MODEL QUALITY MONITORING")
    print("=" * 70)
    cfg.print_config()

    if args.cleanup:
        cleanup(args.endpoint_name)
        return 0

    # Require --endpoint-name for monitoring setup
    if not args.endpoint_name:
        print("\nError: --endpoint-name is required for monitoring setup.")
        print("Deploy an endpoint first:")
        print("  python deploy_model.py realtime")
        return 1

    endpoint_name = args.endpoint_name

    # Validate endpoint exists and is InService
    _validate_endpoint(endpoint_name)

    # Create predictor handle for the existing endpoint
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=cfg.sagemaker_session,
        serializer=CSVSerializer(),
    )

    # Step 1: Generate baseline
    if args.skip_baseline:
        print("\nSkipping baseline generation (--skip-baseline)")
        monitor = ModelQualityMonitor(
            role=cfg.ROLE,
            instance_count=cfg.MONITORING_INSTANCE_COUNT,
            instance_type=cfg.MONITORING_INSTANCE_TYPE,
            volume_size_in_gb=cfg.MONITORING_VOLUME_SIZE_GB,
            max_runtime_in_seconds=cfg.MONITORING_MAX_RUNTIME_SECONDS,
            sagemaker_session=cfg.sagemaker_session,
        )
    else:
        monitor = generate_baseline(predictor, endpoint_name)

    # Step 2: Stream production data
    ground_truth_s3 = stream_production_data(predictor, endpoint_name)

    # Step 3: Create monitoring schedule
    schedule_name = create_monitoring_schedule(monitor, endpoint_name, ground_truth_s3)

    # Step 4: CloudWatch alarm
    create_cloudwatch_alarm(endpoint_name, schedule_name)

    # Summary
    print(f"\n{'='*70}")
    print("MODEL MONITORING SETUP COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Endpoint:   {endpoint_name}")
    print(f"  Schedule:   {schedule_name}")
    print(f"  Alarm:      flight-delay-model-quality-{cfg.CW_ALARM_METRIC}")
    print(f"  Baseline:   Test data (November) — {cfg.BASELINE_SAMPLE_SIZE} records")
    print(f"  Production: December data — {cfg.PRODUCTION_SAMPLE_SIZE} records")
    print(f"\n  Monitor will run {cfg.MONITORING_SCHEDULE_FREQUENCY} and compare")
    print(f"  production metrics against baseline constraints.")
    print(f"  CloudWatch alarm triggers if {cfg.CW_ALARM_METRIC} <= {cfg.CW_ALARM_THRESHOLD}.")
    print(f"\n  To clean up resources:")
    print(f"    python monitor_model.py --cleanup --endpoint-name {endpoint_name}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
