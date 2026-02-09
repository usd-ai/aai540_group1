"""
Deploy Registered Model (v3)

Supports two deployment modes:
  batch     — Batch Transform for offline predictions
  realtime  — Real-time endpoint with data capture (for model monitoring)

Uses centralized configuration from settings.py

Execution order:
  Step 0-3:  Pipeline (FE -> Train -> Eval -> Register)
  Step 4:    deploy_model.py batch|realtime
  Step 5:    monitor_model.py --endpoint-name <name>

Usage:
  python deploy_model.py batch                          # batch transform
  python deploy_model.py batch --input-data s3://...    # custom input
  python deploy_model.py realtime                       # deploy endpoint
  python deploy_model.py realtime --endpoint-name my-ep # custom name
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import boto3
import pandas as pd
from sagemaker.model_monitor import DataCaptureConfig

import settings as cfg


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _timestamp() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")


def _print_approved_model(model_info: dict) -> None:
    print(f"  ARN:     {model_info['ModelPackageArn']}")
    print(f"  Status:  {model_info['ModelApprovalStatus']}")
    print(f"  Created: {model_info['CreationTime']}")


# ─────────────────────────────────────────────────────────────
# Batch Transform
# ─────────────────────────────────────────────────────────────

def run_batch(args: argparse.Namespace) -> int:
    """Run batch transform on the latest approved model."""
    print("\n" + "=" * 70)
    print("DEPLOYING MODEL TO BATCH TRANSFORM")
    print("=" * 70)

    # Find approved model
    print("\nFinding latest approved model...")
    model_info = cfg.find_approved_model()
    model_package_arn = model_info["ModelPackageArn"]
    _print_approved_model(model_info)

    # Create model
    print("\nCreating SageMaker Model from registered package...")
    model = cfg.create_model_from_registry(model_package_arn)

    # Create transformer
    print("\nCreating Batch Transformer...")
    transformer = model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=cfg.get_s3_path("predictions"),
        assemble_with="Line",
        accept="text/csv",
        strategy="SingleRecord",
    )

    print(f"  Instance: {args.instance_type}")
    print(f"  Output:   {cfg.get_s3_path('predictions')}")

    # Run transform
    print("\n" + "=" * 70)
    print("RUNNING BATCH PREDICTION")
    print("=" * 70)

    job_name = f'flight-delay-predictions-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    transformer.transform(
        data=args.input_data,
        content_type=args.content_type,
        split_type="Line",
        job_name=job_name,
        wait=True,
        logs=True,
    )

    print("\nBatch transform complete!")

    # Check output
    print("\nChecking predictions...")
    output_path = transformer.output_path
    print(f"  Predictions saved to: {output_path}")

    s3 = boto3.client("s3", region_name=cfg.REGION)
    response = s3.list_objects_v2(
        Bucket=cfg.BUCKET,
        Prefix=f"{cfg.PREFIX}/predictions/{job_name}/",
    )

    if "Contents" in response:
        print(f"\n  Output files:")
        for obj in response["Contents"]:
            size_mb = obj["Size"] / (1024 * 1024)
            print(f"    {obj['Key']} ({size_mb:.2f} MB)")

    # Download and display sample predictions
    output_file = None
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".out"):
            output_file = obj["Key"]
            break

    if output_file:
        local_file = "sample_predictions.csv"
        s3.download_file(cfg.BUCKET, output_file, local_file)

        predictions = pd.read_csv(local_file, header=None, nrows=10)
        print(f"\n  Sample predictions (first 10):")
        print(predictions)
        print(f"\n  Predictions: 0 = on-time, 1 = delayed")

        all_preds = pd.read_csv(local_file, header=None)
        print(f"\n  Prediction Summary:")
        print(f"    Total predictions: {len(all_preds):,}")
        print(f"    Predicted on-time (0): {(all_preds[0] < 0.5).sum():,} ({(all_preds[0] < 0.5).sum()/len(all_preds)*100:.1f}%)")
        print(f"    Predicted delayed (1): {(all_preds[0] >= 0.5).sum():,} ({(all_preds[0] >= 0.5).sum()/len(all_preds)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("BATCH DEPLOYMENT COMPLETE")
    print("=" * 70)
    return 0


# ─────────────────────────────────────────────────────────────
# Real-time Endpoint (with data capture for monitoring)
# ─────────────────────────────────────────────────────────────

def run_realtime(args: argparse.Namespace) -> int:
    """Deploy the approved model as a real-time endpoint with data capture."""
    print("\n" + "=" * 70)
    print("DEPLOYING MODEL TO REAL-TIME ENDPOINT")
    print("=" * 70)

    # Find approved model
    print("\nFinding latest approved model...")
    model_info = cfg.find_approved_model()
    model_package_arn = model_info["ModelPackageArn"]
    _print_approved_model(model_info)

    # Create model
    print("\nCreating SageMaker Model from registered package...")
    model = cfg.create_model_from_registry(model_package_arn)

    # Endpoint name
    endpoint_name = args.endpoint_name or f"{cfg.ENDPOINT_NAME_PREFIX}-{_timestamp()}"

    print(f"\n  Endpoint: {endpoint_name}")
    print(f"  Instance: {cfg.TRAINING_INSTANCE_TYPE}")
    print(f"  Data capture: {cfg.DATA_CAPTURE_SAMPLING_PERCENTAGE}%")
    print(f"  Capture destination: {cfg.get_s3_path('data_capture')}")

    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=cfg.DATA_CAPTURE_SAMPLING_PERCENTAGE,
        destination_s3_uri=cfg.get_s3_path("data_capture"),
    )

    model.deploy(
        initial_instance_count=1,
        instance_type=cfg.TRAINING_INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        data_capture_config=data_capture_config,
    )

    print(f"\n  Endpoint deployed successfully: {endpoint_name}")

    print("\n" + "=" * 70)
    print("REAL-TIME DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Endpoint name: {endpoint_name}")
    print(f"\n  Next step — set up monitoring:")
    print(f"    python monitor_model.py --endpoint-name {endpoint_name}")
    print("=" * 70 + "\n")
    return 0


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy approved model (batch transform or real-time endpoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_model.py batch                          # batch transform
  python deploy_model.py batch --input-data s3://...    # custom input
  python deploy_model.py realtime                       # deploy endpoint
  python deploy_model.py realtime --endpoint-name my-ep # custom name
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── batch subcommand ──
    batch_p = subparsers.add_parser("batch", help="Run batch transform")
    batch_p.add_argument(
        "--input-data",
        type=str,
        default=f"s3://{cfg.BUCKET}/{cfg.PREFIX}/data/inference/test_inference_small.csv",
        help="S3 URI for input data",
    )
    batch_p.add_argument(
        "--content-type",
        type=str,
        default="text/csv",
        help="MIME type of input data",
    )
    batch_p.add_argument(
        "--instance-type",
        type=str,
        default=cfg.TRANSFORM_INSTANCE_TYPE,
        help="Instance type for transformer",
    )
    batch_p.add_argument(
        "--instance-count",
        type=int,
        default=cfg.TRANSFORM_INSTANCE_COUNT,
        help="Instance count",
    )

    # ── realtime subcommand ──
    rt_p = subparsers.add_parser(
        "realtime",
        help="Deploy real-time endpoint with data capture for monitoring",
    )
    rt_p.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help=f"Custom endpoint name (default: {cfg.ENDPOINT_NAME_PREFIX}-<timestamp>)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "batch":
        return run_batch(args)
    elif args.command == "realtime":
        return run_realtime(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
