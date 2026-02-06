"""
Lightweight pipeline definition utilities for Flight Delay project (v2).

This module generates a minimal, inspectable pipeline definition derived
from `config.settings_v2` and exposes an idempotent `upsert_pipeline`
helper. By default functions avoid making network calls — use
`upsert_pipeline(dry_run=False)` to perform AWS operations.

The module is safe to import (no side-effects).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import boto3

from config import settings_v2 as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_pipeline_summary() -> Dict[str, Any]:
    """Return a small summary/dictionary describing the pipeline using config values.

    This function performs no network calls and is suitable for quick inspection
    and unit testing.
    """
    summary = {
        "PipelineName": cfg.PIPELINE_NAME,
        "ModelPackageGroup": cfg.MODEL_PACKAGE_GROUP,
        "Bucket": cfg.BUCKET,
        "Prefix": cfg.PREFIX,
        "S3PathsSample": {
            "training_input": cfg.get_s3_path("training_input"),
            "training_output": cfg.get_s3_path("training_output"),
        },
        "DefaultHyperparameters": cfg.DEFAULT_HYPERPARAMETERS,
        "F1Threshold": getattr(cfg, "F1_THRESHOLD", None),
    }
    return summary


def build_pipeline_definition() -> Dict[str, Any]:
    """Construct a minimal, serializable pipeline-definition placeholder.

    The returned structure is intentionally lightweight so it can be inspected
    in environments without the full SageMaker SDK or network access.
    """
    summary = get_pipeline_summary()
    # A tiny placeholder for pipeline structure — real pipeline_definition_v2
    # can expand this with ProcessingStep/TrainingStep/ConditionStep objects.
    pipeline_def = {
        "PipelineName": summary["PipelineName"],
        "Version": "v2-preview",
        "Description": "Auto-generated pipeline definition (v2).",
        "Resources": {
            "ModelPackageGroup": summary["ModelPackageGroup"],
            "S3Bucket": summary["Bucket"],
        },
        "Steps": [
            {"Name": "LoadData", "Type": "Processing", "Output": summary["S3PathsSample"]["training_input"]},
            {"Name": "Train", "Type": "Training", "Output": summary["S3PathsSample"]["training_output"]},
            {"Name": "Evaluate", "Type": "Processing", "Metric": "F1", "Threshold": summary["F1Threshold"]},
        ],
    }
    return pipeline_def


def upsert_pipeline(dry_run: bool = True) -> Dict[str, Any] | str:
    """Idempotently create or update the pipeline in SageMaker.

    - If `dry_run` is True (default), returns a serializable summary and
      the pipeline definition as JSON without performing network calls.
    - If `dry_run` is False, performs SageMaker calls to create or update the
      pipeline. Network errors are propagated.
    """
    pipeline_name = cfg.PIPELINE_NAME
    pipeline_def = build_pipeline_definition()

    if dry_run:
        logger.info("Dry-run: returning pipeline summary and definition for '%s'", pipeline_name)
        return {"summary": get_pipeline_summary(), "definition": pipeline_def}

    # Live mode: call SageMaker to upsert the pipeline. Use boto3 client.
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    try:
        # Try to describe existing pipeline
        sm.describe_pipeline(PipelineName=pipeline_name)
        logger.info("Pipeline '%s' exists — updating." , pipeline_name)
        # Update pipeline (SageMaker has update_pipeline API)
        sm.update_pipeline(PipelineName=pipeline_name, PipelineDefinition=json.dumps(pipeline_def))
        return f"updated:{pipeline_name}"
    except sm.exceptions.ResourceNotFoundException:
        logger.info("Pipeline '%s' not found — creating.", pipeline_name)
        sm.create_pipeline(PipelineName=pipeline_name, PipelineDefinition=json.dumps(pipeline_def), PipelineDisplayName=pipeline_name)
        return f"created:{pipeline_name}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upsert pipeline (dry-run default)")
    parser.add_argument("--apply", action="store_true", help="Apply changes (make network calls)")
    args = parser.parse_args()
    result = upsert_pipeline(dry_run=not args.apply)
    print(result)
