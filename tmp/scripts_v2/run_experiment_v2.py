"""
CLI runner to start the SageMaker pipeline (consolidates previous run_experiment/train scripts).

Creates a pipeline execution with hyperparameters provided via CLI or falling
back to defaults from `config.settings_v2.DEFAULT_HYPERPARAMETERS`.

This script is safe to import (no side-effects). Execution only occurs under
`if __name__ == '__main__'`.

Usage examples (run from the repository root):

Dry-run (no network calls will be made):

    python scripts/run_experiment_v2.py --dry-run --MaxDepth 7 --Eta 0.1

Start the pipeline (requires AWS credentials and SageMaker access):

    python scripts/run_experiment_v2.py --MaxDepth 7 --Eta 0.1

Notes:
- Default values are read from `config.settings_v2.DEFAULT_HYPERPARAMETERS`.
- The pipeline name is taken from `config.settings_v2.PIPELINE_NAME`.
- Use `--display-name` to set a friendly name for the pipeline execution.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict

import boto3

from config import settings_v2 as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_pipeline_parameters(overrides: Dict[str, str]) -> list[Dict[str, str]]:
    """Convert overrides dict to SageMaker PipelineParameters list."""
    params = []
    for name, value in overrides.items():
        params.append({"Name": name, "Value": str(value)})
    return params


def pipeline_exists(pipeline_name: str) -> bool:
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    try:
        sm.describe_pipeline(PipelineName=pipeline_name)
        return True
    except sm.exceptions.ResourceNotFoundException:
        return False
    except Exception:
        logger.exception("Error checking pipeline existence")
        raise


def start_pipeline(pipeline_name: str, pipeline_parameters: list[Dict[str, str]], display_name: str | None = None, dry_run: bool = False):
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    if dry_run:
        logger.info("Dry-run mode: would start pipeline '%s' with parameters: %s", pipeline_name, pipeline_parameters)
        return None

    response = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=display_name or f"run-{pipeline_name}",
        PipelineParameters=pipeline_parameters,
    )
    return response.get("PipelineExecutionArn")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start baseline experiment via SageMaker Pipeline")

    # Hyperparameters — defaults pulled from settings_v2.DEFAULT_HYPERPARAMETERS
    defaults = cfg.DEFAULT_HYPERPARAMETERS
    p.add_argument("--MaxDepth", type=int, default=defaults.get("max_depth"), help="XGBoost max_depth")
    p.add_argument("--Eta", type=float, default=defaults.get("eta"), help="XGBoost eta")
    p.add_argument("--NumRound", type=int, default=defaults.get("num_round"), help="Number of boosting rounds")
    p.add_argument("--Subsample", type=float, default=defaults.get("subsample"), help="Subsample")
    p.add_argument("--ColsampleByTree", type=float, default=defaults.get("colsample_bytree"), help="colsample_bytree")
    p.add_argument("--ScalePosWeight", type=float, default=defaults.get("scale_pos_weight"), help="scale_pos_weight")
    p.add_argument("--MinChildWeight", type=int, default=defaults.get("min_child_weight"), help="min_child_weight")

    p.add_argument("--display-name", type=str, default=None, help="Pipeline execution display name")
    p.add_argument("--dry-run", action="store_true", help="Print params but do not start the pipeline")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    pipeline_name = cfg.PIPELINE_NAME

    # Map CLI args to pipeline parameter names expected by the pipeline
    overrides = {
        "MaxDepth": args.MaxDepth,
        "Eta": args.Eta,
        "NumRound": args.NumRound,
        "Subsample": args.Subsample,
        "ColsampleByTree": args.ColsampleByTree,
        "ScalePosWeight": args.ScalePosWeight,
        "MinChildWeight": args.MinChildWeight,
    }

    logger.info("Starting experiment: pipeline=%s, overrides=%s", pipeline_name, overrides)

    # Ensure pipeline exists (idempotent check)
    if not pipeline_exists(pipeline_name):
        logger.error("Pipeline '%s' not found. Please create it first (run pipeline_definition_v2 or pipeline_definition).", pipeline_name)
        return 2

    pipeline_params = build_pipeline_parameters(overrides)

    arn = start_pipeline(pipeline_name, pipeline_params, display_name=args.display_name, dry_run=args.dry_run)
    if arn:
        logger.info("Pipeline execution started. ARN: %s", arn)
    else:
        logger.info("No ARN returned (dry-run or failure).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
