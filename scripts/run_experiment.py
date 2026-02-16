"""
CLI runner to start the SageMaker pipeline with experiment presets.

Supports two experiment modes:
- baseline: less number features , simpler hyperparameters
- improved: 20 features, tuned hyperparameters

Creates a pipeline execution with hyperparameters provided via CLI or falling
back to experiment presets or defaults from `config.settings.DEFAULT_HYPERPARAMETERS`.

This script is safe to import (no side-effects). Execution only occurs under
`if __name__ == '__main__'`.

Usage examples (run from the repository root):

Run baseline experiment:
    python run_experiment.py --experiment baseline

Run improved experiment:
    python run_experiment.py --experiment improved

Custom experiment with specific parameters:
    python run_experiment.py --MaxDepth 7 --Eta 0.1 --UseAdvancedFeatures true

Dry-run (no network calls):
    python run_experiment.py --experiment baseline --dry-run

Notes:
- Experiment presets override default values
- CLI arguments override experiment presets
- The pipeline name is taken from `config.settings.PIPELINE_NAME`
- Use `--display-name` to set a friendly name for the pipeline execution
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict

import boto3

import settings as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ===========================
# EXPERIMENT PRESETS
# ===========================

EXPERIMENTS = {
    'baseline': {
        'description': 'Baseline model, simpler hyperparameters',
        'parameters': {
            'UseAdvancedFeatures': 'false',  # Only 17 features
            'MaxDepth': 1,
            'Eta': 0.9,
            'NumRound': 10,
            'Subsample': 0.4,
            'ColsampleByTree': 0.4,
            'ScalePosWeight': 1.0,
            'MinChildWeight': 20
        }
    },
    'improved': {
        'description': 'Improved model - 20 features, tuned hyperparameters',
        'parameters': {
            'UseAdvancedFeatures': 'true',
            'MaxDepth': 6,
            'Eta': 0.1,
            'NumRound': 100,
            'Subsample': 0.8,
            'ColsampleByTree': 0.8,
            'ScalePosWeight': 5.5,
            'MinChildWeight': 1
        }
    }
}


def build_pipeline_parameters(overrides: Dict[str, str]) -> list[Dict[str, str]]:
    """Convert overrides dict to SageMaker PipelineParameters list."""
    params = []
    for name, value in overrides.items():
        params.append({"Name": name, "Value": str(value)})
    return params


def pipeline_exists(pipeline_name: str) -> bool:
    """Check if SageMaker pipeline exists."""
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    try:
        sm.describe_pipeline(PipelineName=pipeline_name)
        return True
    except Exception as e:
        if 'ResourceNotFound' in str(e):
            return False
        logger.exception("Error checking pipeline existence")
        raise


def start_pipeline(
    pipeline_name: str, 
    pipeline_parameters: list[Dict[str, str]], 
    display_name: str | None = None, 
    dry_run: bool = False
):
    """Start pipeline execution."""
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    
    if dry_run:
        logger.info("="*70)
        logger.info("DRY-RUN MODE")
        logger.info("="*70)
        logger.info("Pipeline: %s", pipeline_name)
        logger.info("Display Name: %s", display_name or f"run-{pipeline_name}")
        logger.info("\nParameters:")
        for param in pipeline_parameters:
            logger.info("  %s = %s", param['Name'], param['Value'])
        logger.info("="*70)
        return None

    response = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=display_name or f"run-{pipeline_name}",
        PipelineParameters=pipeline_parameters,
    )
    return response.get("PipelineExecutionArn")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Start SageMaker Pipeline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiment (17 features, simpler model)
  python run_experiment.py --experiment baseline
  
  # Run improved experiment (20 features, tuned model)
  python run_experiment.py --experiment improved
  
  # Custom parameters
  python run_experiment.py --MaxDepth 7 --Eta 0.1 --UseAdvancedFeatures true
  
  # Dry-run to see parameters without executing
  python run_experiment.py --experiment baseline --dry-run
        """
    )

    # Experiment preset
    p.add_argument(
        "--experiment", 
        type=str, 
        choices=['baseline', 'improved'],
        default=None,
        help="Run predefined experiment: baseline (17 features) or improved (20 features)"
    )

    # Data Source Parameters
    p.add_argument("--RawDataUrl", type=str, default=None, help="S3 path to raw data")
    p.add_argument("--ModelOutputPath", type=str, default=None, help="S3 path for model output")
    
    # Feature Engineering Parameters
    p.add_argument(
        "--UseAdvancedFeatures", 
        type=str, 
        default=None,
        choices=['true', 'false'],
        help="UseAdvancedFeatures"
    )
    
    # Infrastructure Parameters
    p.add_argument("--InstanceType", type=str, default=None, help="Training instance type")
    p.add_argument("--InstanceCount", type=int, default=None, help="Training instance count")
    p.add_argument("--ApprovalStatus", type=str, default=None, help="Model approval status")

    # Hyperparameters
    defaults = cfg.DEFAULT_HYPERPARAMETERS
    p.add_argument("--Objective", type=str, default=None, help="XGBoost objective")
    p.add_argument("--EvalMetric", type=str, default=None, help="XGBoost eval_metric")
    p.add_argument("--MaxDepth", type=int, default=None, help="XGBoost max_depth")
    p.add_argument("--Eta", type=float, default=None, help="XGBoost learning rate (eta)")
    p.add_argument("--NumRound", type=int, default=None, help="Number of boosting rounds")
    p.add_argument("--Subsample", type=float, default=None, help="Row subsample ratio")
    p.add_argument("--ColsampleByTree", type=float, default=None, help="Column subsample ratio")
    p.add_argument("--ScalePosWeight", type=float, default=None, help="Class balancing weight")
    p.add_argument("--MinChildWeight", type=int, default=None, help="Minimum child weight")

    # Execution options
    p.add_argument("--display-name", type=str, default=None, help="Pipeline execution display name")
    p.add_argument("--dry-run", action="store_true", help="Print params but do not start the pipeline")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main execution function."""
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    pipeline_name = cfg.PIPELINE_NAME

    # Start with experiment preset if specified
    if args.experiment:
        experiment_config = EXPERIMENTS[args.experiment]
        logger.info("="*70)
        logger.info("EXPERIMENT: %s", args.experiment.upper())
        logger.info("="*70)
        logger.info("Description: %s", experiment_config['description'])
        logger.info("="*70)
        overrides = experiment_config['parameters'].copy()
    else:
        overrides = {}

    # Override with CLI arguments (CLI takes precedence)
    cli_overrides = {
        "RawDataUrl": args.RawDataUrl,
        "ModelOutputPath": args.ModelOutputPath,
        "UseAdvancedFeatures": args.UseAdvancedFeatures,
        "InstanceType": args.InstanceType,
        "InstanceCount": args.InstanceCount,
        "ApprovalStatus": args.ApprovalStatus,
        "Objective": args.Objective,
        "EvalMetric": args.EvalMetric,
        "MaxDepth": args.MaxDepth,
        "Eta": args.Eta,
        "NumRound": args.NumRound,
        "Subsample": args.Subsample,
        "ColsampleByTree": args.ColsampleByTree,
        "ScalePosWeight": args.ScalePosWeight,
        "MinChildWeight": args.MinChildWeight,
    }
    
    # Filter out None values and merge with experiment preset
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    overrides.update(cli_overrides)

    logger.info("\nStarting pipeline execution:")
    logger.info("  Pipeline: %s", pipeline_name)
    logger.info("  Parameters: %s", overrides)

    # Ensure pipeline exists
    if not pipeline_exists(pipeline_name):
        logger.error(
            "Pipeline '%s' not found. Please create it first:\n"
            "  python pipeline_definition.py",
            pipeline_name
        )
        return 2

    # Generate display name
    if args.display_name:
        display_name = args.display_name
    elif args.experiment:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        display_name = f"{args.experiment}-{timestamp}"
    else:
        display_name = None

    # Build parameters and start pipeline
    pipeline_params = build_pipeline_parameters(overrides)
    arn = start_pipeline(pipeline_name, pipeline_params, display_name=display_name, dry_run=args.dry_run)
    
    if arn:
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE EXECUTION STARTED")
        logger.info("="*70)
        logger.info("ARN: %s", arn)
        logger.info("Display Name: %s", display_name)
        logger.info("\n⏱️  Expected Duration:")
        logger.info("  Feature Engineering: ~2-3 minutes")
        logger.info("  Training: ~8-12 minutes")
        logger.info("  Evaluation: ~2-3 minutes")
        logger.info("  Total: ~12-18 minutes")
        logger.info("\n📊 Monitor in SageMaker Studio:")
        logger.info("  Pipelines → %s → %s", pipeline_name, display_name)
        logger.info("="*70)
    elif not args.dry_run:
        logger.error("Pipeline execution failed (no ARN returned)")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())