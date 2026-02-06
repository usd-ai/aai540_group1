"""
Idempotent deployment helpers for the Flight Delay project (v2).

This module provides utilities to create a SageMaker `Model`, `EndpointConfig`,
and `Endpoint` in an idempotent way. By default functions run in `dry_run`
mode and avoid network calls. Pass `--apply` to the CLI to perform real
AWS operations.

Usage examples (dry-run):

    python scripts/deploy_model_v2.py --model-name my-model --endpoint-name my-endpoint --model-s3-uri s3://bucket/prefix/models/model.tar.gz

To apply changes (will call SageMaker APIs):

    python scripts/deploy_model_v2.py --apply --model-name my-model --endpoint-name my-endpoint --model-s3-uri s3://...

The module is safe to import.
"""
from __future__ import annotations

import logging
import sys
from typing import Dict

import boto3

from config import settings_v2 as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_primary_container(image_uri: str, model_s3_uri: str) -> Dict[str, str]:
    return {"Image": image_uri, "ModelDataUrl": model_s3_uri}


def create_model(model_name: str, primary_container: Dict[str, str], role: str = cfg.ROLE, dry_run: bool = True) -> str:
    """Create a SageMaker Model if it doesn't exist. Returns model name or dry-run message."""
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    if dry_run:
        logger.info("Dry-run: would create Model '%s' with container %s", model_name, primary_container)
        return f"dry:{model_name}"

    try:
        sm.describe_model(ModelName=model_name)
        logger.info("Model '%s' already exists.", model_name)
        return model_name
    except sm.exceptions.ClientError:
        pass

    sm.create_model(ModelName=model_name, PrimaryContainer=primary_container, ExecutionRoleArn=role)
    logger.info("Created Model '%s'", model_name)
    return model_name


def create_endpoint_config(config_name: str, model_name: str, instance_type: str = cfg.INFERENCE_INSTANCE_TYPES[0], dry_run: bool = True) -> str:
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    if dry_run:
        logger.info("Dry-run: would create EndpointConfig '%s' for model '%s' on %s", config_name, model_name, instance_type)
        return f"dry:{config_name}"

    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        logger.info("EndpointConfig '%s' already exists.", config_name)
        return config_name
    except sm.exceptions.ClientError:
        pass

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": cfg.TRANSFORM_INSTANCE_COUNT,
                "InstanceType": instance_type,
            }
        ],
    )
    logger.info("Created EndpointConfig '%s'", config_name)
    return config_name


def create_or_update_endpoint(endpoint_name: str, config_name: str, dry_run: bool = True) -> str:
    sm = boto3.client("sagemaker", region_name=cfg.REGION)
    if dry_run:
        logger.info("Dry-run: would create or update Endpoint '%s' using config '%s'", endpoint_name, config_name)
        return f"dry:{endpoint_name}"

    try:
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp.get("EndpointStatus")
        logger.info("Endpoint '%s' exists with status %s — updating to config %s", endpoint_name, status, config_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        return endpoint_name
    except sm.exceptions.ClientError:
        # Assume not found
        logger.info("Creating Endpoint '%s' with config '%s'", endpoint_name, config_name)
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        return endpoint_name


def deploy_model_flow(model_name: str, model_s3_uri: str, endpoint_name: str, instance_type: str | None = None, apply: bool = False) -> Dict[str, str]:
    """High-level flow to create a model, endpoint config, and endpoint.

    Returns a small dict describing the actions taken (or that would be taken).
    """
    dry_run = not apply
    image_uri = cfg.xgboost_image_uri()
    primary_container = build_primary_container(image_uri, model_s3_uri)

    model_result = create_model(model_name, primary_container, dry_run=dry_run)
    config_name = f"{model_name}-config"
    inst_type = instance_type or cfg.INFERENCE_INSTANCE_TYPES[0]
    config_result = create_endpoint_config(config_name, model_name, instance_type=inst_type, dry_run=dry_run)
    endpoint_result = create_or_update_endpoint(endpoint_name, config_name, dry_run=dry_run)

    return {"model": model_result, "endpoint_config": config_result, "endpoint": endpoint_result}


def parse_args(argv: list[str]) -> any:
    import argparse

    p = argparse.ArgumentParser(description="Deploy model to SageMaker (dry-run default)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-s3-uri", required=True)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--instance-type", default=cfg.INFERENCE_INSTANCE_TYPES[0])
    p.add_argument("--apply", action="store_true", help="Apply changes (make network calls)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)
    result = deploy_model_flow(args.model_name, args.model_s3_uri, args.endpoint_name, instance_type=args.instance_type, apply=args.apply)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
