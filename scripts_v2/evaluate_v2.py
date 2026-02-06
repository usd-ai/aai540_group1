"""
Evaluation utilities (v2) for the Flight Delay project.

Provides `compute_metrics` for offline arrays and a small CLI to evaluate a
predictions CSV (local file). The module is safe to import and avoids network
access unless explicitly implemented by the caller.
"""
from __future__ import annotations

from typing import Dict, Iterable

import logging

from config import settings_v2 as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_metrics(y_true: Iterable[int], y_proba: Iterable[float], threshold: float | None = None) -> Dict[str, float]:
    """Compute evaluation metrics given true labels and predicted probabilities.

    Returns dict with keys: auc, f1, precision, recall, accuracy, threshold_used
    """
    try:
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    except Exception as exc:  # pragma: no cover - sklearn may not be available in some test envs
        logger.error("scikit-learn is required for compute_metrics: %s", exc)
        raise

    y_true_list = list(y_true)
    y_proba_list = list(y_proba)
    if threshold is None:
        threshold = getattr(cfg, "PREDICTION_THRESHOLD", 0.5)

    # Ensure lengths match
    if len(y_true_list) != len(y_proba_list):
        raise ValueError("y_true and y_proba must have the same length")

    # Compute binary predictions
    y_pred = [1 if p >= threshold else 0 for p in y_proba_list]

    metrics = {
        "auc": float(roc_auc_score(y_true_list, y_proba_list)) if len(set(y_true_list)) > 1 else 0.0,
        "f1": float(f1_score(y_true_list, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_list, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_list, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_list, y_pred)),
        "threshold_used": float(threshold),
    }
    return metrics


def evaluate_local_csv(path: str, label_col: str = "label", prob_col: str = "probability", threshold: float | None = None) -> Dict[str, float]:
    """Load a local CSV and compute metrics.

    The function intentionally rejects s3:// paths to keep evaluation offline
    and predictable in notebooks; download S3 objects separately if needed.
    """
    if path.startswith("s3://"):
        raise ValueError("s3:// paths are not supported by evaluate_local_csv. Download the file locally first.")

    try:
        import pandas as pd
    except Exception as exc:
        logger.error("pandas is required to read CSV files: %s", exc)
        raise

    df = pd.read_csv(path)
    if label_col not in df.columns or prob_col not in df.columns:
        raise KeyError(f"Columns {label_col} and {prob_col} must be present in CSV")

    y_true = df[label_col].astype(int).tolist()
    y_proba = df[prob_col].astype(float).tolist()
    return compute_metrics(y_true, y_proba, threshold=threshold)


def parse_args(argv: list[str]):
    import argparse

    p = argparse.ArgumentParser(description="Evaluate predictions CSV (local)")
    p.add_argument("--predictions", required=True, help="Local CSV file with predictions")
    p.add_argument("--label-col", default="label")
    p.add_argument("--prob-col", default="probability")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output-json", default=None, help="Optional path to write metrics JSON to")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import json
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    metrics = evaluate_local_csv(args.predictions, label_col=args.label_col, prob_col=args.prob_col, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(metrics, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
