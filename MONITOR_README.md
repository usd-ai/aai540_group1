# Model Quality Monitoring — monitor_model.py

## Prerequisites

Before running this script, the following must be completed in order:

1. `prepare-data-for-feature-engineering.py` — raw data uploaded to S3
2. `pipeline_definition.py` — pipeline created in SageMaker
3. `run_experiment.py` — pipeline executed (FE → Train → Eval → Register)
4. Approve the model in SageMaker Studio (Model Registry → flight-delay-models → Approve)
5. `deploy_model.py realtime` — real-time endpoint deployed with data capture enabled

## Usage

### Full setup (baseline + production streaming + schedule + alarm)

```bash
python monitor_model.py --endpoint-name <your-endpoint-name>
```

Example:

```bash
python monitor_model.py --endpoint-name flight-delay-endpoint-2026-02-09-2054
```

### Skip baseline (reuse a previously generated baseline)

Only use this if baseline has already completed successfully.

```bash
python monitor_model.py --endpoint-name <your-endpoint-name> --skip-baseline
```

### Cleanup (delete endpoint, schedule, and alarm)

```bash
python monitor_model.py --cleanup --endpoint-name <your-endpoint-name>
```

## What It Does

| Step | Description |
|------|-------------|
| 1. Baseline | Sends 500 test records (November) through the endpoint, collects predictions, and runs a SageMaker Model Quality baseline job to establish reference metrics and constraints |
| 2. Production Streaming | Streams 500 production records (December) through the endpoint and uploads ground truth labels to S3 |
| 3. Monitoring Schedule | Creates a daily monitoring schedule that compares production predictions against baseline constraints to detect drift |
| 4. CloudWatch Alarm | Sets up an alarm that triggers if F1 score drops to 0.2 or below |

## Configuration

All settings are defined in `settings.py`:

| Setting | Value |
|---------|-------|
| Baseline sample size | 500 records |
| Production sample size | 500 records |
| Schedule frequency | Daily (midnight UTC) |
| CloudWatch alarm metric | F1 |
| CloudWatch alarm threshold | <= 0.2 |
| Monitoring instance | ml.m5.xlarge |
| Problem type | BinaryClassification |

## S3 Paths

| Data | Location |
|------|----------|
| Baseline predictions | `s3://<bucket>/aai540-group1/monitoring/baselining/data/` |
| Baseline results | `s3://<bucket>/aai540-group1/monitoring/baselining/results/` |
| Data capture | `s3://<bucket>/aai540-group1/monitoring/datacapture/` |
| Ground truth | `s3://<bucket>/aai540-group1/monitoring/ground_truth/` |
| Monitoring reports | `s3://<bucket>/aai540-group1/monitoring/reports/` |

## Monitoring Reports

Reports are generated automatically by the daily schedule. After each run, results appear in the `monitoring/reports/` S3 path. These include:

- **constraint_violations.json** — which baseline constraints were violated
- **statistics.json** — current model quality metrics (F1, precision, recall, AUC, accuracy)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No approved models found` | Approve the model in SageMaker Studio Model Registry first |
| `Endpoint not found` | Run `python deploy_model.py realtime` first |
| `Endpoint status is not InService` | Wait for endpoint deployment to finish (5-10 min) |
| `No CSV files found` | Ensure the pipeline ran successfully and produced processed data |
| Double slash `//` in S3 path | Already fixed — ensure you have the latest code |
| `'NoneType' has no attribute 'suggested_constraints'` | Do not use `--skip-baseline` unless a baseline has previously completed successfully |
