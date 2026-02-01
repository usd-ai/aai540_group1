# aai540_group1

# Flight Delay Prediction - MLOps Pipeline

End-to-end ML pipeline for predicting flight delays using AWS SageMaker, built for AAI-540 MLOps course.

## Overview

Automated binary classification system predicting flight delays (>15 minutes) for airline operational planning. Implements production-grade MLOps practices including automated training, evaluation, conditional model registration, and deployment pipelines.

**Tech Stack:** AWS SageMaker, XGBoost, Python, S3, Model Registry, Batch Transform

## Project Structure
```
├── scripts/
│   └── evaluate.py              # Model evaluation script for SageMaker Processing
├── pipeline_definition.py       # Creates/updates SageMaker Pipeline
├── run_experiment.py            # Executes baseline training experiment
├── requirements.txt             # Python dependencies
└── README.md
```

## Pipeline Architecture
```
Training → Evaluation → Condition (F1 ≥ 0.70?) → Model Registration
   ↓           ↓              ↓                        ↓
XGBoost    Test Set      Pass/Fail Gate      SageMaker Model Registry
10-15min    2-5min        Automated           (flight-delay-models)
```

## Prerequisites

- AWS SageMaker access with execution role
- Model Package Group: `flight-delay-models`
- Training data in S3:
  - `s3://BUCKET/PREFIX/data/train/train.csv`
  - `s3://PREFIX/data/validation/val.csv`
  - `s3://PREFIX/data/test/test.csv`

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create pipeline (one-time)
python pipeline_definition.py
```

## Usage
```bash
# Run baseline experiment
python run_experiment.py
```

**Baseline Hyperparameters:**
- Max Depth: 6
- Learning Rate (Eta): 0.1
- Boosting Rounds: 100
- Subsample: 0.8
- Column Subsample: 0.8

## Monitoring

View pipeline execution in **SageMaker Studio**:
1. Navigate to **Pipelines** → `FlightDelayTrainingPipeline`
2. Find execution by timestamp
3. Monitor real-time step execution

## Success Criteria

**Model Registration Requirements:**
- F1-score ≥ 0.70 on test set
- Automatic registration to Model Registry
- Status: `PendingManualApproval`

## Pipeline Steps

| Step | Component | Duration | Output |
|------|-----------|----------|--------|
| 1. Training | SageMaker Training Job (XGBoost) | 10-15 min | model.tar.gz |
| 2. Evaluation | SageMaker Processing Job | 2-5 min | metrics.json |
| 3. Condition Check | Pipeline Logic | <1 min | Pass/Fail |
| 4. Registration | Model Registry | 1-2 min | Model Package |

## Dataset

**Source:** 2015 Flight Delays and Cancellations (U.S. DOT)
- **Training:** 4.3M samples
- **Validation:** 483K samples  
- **Test:** 462K samples
- **Features:** 20 engineered features (temporal, historical, route-based)
- **Target:** Binary delay indicator (>15 minutes)

## Model Registry

Approved models stored in: `flight-delay-models`

Access via SageMaker Studio: **Model Registry** → `flight-delay-models`


## Configuration

Update `pipeline_definition.py` for:
- S3 bucket/prefix paths
- Instance types
- F1 threshold
- Hyperparameter ranges


## License

Academic project - AAI-540 MLOps Course