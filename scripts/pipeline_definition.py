"""
SageMaker Pipeline Definition for Flight Delay Prediction (v3)
End-to-end pipeline: Feature Engineering → Training → Evaluation → Registration
Uses centralized configuration from settings
"""
import os
import boto3
import sagemaker

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile

import settings as cfg

# ===========================
# UPLOAD SCRIPTS TO S3
# ===========================
def upload_scripts():
    """Upload feature engineering and evaluation scripts to S3"""
    print("📤 Uploading scripts to S3...")
    
    s3 = boto3.client('s3')
    
    scripts = {
        'feature_engineering.py': f'{cfg.PREFIX}/scripts/feature_engineering.py',
        'evaluate.py': f'{cfg.PREFIX}/scripts/evaluate.py'
    }
    
    for local_script, s3_key in scripts.items():
        if os.path.exists(local_script):
            s3.upload_file(local_script, cfg.BUCKET, s3_key)
            print(f"✅ Uploaded {local_script} to s3://{cfg.BUCKET}/{s3_key}")
        else:
            print(f"⚠️  Warning: {local_script} not found")

# ===========================
# DEFINE PIPELINE
# ===========================
def create_pipeline():
    """Create SageMaker Pipeline with Feature Engineering"""
    
    # Setup
    role = cfg.ROLE
    session = cfg.sagemaker_session
    region = cfg.REGION
    
    print(f"\n{'='*70}")
    print(f"CREATING SAGEMAKER PIPELINE: {cfg.PIPELINE_NAME}")
    print(f"{'='*70}")
    print(f"Region: {region}")
    print(f"Bucket: {cfg.BUCKET}")
    print(f"Prefix: {cfg.PREFIX}")
    
    # ===========================
    # PIPELINE PARAMETERS
    # ===========================
    
    # 1. Data Source Parameters
    raw_data_url = ParameterString(
        name="RawDataUrl", 
        default_value=f's3://{cfg.BUCKET}/{cfg.PREFIX}/pipeline-data/raw/'
    )
    model_output_path = ParameterString(
        name="ModelOutputPath", 
        default_value=cfg.get_s3_path("models")
    )
    
    # 2. Feature Engineering Parameters
    include_volume_features = ParameterString(
        name='IncludeVolumeFeatures', 
        default_value='true'
    )
    
    # 3. Infrastructure Parameters
    instance_type = ParameterString(
        name="InstanceType", 
        default_value=cfg.TRAINING_INSTANCE_TYPE
    )
    instance_count = ParameterInteger(
        name="InstanceCount", 
        default_value=cfg.TRAINING_INSTANCE_COUNT
    )
    approval_status = ParameterString(
        name="ApprovalStatus", 
        default_value=cfg.MODEL_APPROVAL_STATUS
    )
    
    # 4. Model Hyperparameters
    objective = ParameterString(name="Objective", default_value="binary:logistic")
    eval_metric = ParameterString(name="EvalMetric", default_value="auc")
    max_depth = ParameterInteger(name='MaxDepth', default_value=cfg.DEFAULT_HYPERPARAMETERS['max_depth'])
    eta = ParameterFloat(name='Eta', default_value=cfg.DEFAULT_HYPERPARAMETERS['eta'])
    num_round = ParameterInteger(name='NumRound', default_value=cfg.DEFAULT_HYPERPARAMETERS['num_round'])
    subsample = ParameterFloat(name='Subsample', default_value=cfg.DEFAULT_HYPERPARAMETERS['subsample'])
    colsample_bytree = ParameterFloat(name='ColsampleByTree', default_value=cfg.DEFAULT_HYPERPARAMETERS['colsample_bytree'])
    scale_pos_weight = ParameterFloat(name='ScalePosWeight', default_value=cfg.DEFAULT_HYPERPARAMETERS['scale_pos_weight'])
    min_child_weight = ParameterInteger(name='MinChildWeight', default_value=cfg.DEFAULT_HYPERPARAMETERS['min_child_weight'])
    
    print(f"\n📋 Pipeline Parameters defined:")
    print(f"   - Feature Engineering (IncludeVolumeFeatures)")
    print(f"   - Data Sources (Raw Data, Model Output)")
    print(f"   - Infrastructure (Instance Type/Count)")
    print(f"   - Hyperparameters (MaxDepth, Eta, etc.)")
    
    # ===========================
    # STEP 0: FEATURE ENGINEERING
    # ===========================
    print(f"\n🔧 Defining Feature Engineering Step...")
    
    feature_script_path = f's3://{cfg.BUCKET}/{cfg.PREFIX}/scripts/feature_engineering.py'
    
    feature_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve('sklearn', region, version='1.0-1'),
        instance_type=cfg.PROCESSING_INSTANCE_TYPE,
        instance_count=cfg.PROCESSING_INSTANCE_COUNT,
        base_job_name='flight-delay-features',
        command=['python3']
    )
    
    feature_engineering_step = ProcessingStep(
        name='FeatureEngineering',
        processor=feature_processor,
        code=feature_script_path,
        job_arguments=[
            '--include-volume-features', include_volume_features
        ],
        inputs=[
            ProcessingInput(
                source=raw_data_url,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train',
                source='/opt/ml/processing/output/train',
                destination=f's3://{cfg.BUCKET}/{cfg.PREFIX}/processed-data/train/'
            ),
            ProcessingOutput(
                output_name='validation',
                source='/opt/ml/processing/output/validation',
                destination=f's3://{cfg.BUCKET}/{cfg.PREFIX}/processed-data/validation/'
            ),
            ProcessingOutput(
                output_name='test',
                source='/opt/ml/processing/output/test',
                destination=f's3://{cfg.BUCKET}/{cfg.PREFIX}/processed-data/test/'
            )
        ]
    )
    
    print("   ✅ Feature engineering step defined")
    
    # ===========================
    # STEP 1: TRAINING
    # ===========================
    print(f"\n🔧 Defining Training Step...")
    
    xgboost_container = cfg.xgboost_image_uri()
    
    xgb_estimator = Estimator(
        image_uri=xgboost_container,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=model_output_path,
        sagemaker_session=session,
        base_job_name='flight-delay-training'
    )
    
    xgb_estimator.set_hyperparameters(
        objective=objective,
        max_depth=max_depth,
        eta=eta,
        num_round=num_round,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,        
        eval_metric=eval_metric
    )
    
    training_step = TrainingStep(
        name='TrainFlightDelayModel',
        estimator=xgb_estimator,
        inputs={
            'train': TrainingInput(
                s3_data=feature_engineering_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=feature_engineering_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )
    
    print("   ✅ Training step defined")
    
    # ===========================
    # STEP 2: EVALUATION
    # ===========================
    print(f"\n🔧 Defining Evaluation Step...")
    
    eval_script_path = f's3://{cfg.BUCKET}/{cfg.PREFIX}/scripts/evaluate.py'
    
    eval_processor = ScriptProcessor(
        role=role,
        image_uri=cfg.xgboost_image_uri(),
        instance_type=cfg.PROCESSING_INSTANCE_TYPE,
        instance_count=cfg.PROCESSING_INSTANCE_COUNT,
        base_job_name='flight-delay-evaluation',
        command=['python3']
    )    
    
    # Define property file for condition check
    evaluation_report = PropertyFile(
        name='evaluation',
        output_name='evaluation',
        path='evaluation.json'
    )
    
    evaluation_step = ProcessingStep(
        name='EvaluateModel',
        processor=eval_processor,
        code=eval_script_path,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination=cfg.PROCESSING_MODEL_PATH
            ),
            ProcessingInput(
                source=feature_engineering_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination=cfg.PROCESSING_TEST_PATH
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation',
                source=cfg.PROCESSING_EVALUATION_PATH,
                destination=cfg.get_s3_path('evaluation')
            )
        ],
        property_files=[evaluation_report]  
    )
    
    print("   ✅ Evaluation step defined")
    
    # ===========================
    # STEP 3: CONDITION CHECK
    # ===========================
    print(f"\n🔧 Defining Condition Step...")
    
    f1_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file='evaluation',
            json_path='metrics.f1_score.value'
        ),
        right=cfg.F1_THRESHOLD
    )
    
    print(f"   ✅ Condition: F1 >= {cfg.F1_THRESHOLD}")
    
    # ===========================
    # STEP 4: MODEL REGISTRATION
    # ===========================
    print(f"\n🔧 Defining Model Registration Step...")
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
            content_type='application/json'
        )
    )
    
    register_step = RegisterModel(
        name='RegisterFlightDelayModel',
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=cfg.INFERENCE_INSTANCE_TYPES,
        transform_instances=cfg.TRANSFORM_INSTANCE_TYPES,
        model_package_group_name=cfg.MODEL_PACKAGE_GROUP,
        approval_status=approval_status,
        model_metrics=model_metrics
    )
    
    print("   ✅ Model registration step defined")
    
    # ===========================
    # STEP 5: CONDITIONAL STEP
    # ===========================
    print(f"\n🔧 Defining Conditional Step...")
    
    condition_step = ConditionStep(
        name='CheckF1Threshold',
        conditions=[f1_condition],
        if_steps=[register_step],
        else_steps=[]
    )
    
    print("   ✅ Conditional step defined")
    
    # ===========================
    # CREATE PIPELINE
    # ===========================
    print(f"\n🏗️  Creating Pipeline...")
    
    pipeline = Pipeline(
        name=cfg.PIPELINE_NAME,
        parameters=[
            # Data and features
            raw_data_url, model_output_path, include_volume_features,
            # Infrastructure
            instance_type, instance_count, approval_status,
            # Hyperparameters
            objective, eval_metric, max_depth, eta, num_round, subsample,
            colsample_bytree, scale_pos_weight, min_child_weight
        ],
        steps=[
            feature_engineering_step,  # Step 0: Feature Engineering
            training_step,              # Step 1: Training
            evaluation_step,            # Step 2: Evaluation
            condition_step              # Step 3: Condition → Registration
        ],
        sagemaker_session=session
    )
    
    return pipeline, role

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    # Upload scripts
    upload_scripts()
    
    # Create pipeline
    pipeline, role = create_pipeline()
    
    # Upsert pipeline to SageMaker
    print(f"\n📤 Uploading pipeline definition to SageMaker...")
    pipeline.upsert(role_arn=role)
    
    print(f"\n{'='*70}")
    print(f"🎉 PIPELINE CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nPipeline Name: {cfg.PIPELINE_NAME}")
    print(f"\n📊 Pipeline Steps:")
    print(f"   0. Feature Engineering (raw parquet → processed CSV)")
    print(f"   1. Training (XGBoost)")
    print(f"   2. Evaluation (F1, Precision, Recall)")
    print(f"   3. Conditional Registration (if F1 >= {cfg.F1_THRESHOLD})")
    print(f"\n✅ View pipeline in SageMaker Studio")
    print(f"✅ Ready to run experiments with run_experiment.py")
    print(f"{'='*70}\n")