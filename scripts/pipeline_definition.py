"""
SageMaker Pipeline Definition for Flight Delay Prediction (v2)
Creates an automated ML pipeline with training, evaluation, and model registration
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

import setting as cfg

# ===========================
# UPLOAD EVALUATION SCRIPT TO S3
# ===========================
def upload_evaluation_script():
    """Upload evaluation script to S3 for use in ProcessingStep"""
    print("📤 Uploading evaluation script to S3...")
    
    s3 = boto3.client('s3')
    
    # Both scripts are in the same directory
    local_script = 'evaluate.py'
    s3_key = f'{cfg.PREFIX}/scripts/evaluate.py'
    
    if not os.path.exists(local_script):
        raise FileNotFoundError(f"Evaluation script not found at {local_script}")
    
    s3.upload_file(local_script, cfg.BUCKET, s3_key)
    print(f"✅ Uploaded to s3://{cfg.BUCKET}/{s3_key}")

# ===========================
# DEFINE PIPELINE
# ===========================
def create_pipeline():
    """Create SageMaker Pipeline"""
    
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
    training_data_url = ParameterString(name="TrainingDataUrl", default_value=cfg.get_s3_path("train"))
    validation_data_url = ParameterString(name="ValidationDataUrl", default_value=cfg.get_s3_path("validation"))
    model_output_path = ParameterString(name="ModelOutputPath", default_value=cfg.get_s3_path("models"))
    input_content_type = ParameterString(name="InputContentType", default_value="application/x-parquet")
    
    # 2. Infrastructure Parameters
    instance_type = ParameterString(name="InstanceType", default_value=cfg.TRAINING_INSTANCE_TYPE)
    instance_count = ParameterInteger(name="InstanceCount", default_value=cfg.TRAINING_INSTANCE_COUNT)
    approval_status = ParameterString(name="ApprovalStatus", default_value=cfg.MODEL_APPROVAL_STATUS)
    
    # 3. Model Hyperparameters
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
    print(f"   - Data Sources (Training, Validation, Output)")
    print(f"   - Infrastructure (Instance Type/Count)")
    print(f"   - Hyperparameters (MaxDepth, Eta, etc.)")
    
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
            'train': TrainingInput(s3_data=training_data_url, content_type=input_content_type),
            'validation': TrainingInput(s3_data=validation_data_url, content_type=input_content_type)
        }
    )
    
    print("   ✅ Training step defined")
    
    # ===========================
    # STEP 2: EVALUATION
    # ===========================
    print(f"🔧 Defining Evaluation Step...")
    
    script_path = f's3://{cfg.BUCKET}/{cfg.PREFIX}/scripts/evaluate.py'
    
    script_processor = ScriptProcessor(
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
    
    # NOTE: We pass the Validation Data as input to the evaluation script
    # This allows us to evaluate on the validation set or a separate test set
    evaluation_step = ProcessingStep(
        name='EvaluateModel',
        processor=script_processor,
        code=script_path,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination=cfg.PROCESSING_MODEL_PATH
            ),
            ProcessingInput(
                source=validation_data_url, # Using validation data for evaluation in pipeline
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
        job_arguments=[
            '--input-content-type', input_content_type 
        ],
        property_files=[evaluation_report]  
    )    
    # ===========================
    # STEP 3: CONDITION CHECK
    # ===========================
    print(f"🔧 Defining Condition Step...")
    
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
    print(f"🔧 Defining Model Registration Step...")
    
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
    print(f"🔧 Defining Conditional Step...")
    
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
            training_data_url, validation_data_url, model_output_path, input_content_type,
            instance_type, instance_count, approval_status,
            objective, eval_metric, max_depth, eta, num_round, subsample,
            colsample_bytree, scale_pos_weight, min_child_weight
        ],
        steps=[training_step, evaluation_step, condition_step],
        sagemaker_session=session
    )
    
    return pipeline, role

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    # Upload evaluation script
    upload_evaluation_script()
    
    # Create pipeline
    pipeline, role = create_pipeline()
    
    # Upsert pipeline to SageMaker
    print(f"\n📤 Uploading pipeline definition to SageMaker...")
    pipeline.upsert(role_arn=role)
    
    print(f"\n{'='*70}")
    print(f"🎉 PIPELINE CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nPipeline Name: {cfg.PIPELINE_NAME}")
    print(f"\n✅ View pipeline in SageMaker Studio")
    print(f"✅ Ready to run experiments with run_experiment.py")
    print(f"{'='*70}\n")
